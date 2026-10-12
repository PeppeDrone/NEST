"""Fixed-noise timestep probe (PLAN.md E7 / Fig. 6b): how the denoiser's clean-motion estimate behaves
across diffusion noise levels, with and without the prompt.

    $PY analysis/probe_timesteps.py [--runs "v7_*" "abl_minus_all_*"] [--force]

For each run: EMA snapshots at an early step (first post-prior), the selected step and the last step.
All validation clips, one fixed Gaussian noise draw per clip, timesteps T. Per (run, step, t, condition):
  - coordinate MSE of the x0 estimate (normalized state, teacher-visible joints)  [the training loss]
  - amplitude ratio: mean |x0 residual| / mean |teacher residual| (body units, observed-start joints)
condition = "prompt" (caption conditioning) or "null" (all conditioning dropped, the CFG unconditional
branch). Output: analysis/artifacts/probe/timesteps.csv (+ done.json).
"""
import argparse
import fnmatch
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
from neomotion import data as D                                                    # noqa: E402
from neomotion.common.clip_text import FrozenCLIPTextConditioner                    # noqa: E402
from neomotion.conditioning import encode_prompt_table, prompt_batch                # noqa: E402
from neomotion.diffusion import cosine_schedule                                     # noqa: E402
from neomotion.inference import load_generator, model_inputs, per_clip_noise        # noqa: E402
from neomotion.representation import reference_frame, encode_future                 # noqa: E402
from pipeline import deterministic_backends, DEFAULT_MANIFEST                       # noqa: E402

TIMESTEPS = [0, 50, 100, 250, 400, 500, 600, 750, 900, 999]
NOISE_SEED = 777


def steps_for(run):
    run_dir = ROOT / "runs" / run
    prior = json.loads((run_dir / "resolved_config.json").read_text())["prior_steps"]
    snaps = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "ema_snapshots").glob("step_*.pt"))
    post = [s for s in snaps if s > prior]
    selection = json.loads((ROOT / "analysis/selection/selection.json").read_text())
    chosen = {"early": post[0], "selected": selection[run]["step"], "last": post[-1]}
    return chosen


@torch.no_grad()
def probe(model, scales, data, table, alpha_bar, device, batch_size=16):
    rows = []
    for start in range(0, len(data["ids"]), batch_size):
        batch = D.tensor_batch(data, list(range(start, min(start + batch_size, len(data["ids"])))), device)
        frame = reference_frame(batch["pose0"], batch["visible0"], batch["image_size"])
        target = encode_future(batch["target"], batch["valid"], frame, scales)          # normalized state
        branch = (~batch["visible0"]).long()
        state_scales = scales[branch, torch.arange(20, device=device)[None]]           # [B,20,2]
        noise = per_clip_noise(batch["ids"], NOISE_SEED, device)
        text = prompt_batch(table, batch["prompts"], device)
        observed = (batch["valid"] & batch["visible0"][:, None]).float()                # [B,19,20]
        true_amp = (torch.linalg.vector_norm(target * state_scales[:, None], dim=-1) * observed).sum((1, 2))
        for t in TIMESTEPS:
            ts = torch.full((len(batch["ids"]),), t, device=device, dtype=torch.long)
            a, s = alpha_bar[ts].sqrt()[:, None, None, None], (1 - alpha_bar[ts]).sqrt()[:, None, None, None]
            noisy = a * target + s * noise
            for cond in ("prompt", "null"):
                dropped = torch.full((len(batch["ids"]),), cond == "null", device=device)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with amp:
                    clean, _ = model(noisy, ts, **model_inputs(batch, frame, text, dropped))
                clean = clean.float()
                mse = ((clean - target).square().mean(-1) * batch["valid"]).sum((1, 2)) / batch["valid"].sum((1, 2)).clamp_min(1)
                pred_amp = (torch.linalg.vector_norm(clean * state_scales[:, None], dim=-1) * observed).sum((1, 2))
                for i, clip in enumerate(batch["ids"]):
                    rows.append({"clip_id": clip, "patient_id": batch["patients"][i], "t": t, "alpha_bar": float(alpha_bar[t]),
                                 "condition": cond, "coordinate_mse": float(mse[i]),
                                 "pred_amplitude_sum": float(pred_amp[i]), "teacher_amplitude_sum": float(true_amp[i]),
                                 "observed_tokens": float(observed[i].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["v7_manual_*", "v7_automatic_*", "abl_minus_all_*"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = ROOT / "analysis/artifacts/probe"; out.mkdir(parents=True, exist_ok=True)
    csv = out / "timesteps.csv"
    if csv.exists() and not args.force:
        print(f"exists: {csv} (use --force)"); return
    deterministic_backends()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted(p.name for p in (ROOT / "runs").iterdir() if any(fnmatch.fnmatch(p.name, r) for r in args.runs))
    encoder = FrozenCLIPTextConditioner(local_files_only=True)
    splits, tables, rows = {}, {}, []
    for run in runs:
        for label, step in steps_for(run).items():
            ckpt = ROOT / "runs" / run / "ema_snapshots" / f"step_{step:06d}.pt"
            model, scales, cfg, payload = load_generator(ckpt, device, "ema")
            source = payload["provenance"]["prompt_source"]
            if source not in splits:
                splits[source] = D.load_split(DEFAULT_MANIFEST, source, "val")
                tables[source] = encode_prompt_table(sorted(set(splits[source]["prompts"])), encoder, device, batch_size=1)
            alpha_bar = cosine_schedule(cfg.diffusion_steps, device)
            for r in probe(model, scales, splits[source], tables[source], alpha_bar, device):
                rows.append({"run": run, "arm": run.rsplit("_s", 1)[0], "step": step, "which": label, **r})
            print(f"{run} {label} step {step}: done", flush=True)
            del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(csv, index=False)
    (out / "done.json").write_text(json.dumps({"runs": runs, "timesteps": TIMESTEPS, "noise_seed": NOISE_SEED}, indent=2))
    df = pd.DataFrame(rows)
    summ = (df.groupby(["arm", "which", "condition", "t"])[["pred_amplitude_sum", "teacher_amplitude_sum"]].sum()
              .assign(amplitude_ratio=lambda x: x.pred_amplitude_sum / x.teacher_amplitude_sum)["amplitude_ratio"].unstack("t").round(2))
    print(summ.to_string())


if __name__ == "__main__":
    main()
