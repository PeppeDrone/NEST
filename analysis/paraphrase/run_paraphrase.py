"""Generate the test-split trajectories of NEST manual (seed 42) under original and paraphrased captions.

    CUDA_VISIBLE_DEVICES="" $PY analysis/paraphrase/run_paraphrase.py [--device cpu] [--limit N] [--force]

Conditions: original (each clip's own manual caption), A and B (the clip's caption replaced by
paraphrase A or B from the frozen paraphrases_v1.json). All three conditions are generated in this run,
on the same device, with the pipeline's own generator, per-clip noise (seeds 31415 and 92653), 50 DDIM
steps and guidance 2.0, so the only difference between conditions is the caption text.
Model: v7_manual_s42 at its selected step (2,600), the manual replica with the highest validation
mean command success (stated before generation).
Output: analysis/artifacts/paraphrase/<condition>/{trajectories_seed*.npz, per_clip.csv, kinematics_per_clip_limb.csv}
        analysis/artifacts/paraphrase/run.json (model identity, device, settings, paraphrase SHA-256)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import pipeline as P                                                     # noqa: E402
import kinematics as K                                                   # noqa: E402
from neomotion import data as D                                          # noqa: E402
from neomotion.common.clip_text import FrozenCLIPTextConditioner         # noqa: E402
from neomotion.conditioning import encode_prompt_table                   # noqa: E402
from neomotion.evaluation import trajectory_metrics                      # noqa: E402

OUT = ROOT / "analysis/artifacts/paraphrase"
RUN, STEP = "v7_manual_s42", 2600
SEEDS = (31415, 92653)
SETTINGS = {"cfg_scale": 2.0, "inference_steps": 50, "seeds": list(SEEDS)}


def frozen_paraphrases():
    text = (HERE / "paraphrases_v1.json").read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != (HERE / "paraphrases_v1.sha256").read_text().split()[0]:
        raise SystemExit("paraphrases_v1.json does not match its frozen SHA-256")
    return json.loads(text), sha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, help="first N test clips only (timing / smoke test; written to a _smoke folder)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    payload, sha = frozen_paraphrases()
    by_original = {it["original"]: it for it in payload["items"]}
    out = OUT if not args.limit else OUT / "_smoke"
    out.mkdir(parents=True, exist_ok=True)

    P.deterministic_backends()
    ctx = P.Context({"name": "paraphrase", "split": "test", "output_dir": str(out.relative_to(ROOT) / "_ctx"),
                     "device": args.device}, False)
    spec = {"id": f"{RUN}@{STEP:06d}", "run": RUN, "step": STEP, "kind": "model", "weights": "ema",
            "checkpoint": ROOT / "runs" / RUN / "ema_snapshots" / f"step_{STEP:06d}.pt"}
    gen = P.ModelGenerator(ctx, spec)
    data = ctx.split(gen.source)
    idx = list(range(len(data["ids"])))[: args.limit or None]
    original = [data["prompts"][i] for i in idx]
    conditions = {"original": original,
                  "A": [by_original[p]["paraphrase_A"] for p in original],
                  "B": [by_original[p]["paraphrase_B"] for p in original]}
    # one CLIP table for every prompt, encoded one prompt at a time (batch-independent, as in the pipeline)
    encoder = FrozenCLIPTextConditioner(local_files_only=True)
    ctx._table = encode_prompt_table(sorted({p for v in conditions.values() for p in v}), encoder, args.device, batch_size=1)
    del encoder

    t = np.load(ROOT / "analysis/artifacts/test_partA/teacher/teacher.npz")
    tindex = {c: i for i, c in enumerate(t["ids"])}
    full = D.tensor_batch(data, idx)
    timing = {}
    for name, prompts in conditions.items():
        cdir = out / name; cdir.mkdir(exist_ok=True)
        if (cdir / "per_clip.csv").exists() and not args.force:
            print(f"{name}: exists (use --force)"); continue
        start, rows, kin = time.time(), [], []
        for seed in SEEDS:
            pred = gen(data, idx, prompts, seed, SETTINGS["cfg_scale"], SETTINGS["inference_steps"])
            rows += [dict(r, seed=seed) for r in trajectory_metrics(pred, full)]
            ids = [data["ids"][i] for i in idx]
            np.savez_compressed(cdir / f"trajectories_seed{seed}.npz", ids=P.to_np_str(ids), prompts=P.to_np_str(prompts),
                                xy_norm=pred["xy_norm"].numpy().astype(np.float32), render_visible=pred["render_visible"].numpy())
            ti = [tindex[c] for c in ids]
            valid = np.concatenate([t["visible0"][ti][:, None], t["valid"][ti]], 1)
            xb = K.body_units(pred["xy_norm"].double(), torch.as_tensor(t["pose0"][ti]), torch.as_tensor(t["visible0"][ti]),
                              torch.as_tensor(t["image_size"][ti])).numpy()
            df = P._feature_frame(K.limb_features(xb, valid, t["timestamps"][ti]), ids); df["seed"] = seed
            kin.append(df)
        per_clip = pd.DataFrame(rows)
        per_clip["prompt"] = per_clip.clip_id.map(dict(zip([data["ids"][i] for i in idx], prompts)))
        per_clip["original_prompt"] = per_clip.clip_id.map(dict(zip([data["ids"][i] for i in idx], original)))
        per_clip.to_csv(cdir / "per_clip.csv", index=False)
        pd.concat(kin, ignore_index=True).to_csv(cdir / "kinematics_per_clip_limb.csv", index=False)
        timing[name] = round(time.time() - start, 1)
        print(f"{name}: {len(idx)} clips x {len(SEEDS)} seeds in {timing[name]:.0f} s", flush=True)
    (out / "run.json").write_text(json.dumps({"model": gen.identity, "device": args.device, "settings": SETTINGS,
                                              "paraphrase_file_sha256": sha, "n_clips": len(idx), "seconds": timing,
                                              "torch": torch.__version__}, indent=2, default=str))


if __name__ == "__main__":
    main()
