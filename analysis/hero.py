"""Single-subject generation for the hero figure (PLAN.md Fig. 4). Run after test_partA.

    $PY analysis/hero.py [--seeds 8] [--force]

Subject rule (stated, not chosen by appearance):
  1. patient = the test patient with the median per-patient NEST-manual command success (test_partA,
     mean over the 3 training seeds' selected checkpoints; ties -> lower patient id);
  2. clip = among that patient's clips with all 12 limb joints observed at frame 0, the clip whose
     NEST-manual success is the median (ties -> lower clip id).
Models: the selected seed-42 checkpoints of NEST manual and NEST automatic (selection.json).
Generates the 13 commands + the clip's own caption (manual and automatic caption) x N sampling seeds.
Output: analysis/artifacts/hero/{choice.json, trajectories.npz}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import pipeline as P                                       # noqa: E402
from stats import successes, clip_scores                   # noqa: E402

LIMB_JOINTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def choose(test_dir):
    frames = []
    for d in sorted(test_dir.iterdir()):
        if "@" in d.name and d.name.startswith("v7_manual") and (d / "commands/rows.csv").exists():
            s = successes(pd.read_csv(d / "commands/rows.csv")); s["arm"], s["run"] = "v7_manual", d.name.split("@")[0]
            frames.append(s)
    cs = clip_scores(pd.concat(frames)).groupby(["patient_id", "clip_id"]).score.mean().reset_index()
    per_patient = cs.groupby("patient_id").score.mean().sort_values(kind="stable")
    ordered = per_patient.reset_index().sort_values(["score", "patient_id"])
    patient = ordered.iloc[(len(ordered) - 1) // 2].patient_id
    t = np.load(test_dir / "teacher/teacher.npz")
    full = {c for c, v in zip(t["ids"], t["visible0"]) if v[LIMB_JOINTS].all()}
    cand = cs[(cs.patient_id == patient) & cs.clip_id.isin(full)].sort_values(["score", "clip_id"])
    if cand.empty:
        raise SystemExit(f"No fully observed clip for {patient}")
    clip = cand.iloc[(len(cand) - 1) // 2].clip_id
    return patient, clip, per_patient.to_dict(), float(cand[cand.clip_id == clip].score.iloc[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    test_dir = ROOT / "analysis/artifacts/test_partA"
    out = ROOT / "analysis/artifacts/hero"; out.mkdir(parents=True, exist_ok=True)
    if (out / "trajectories.npz").exists() and not args.force:
        print("exists (use --force)"); return
    patient, clip, per_patient, clip_score = choose(test_dir)
    selection = json.loads((ROOT / "analysis/selection/selection.json").read_text())
    P.deterministic_backends()
    ctx = P.Context({"name": "hero", "split": "test", "output_dir": "analysis/artifacts/hero/_ctx"}, False)
    seeds = list(range(1000, 1000 + args.seeds))
    arrays, meta = {}, {"patient": patient, "clip": clip, "per_patient_manual_success": per_patient,
                        "clip_manual_success": clip_score, "seeds": seeds, "models": {}, "prompts": {}}
    commands = P.command_set()
    for arm in ("v7_manual", "v7_automatic"):
        run = f"{arm}_s42"
        step = selection[run]["step"]
        spec = {"id": f"{run}@{step:06d}", "run": run, "step": step, "kind": "model", "weights": "ema",
                "checkpoint": ROOT / "runs" / run / "ema_snapshots" / f"step_{step:06d}.pt"}
        gen = P.ModelGenerator(ctx, spec)
        data = ctx.split(gen.source)
        i = data["ids"].index(clip)
        prompts = dict(commands, own_caption=data["prompts"][i])
        meta["models"][arm] = gen.identity; meta["prompts"][arm] = prompts
        for name, prompt in prompts.items():
            xs, vs = [], []
            for seed in seeds:
                pred = gen(data, [i], [prompt], seed, 2.0, 50)
                xs.append(pred["xy_norm"][0].numpy()); vs.append(pred["render_visible"][0].numpy())
            arrays[f"{arm}__{name}__xy"] = np.stack(xs).astype(np.float32)     # [seeds,20,20,2]
            arrays[f"{arm}__{name}__vis"] = np.stack(vs)
        gen.close()
        print(f"{arm}: {len(prompts)} prompts x {len(seeds)} seeds", flush=True)
    np.savez_compressed(out / "trajectories.npz", **arrays)
    (out / "choice.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("patient", "clip", "clip_manual_success")}, indent=2))


if __name__ == "__main__":
    main()
