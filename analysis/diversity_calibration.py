"""Sample diversity and amplitude calibration on the locked test artifacts (CPU only, no sampling).

    $PY analysis/diversity_calibration.py

Diversity: for every test clip and model, the two sampling seeds (31415, 92653) of the clip's own
caption are compared frame by frame on teacher-visible joints in starting-pose body units:
  spread       = mean joint distance between the two samples (body units)
  rel_spread   = spread / mean displacement from frame 0 of the two samples (scale-free)
The hero clip (8 seeds x 13 commands, analysis/artifacts/hero) adds the mean pairwise distance (APD)
over 8 samples per command.
Calibration: does generated motion track the infant's own recorded motion, clip by clip?
  rho_all      Spearman(pred_amplitude, teacher_amplitude) over all clips (seed-mean per clip)
  rho_within   mean Spearman within manual caption type (still, 1, 2, 3-4 limbs), weighted by n:
               adaptation beyond what the caption says
  slope        OLS slope of log(pred) on log(teacher): 1 = proportional, 0 = ignores the infant
All per run, then mean +- s.d. over the three training seeds of each arm.
Output: analysis/artifacts/diversity_calibration/{per_clip.csv, per_run.csv, summary.csv, hero_apd.csv}
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import kinematics as K                                                  # noqa: E402

ART = ROOT / "analysis/artifacts"
OUT = ART / "diversity_calibration"
CAPTION_TYPES = ["still/none", "1 limb", "2 limbs", "3-4 limbs"]
SEEDS = (31415, 92653)


def arm_of(name):
    return name.split("@")[0].rsplit("_s", 1)[0] if "@" in name else name.removeprefix("baseline_")


def model_dirs():
    for folder in ("test_partA", "test_partB"):
        for d in sorted((ART / folder).iterdir()):
            if (d / "validate" / "per_clip.csv").exists() and ("@" in d.name or d.name.startswith("baseline_")):
                yield folder, d


def to_body(xy, t, idx):
    return K.body_units(torch.as_tensor(xy), torch.as_tensor(t["pose0"][idx]), torch.as_tensor(t["visible0"][idx]),
                        torch.as_tensor(t["image_size"][idx])).numpy()


def diversity_rows(d, t, tid, clip_type):
    zs = [np.load(d / "validate" / f"trajectories_seed{s}.npz") for s in SEEDS]
    if any(not (d / "validate" / f"trajectories_seed{s}.npz").exists() for s in SEEDS):
        return []
    ids = list(zs[0]["ids"].astype(str))
    idx = np.array([tid[c] for c in ids])
    a, b = (to_body(z["xy_norm"].astype(np.float64), t, idx)[:, 1:] for z in zs)          # frames 1..19
    valid = t["valid"][idx].astype(bool)                                                   # [N,19,20]
    a0, b0 = (to_body(z["xy_norm"].astype(np.float64), t, idx)[:, :1] for z in zs)
    dist = np.linalg.norm(a - b, axis=-1)
    disp = 0.5 * (np.linalg.norm(a - a0, axis=-1) + np.linalg.norm(b - b0, axis=-1))
    n = valid.sum((1, 2)).clip(min=1)
    spread = (dist * valid).sum((1, 2)) / n
    mean_disp = (disp * valid).sum((1, 2)) / n
    return [{"clip_id": c, "spread": float(s), "displacement": float(m),
             "rel_spread": float(s / m) if m > 1e-6 else np.nan, "caption_type": clip_type[c]}
            for c, s, m in zip(ids, spread, mean_disp)]


def calibration(v):
    g = v.groupby("clip_id").agg(pred=("pred_amplitude", "mean"), teacher=("teacher_amplitude", "mean"),
                                 ct=("caption_type_manual", "first"))
    g = g[(g.pred > 0) & (g.teacher > 0)]
    rho_all = spearmanr(g.pred, g.teacher).statistic if len(g) > 2 and g.pred.std() > 0 else np.nan
    within, weights = [], []
    for ct, h in g.groupby("ct"):
        if len(h) > 5 and h.pred.std() > 0:
            within.append(spearmanr(h.pred, h.teacher).statistic); weights.append(len(h))
    rho_within = float(np.average(within, weights=weights)) if within else np.nan
    slope = np.polyfit(np.log(g.teacher), np.log(g.pred), 1)[0] if len(g) > 2 and g.pred.std() > 0 else np.nan
    return {"rho_all": rho_all, "rho_within": rho_within, "slope": slope}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.load(ART / "test_partA/teacher/teacher.npz")
    tid = {c: i for i, c in enumerate(t["ids"].astype(str))}
    clip_type = pd.read_csv(ART / "test_partA/teacher/clips.csv").set_index("clip_id").caption_type_manual.to_dict()

    clips, runs = [], []
    for folder, d in model_dirs():
        name, arm = d.name, arm_of(d.name)
        v = pd.read_csv(d / "validate" / "per_clip.csv")
        rows = diversity_rows(d, t, tid, clip_type) if "@" in name else []
        for r in rows:
            r.update(model=name, arm=arm)
        clips += rows
        rec = {"model": name, "arm": arm, "split_folder": folder, **calibration(v)}
        if rows:
            df = pd.DataFrame(rows)
            rec["spread"] = df.spread.mean(); rec["rel_spread"] = df.rel_spread.mean()
            for ct in CAPTION_TYPES:
                h = df[df.caption_type == ct]
                rec[f"spread[{ct}]"] = h.spread.mean(); rec[f"rel_spread[{ct}]"] = h.rel_spread.mean()
        runs.append(rec)
        print(f"{name}: rho_within={rec['rho_within']:.2f} slope={rec['slope']:.2f}"
              + (f" rel_spread={rec['rel_spread']:.2f}" if rows else ""), flush=True)

    # teacher reference for calibration columns is trivial (rho=1); the arms are compared to each other
    pd.DataFrame(clips).to_csv(OUT / "per_clip.csv", index=False)
    per_run = pd.DataFrame(runs)
    per_run.to_csv(OUT / "per_run.csv", index=False)
    cols = [c for c in per_run.columns if c not in ("model", "arm", "split_folder")]
    summary = per_run.groupby("arm")[cols].agg(["mean", "std"])
    summary.columns = [f"{a}__{b}" for a, b in summary.columns]
    summary.to_csv(OUT / "summary.csv")

    hero = ART / "hero/trajectories.npz"
    if hero.exists():
        z = np.load(hero); meta = json.loads((ART / "hero/choice.json").read_text())
        i = tid[meta["clip"]]
        rows = []
        for key in z.files:
            if not key.endswith("__xy"):
                continue
            arm, cmd, _ = key.split("__")
            xb = to_body(z[key].astype(np.float64), t, np.full(z[key].shape[0], i))[:, 1:]
            ok = t["valid"][i].astype(bool)
            d = [np.linalg.norm(xb[p] - xb[q], axis=-1)[ok].mean() for p, q in combinations(range(len(xb)), 2)]
            rows.append({"arm": arm, "command": cmd, "apd": float(np.mean(d)), "n_samples": len(xb)})
        pd.DataFrame(rows).to_csv(OUT / "hero_apd.csv", index=False)
    (OUT / "done.json").write_text(json.dumps({"models": len(runs), "clips_rows": len(clips)}, indent=2))
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
