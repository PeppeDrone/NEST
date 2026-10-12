"""Analyse paraphrase robustness of NEST manual (seed 42) on the test split (CPU).

    $PY analysis/paraphrase/analyze_paraphrase.py

Inputs: analysis/artifacts/paraphrase/{original,A,B}/ (run_paraphrase.py) and
        analysis/paraphrase/parser_check.csv (check_parser.py).
Subsets (per paraphrase version): 'match' = clips whose paraphrase keeps the original limb tokens
(only the CLIP text differs), 'changed' = clips whose paraphrase changes the tokens, 'all'.
Measures, each against the original caption on the same clips, seeds and noise:
  change_ratio   mean joint distance paraphrase-vs-original (same seed) divided by the mean distance
                 between the two noise seeds under the original caption (teacher-visible joints,
                 frames 1-19, starting-pose body units); < 1: wording moves the trajectory less than
                 resampling the noise does
  error          patient-balanced body-scale position error and its change
  amp_ratio      generated/teacher amplitude (sum over clips) by manual caption type
  sel_ratio      single-limb captions: median distal path length of the labelled limb / other limbs
  slope, rho     calibration of generated to recorded amplitude across clips
95% intervals: bootstrap over infants and clips within infants (2,000 draws); n = 5 infants.
Output: analysis/artifacts/paraphrase/analysis/{summary.csv, change_by_type.csv, summary.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import kinematics as K                                                  # noqa: E402
from neomotion.conditioning import parse_prompt                          # noqa: E402

ART = ROOT / "analysis/artifacts/paraphrase"
OUT = ART / "analysis"
SEEDS = (31415, 92653)
CAPTION_TYPES = ["still/none", "1 limb", "2 limbs", "3-4 limbs"]
LIMB_ORDER = ("left_arm", "right_arm", "left_leg", "right_leg")          # conditioning order
N_BOOT = 2000


def body(z, t, tindex):
    ti = [tindex[c] for c in z["ids"]]
    xb = K.body_units(torch.as_tensor(z["xy_norm"]).double(), torch.as_tensor(t["pose0"][ti]),
                      torch.as_tensor(t["visible0"][ti]), torch.as_tensor(t["image_size"][ti])).numpy()
    return xb[:, 1:], t["valid"][ti].astype(bool)                           # frames 1..19, teacher support


def mean_dist(a, b, valid):
    d = np.linalg.norm(a - b, axis=-1)
    return (d * valid).sum((1, 2)) / valid.sum((1, 2)).clip(min=1)


def boot_ci(df, stat, n=N_BOOT, seed=0):
    """Hierarchical bootstrap: infants, then clips within infants."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby("patient_id")}
    patients = list(groups)
    vals = []
    for _ in range(n):
        parts = []
        for p in rng.choice(patients, len(patients)):
            g = groups[p]
            parts.append(g.iloc[rng.integers(0, len(g), len(g))])
        vals.append(stat(pd.concat(parts)))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (np.nan, np.nan)


def labelled_limb(prompt):
    codes = parse_prompt(prompt)[0]
    moving = [l for l, c in zip(LIMB_ORDER, codes) if int(c[0]) == 1 and int(c[1]) == 1]
    return moving[0] if len(moving) == 1 else None


def calibration(g):
    g = g[(g.pred > 0) & (g.teacher > 0)]
    if len(g) < 5 or g.pred.std() == 0:
        return np.nan, np.nan
    return float(np.polyfit(np.log(g.teacher), np.log(g.pred), 1)[0]), float(spearmanr(g.pred, g.teacher).statistic)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.load(ROOT / "analysis/artifacts/test_partA/teacher/teacher.npz")
    tindex = {c: i for i, c in enumerate(t["ids"])}
    clips = pd.read_csv(ROOT / "analysis/artifacts/test_partA/teacher/clips.csv").set_index("clip_id")
    parser = pd.read_csv(HERE / "parser_check.csv")
    match = {(r.original, r.version): bool(r.tokens_match) for r in parser.itertuples()}

    traj = {c: {s: np.load(ART / c / f"trajectories_seed{s}.npz") for s in SEEDS} for c in ("original", "A", "B")}
    per_clip = {c: pd.read_csv(ART / c / "per_clip.csv") for c in ("original", "A", "B")}
    kin = {c: pd.read_csv(ART / c / "kinematics_per_clip_limb.csv") for c in ("original", "A", "B")}
    ids = list(traj["original"][SEEDS[0]]["ids"])
    xo = {s: body(traj["original"][s], t, tindex) for s in SEEDS}
    seed_spread = mean_dist(xo[SEEDS[0]][0], xo[SEEDS[1]][0], xo[SEEDS[0]][1])           # per clip

    # ---------- per-clip table: wording-induced change, error, amplitude, for each version
    rows = []
    for version in ("A", "B"):
        dw = np.mean([mean_dist(body(traj[version][s], t, tindex)[0], xo[s][0], xo[s][1]) for s in SEEDS], axis=0)
        po = per_clip["original"].groupby("clip_id")[["body_scale_error", "pred_amplitude", "teacher_amplitude"]].mean()
        pv = per_clip[version].groupby("clip_id")[["body_scale_error", "pred_amplitude"]].mean()
        prompts = per_clip[version].drop_duplicates("clip_id").set_index("clip_id")
        for i, c in enumerate(ids):
            orig = prompts.loc[c, "original_prompt"]
            rows.append({"clip_id": c, "patient_id": clips.loc[c, "patient_id"], "caption_type": clips.loc[c, "caption_type_manual"],
                         "version": version, "tokens_match": match[(orig, version)], "original_prompt": orig,
                         "paraphrase": prompts.loc[c, "prompt"], "wording_change": dw[i], "seed_spread": seed_spread[i],
                         "error_original": po.loc[c, "body_scale_error"], "error_paraphrase": pv.loc[c, "body_scale_error"],
                         "amp_original": po.loc[c, "pred_amplitude"], "amp_paraphrase": pv.loc[c, "pred_amplitude"],
                         "amp_teacher": po.loc[c, "teacher_amplitude"]})
    clip_df = pd.DataFrame(rows)
    clip_df.to_csv(OUT / "per_clip.csv", index=False)

    # ---------- labelled-limb concentration (single-limb captions, labels from the original caption)
    def sel_ratio(kdf, sub_ids):
        vals = []
        for c in sub_ids:
            limb = labelled_limb(clips.loc[c, "manual_prompt"])
            if limb is None:
                continue
            g = kdf[kdf.clip_id == c].groupby("limb").distal_path_length.mean()
            if limb in g and g.drop(limb).mean() > 0:
                vals.append(g[limb] / max(g.drop(limb).mean(), 1e-3))
        return float(np.nanmedian(vals)) if vals else np.nan

    # ---------- summaries per version x subset
    stat_change = lambda d: d.wording_change.mean() / d.seed_spread.mean()
    stat_derr = lambda d: d.groupby("patient_id").apply(lambda g: (g.error_paraphrase - g.error_original).mean(), include_groups=False).mean()
    stat_dlogamp = lambda d: np.log(d.amp_paraphrase.sum() / d.amp_original.sum())
    summary, by_type = [], []
    for version in ("A", "B", "pooled"):
        for subset in ("all", "match", "changed"):
            d = clip_df if version == "pooled" else clip_df[clip_df.version == version]
            if subset != "all":
                d = d[d.tokens_match == (subset == "match")]
            if d.empty:
                continue
            single = d[d.caption_type == "1 limb"].clip_id.tolist()
            cal_o = calibration(pd.DataFrame({"pred": d.amp_original, "teacher": d.amp_teacher}))
            cal_p = calibration(pd.DataFrame({"pred": d.amp_paraphrase, "teacher": d.amp_teacher}))
            rec = {"version": version, "subset": subset, "n_clips": len(d), "n_patients": d.patient_id.nunique(),
                   "change_ratio": stat_change(d), "change_ratio_ci": boot_ci(d, stat_change),
                   "wording_change": d.wording_change.mean(), "seed_spread": d.seed_spread.mean(),
                   "error_original": d.groupby("patient_id").error_original.mean().mean(),
                   "error_paraphrase": d.groupby("patient_id").error_paraphrase.mean().mean(),
                   "delta_error": stat_derr(d), "delta_error_ci": boot_ci(d, stat_derr, seed=1),
                   "amp_ratio_original": d.amp_original.sum() / d.amp_teacher.sum(),
                   "amp_ratio_paraphrase": d.amp_paraphrase.sum() / d.amp_teacher.sum(),
                   "delta_log_amp": stat_dlogamp(d), "delta_log_amp_ci": boot_ci(d, stat_dlogamp, seed=2),
                   "sel_ratio_original": sel_ratio(kin["original"], single),
                   "sel_ratio_paraphrase": (np.nanmean([sel_ratio(kin[v], d[(d.version == v) & (d.caption_type == "1 limb")].clip_id.tolist())
                                                        for v in ("A", "B")]) if version == "pooled" else sel_ratio(kin[version], single)),
                   "n_single_limb": len(single),
                   "slope_original": cal_o[0], "rho_original": cal_o[1], "slope_paraphrase": cal_p[0], "rho_paraphrase": cal_p[1]}
            summary.append(rec)
            for ct in CAPTION_TYPES:
                e = d[d.caption_type == ct]
                if len(e):
                    by_type.append({"version": version, "subset": subset, "caption_type": ct, "n_clips": len(e),
                                    "change_ratio": stat_change(e), "amp_ratio_original": e.amp_original.sum() / e.amp_teacher.sum(),
                                    "amp_ratio_paraphrase": e.amp_paraphrase.sum() / e.amp_teacher.sum(),
                                    "error_original": e.error_original.mean(), "error_paraphrase": e.error_paraphrase.mean()})
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(by_type).to_csv(OUT / "change_by_type.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    cols = ["version", "subset", "n_clips", "change_ratio", "error_original", "error_paraphrase", "delta_error",
            "amp_ratio_original", "amp_ratio_paraphrase", "sel_ratio_original", "sel_ratio_paraphrase", "slope_original", "slope_paraphrase"]
    print(s[cols].round(3).to_string(index=False))
    print(pd.DataFrame(by_type).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
