"""Prespecified statistics (PLAN.md §5) from saved pipeline artifacts. CPU only.

    $PY analysis/stats.py partA analysis/artifacts/test_partA        # gate: manual vs automatic
    $PY analysis/stats.py partB analysis/artifacts/test_partB --reference analysis/artifacts/test_partA

Unit of inference = patient. Command-test successes are rebuilt per (clip, sampling seed, limb, test)
from commands/rows.csv with exactly the definitions of neomotion.runner.summarize_commands.
Primary endpoint (Part A): mean success over 4 tests x 4 limbs, NEST manual minus NEST automatic.
  - point estimate: mean over patients of (mean over training seeds of the per-patient success)
  - 95% CI: hierarchical bootstrap (patients, then clips within patient, then training seeds), 10,000 draws
  - exact one-sided sign-flip test over patients (2^n_patients permutations of per-patient differences)
Secondary (Holm across them): the four per-test differences, kinematic Fréchet (all clips), and
body-scale error. Part B: each ablation arm minus full NEST manual, effect sizes with the same CIs.
"""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LIMBS = ("right_arm", "left_arm", "right_leg", "left_leg")
TESTS = ("amplitude_order", "speed_order", "moves_more_than_still", "specificity")
RNG_SEED, N_BOOT = 20261007, 10000
warnings.filterwarnings("ignore", "Mean of empty slice")


def arm_of(run):
    return run.rsplit("_s", 1)[0]


def successes(rows):
    """rows.csv of one model -> long DataFrame of binary successes (summarize_commands semantics)."""
    key = {(r.clip_id, r.seed, r.command): r for r in rows.itertuples()}
    out = []
    for r in rows[rows.command == "still"].itertuples():
        for limb in LIMBS:
            amp, spd = f"{limb}_amplitude", f"{limb}_speed"
            if pd.isna(getattr(r, amp)):
                continue
            low, high, fast = (key[(r.clip_id, r.seed, f"{limb}_{c}")] for c in ("low_slow", "high_slow", "high_fast"))
            res = {"amplitude_order": getattr(high, amp) > getattr(low, amp),
                   "speed_order": getattr(fast, spd) > getattr(high, spd),
                   "moves_more_than_still": getattr(fast, spd) > getattr(r, spd)}
            gains = [(getattr(fast, f"{o}_speed") + .01) / (getattr(r, f"{o}_speed") + .01)
                     for o in LIMBS if o != limb and not pd.isna(getattr(r, f"{o}_speed"))]
            gain = (getattr(fast, spd) + .01) / (getattr(r, spd) + .01)
            if gains:
                res["specificity"] = gain > max(gains) and gain > 1
            for test, ok in res.items():
                out.append({"patient_id": r.patient_id, "clip_id": r.clip_id, "seed": r.seed,
                            "limb": limb, "test": test, "success": float(ok)})
    return pd.DataFrame(out)


SELECTION = None   # set from --selection: only each run's selected step is used


def selected(model_dir_name):
    if SELECTION is None:
        return True
    run, step = model_dir_name.split("@")
    return run in SELECTION and int(step) == SELECTION[run]["step"]


def load_arm_successes(artifacts, arm_filter):
    frames = []
    for d in sorted(Path(artifacts).iterdir()):
        if "@" not in d.name or not (d / "commands" / "rows.csv").exists() or not selected(d.name):
            continue
        run = d.name.split("@")[0]
        if not arm_filter(arm_of(run)):
            continue
        s = successes(pd.read_csv(d / "commands" / "rows.csv"))
        s["arm"], s["run"] = arm_of(run), run
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def clip_scores(s, tests=TESTS):
    """Per (arm, run, patient, clip): mean success over the selected tests, limbs and sampling seeds."""
    s = s[s.test.isin(tests)]
    return s.groupby(["arm", "run", "patient_id", "clip_id"]).success.mean().rename("score").reset_index()


def estimate(cs, arm_a, arm_b):
    """Patient-level mean of (arm_a - arm_b), averaging training seeds within arm, clips within patient."""
    per = cs.groupby(["arm", "run", "patient_id"]).score.mean().groupby(["arm", "patient_id"]).mean()
    diff = (per.loc[arm_a] - per.loc[arm_b]).dropna()
    return float(diff.mean()), diff


def bootstrap(cs, arm_a, arm_b, rng):
    """Hierarchical bootstrap: patients, then clips within patient, then training seeds within arm."""
    patients = sorted(cs.patient_id.unique())
    clips = {p: sorted(cs[cs.patient_id == p].clip_id.unique()) for p in patients}
    mats = {}                                           # (arm, patient) -> [n_runs, n_clips] (NaN if missing)
    for a in (arm_a, arm_b):
        runs = sorted(cs[cs.arm == a].run.unique())
        for p in patients:
            g = cs[(cs.arm == a) & (cs.patient_id == p)].pivot_table(index="run", columns="clip_id", values="score")
            mats[a, p] = g.reindex(index=runs, columns=clips[p]).values
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        vals = []
        for p in rng.choice(patients, len(patients)):
            ci = rng.integers(0, len(clips[p]), len(clips[p]))
            arm_means = []
            for a in (arm_a, arm_b):
                m = mats[a, p]
                ri = rng.integers(0, m.shape[0], m.shape[0])
                arm_means.append(np.nanmean(m[np.ix_(ri, ci)]))
            vals.append(arm_means[0] - arm_means[1])
        draws[b] = np.nanmean(vals)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def sign_flip_p(diff):
    """Exact one-sided p for mean(diff) > 0 over all 2^n sign assignments of patient differences."""
    d = diff.values
    obs = d.mean()
    flips = [np.mean(d * np.array(s)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def per_patient_metric(artifacts, arm, column, subdir="validate", file="per_clip.csv"):
    vals = []
    for d in Path(artifacts).iterdir():
        if "@" in d.name and selected(d.name) and arm_of(d.name.split("@")[0]) == arm and (d / subdir / file).exists():
            df = pd.read_csv(d / subdir / file)
            vals.append(df.groupby("patient_id")[column].mean().rename(d.name))
    return pd.concat(vals, axis=1).mean(axis=1)


def contrast(cs, arm_a, arm_b, rng, label, two_sided=False):
    est, diff = estimate(cs, arm_a, arm_b)
    p = min(1.0, 2 * min(sign_flip_p(diff), sign_flip_p(-diff))) if two_sided else sign_flip_p(diff)
    return {"contrast": label, "estimate": est, "ci95": bootstrap(cs, arm_a, arm_b, rng),
            "p_sign_flip_one_sided" if not two_sided else "p_sign_flip_two_sided": p,
            "per_patient": diff.round(4).to_dict(), "n_patients": len(diff)}


def part_a(args):
    rng = np.random.default_rng(RNG_SEED)
    s = load_arm_successes(args.artifacts, lambda a: a in ("v7_manual", "v7_automatic"))
    result = {"primary": contrast(clip_scores(s), "v7_manual", "v7_automatic", rng, "manual - automatic, all 16 cells")}
    secondary = [contrast(clip_scores(s, (t,)), "v7_manual", "v7_automatic", rng, f"manual - automatic, {t}") for t in TESTS]
    # Body-scale error (lower is better): automatic - manual, so positive favours manual.
    em, ea = (per_patient_metric(args.artifacts, a, "body_scale_error") for a in ("v7_manual", "v7_automatic"))
    d = (ea - em).dropna()
    secondary.append({"contrast": "body_scale_error, automatic - manual (positive favours manual)",
                      "estimate": float(d.mean()), "p_sign_flip_one_sided": sign_flip_p(d), "per_patient": d.round(4).to_dict()})
    adj = holm([c["p_sign_flip_one_sided"] for c in secondary])
    for c, a in zip(secondary, adj):
        c["p_holm"] = a
    result["secondary"] = secondary
    result["absolute"] = {a: float(clip_scores(s[s.arm == a]).groupby("patient_id").score.mean().mean())
                          for a in ("v7_manual", "v7_automatic")}
    result["gate_passed"] = bool(result["primary"]["estimate"] > 0 and result["primary"]["ci95"][0] > 0)
    result["note"] = ("With n patients the smallest attainable exact one-sided p is 1/2^n; the CI and the "
                      "per-patient differences carry the evidence.")
    write(args, result, "partA")


def part_b(args):
    rng = np.random.default_rng(RNG_SEED)
    ab = load_arm_successes(args.artifacts, lambda a: a.startswith("abl_"))
    ref = load_arm_successes(args.reference, lambda a: a == "v7_manual")
    s = pd.concat([ab, ref], ignore_index=True)
    arms = sorted(ab.arm.unique())
    result = {"effects_vs_v7_manual": []}
    for arm in arms:
        for tests, name in [(TESTS, "all 16 cells")] + [((t,), t) for t in TESTS]:
            result["effects_vs_v7_manual"].append(contrast(clip_scores(s, tests), arm, "v7_manual", rng, f"{arm} - v7_manual, {name}", two_sided=True))
    result["note"] = "Explanatory (PLAN.md §5): effect sizes with CIs; no multiplicity-adjusted claims."
    write(args, result, "partB")


def write(args, result, name):
    out = Path(args.artifacts) / ("stats_selected" if SELECTION else "stats"); out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    lines = [f"# {name} statistics ({args.artifacts})\n"]
    def row(c):
        ci = c.get("ci95")
        return (f"| {c['contrast']} | {c['estimate']:+.3f} | " + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "–") +
                f" | {c.get('p_sign_flip_one_sided', c.get('p_sign_flip_two_sided')):.3f} | " + (f"{c['p_holm']:.3f}" if "p_holm" in c else "–") + " |")
    sided = "2-sided" if name == "partB" else "1-sided"
    lines += [f"| contrast | estimate | 95% CI | p (sign-flip, {sided}) | p (Holm) |", "|---|---|---|---|---|"]
    for c in [result.get("primary")] + result.get("secondary", []) + result.get("effects_vs_v7_manual", []):
        if c:
            lines.append(row(c))
    if "gate_passed" in result:
        lines += ["", f"Absolute mean success: {json.dumps({k: round(v, 3) for k, v in result['absolute'].items()})}",
                  f"**Gate (primary estimate > 0 and CI excludes 0): {'PASSED' if result['gate_passed'] else 'NOT PASSED'}**"]
    lines += ["", result["note"]]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["partA", "partB"])
    ap.add_argument("artifacts")
    ap.add_argument("--reference", help="Part B: artifacts folder holding the NEST manual models (test_partA)")
    ap.add_argument("--selection", help="selection.json; restrict to each run's selected step (needed for the "
                                        "validation sweep, which holds every snapshot)")
    args = ap.parse_args()
    global SELECTION
    if args.selection:
        SELECTION = json.loads(Path(args.selection).read_text())
    part_a(args) if args.part == "partA" else part_b(args)


if __name__ == "__main__":
    main()
