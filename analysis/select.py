"""Prespecified checkpoint-selection rule (SELECTION_RULE.md), applied to the validation sweep.

    $PY analysis/select.py [--sweep analysis/artifacts/val] [--out analysis/selection/selection.json]

Per run: among post-prior EMA snapshots with validation command tests, choose the EARLIEST step
whose 16 command-test cells (4 limbs x 4 tests) are all >= 0.80. If none qualifies, choose the
step maximizing the minimum cell. Ties -> lower patient-mean body_scale_error, then earlier step.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.80


def select(commands, validate):
    out = {}
    err = validate.set_index("model_id")["body_scale_error"] if len(validate) else pd.Series(dtype=float)
    for run, g in commands[commands["model_id"].str.contains("@")].groupby("run"):
        cells = g.pivot_table(index="step", columns=["limb", "test"], values="fraction")
        complete = cells.dropna()
        if complete.empty:
            continue
        mins = complete.min(axis=1)
        qualifying = mins[mins >= THRESHOLD]
        if len(qualifying):
            step, rule = int(qualifying.index.min()), "earliest step with all cells >= 0.80"
        else:
            best = mins[mins == mins.max()].index
            ids = {s: f"{run}@{s:06d}" for s in best}
            step = int(sorted(best, key=lambda s: (err.get(ids[s], float("inf")), s))[0])
            rule = "no step reached 0.80 in all cells; max-min cell (ties: lower error)"
        out[run] = {"step": step, "rule": rule, "min_cell": float(mins.loc[step]),
                    "n_cells_ge_threshold": int((complete.loc[step] >= THRESHOLD).sum()),
                    "body_scale_error": float(err.get(f"{run}@{step:06d}", float("nan"))),
                    "steps_evaluated": [int(s) for s in complete.index]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(ROOT / "analysis/artifacts/val"))
    ap.add_argument("--out", default=str(ROOT / "analysis/selection/selection.json"))
    args = ap.parse_args()
    tables = Path(args.sweep) / "tables"
    commands = pd.read_csv(tables / "commands.csv")
    validate = pd.read_csv(tables / "validate.csv") if (tables / "validate.csv").exists() else pd.DataFrame()
    result = select(commands, validate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    for run, r in sorted(result.items()):
        print(f"{run:28s} step {r['step']:5d}  min cell {r['min_cell']:.2f}  cells>=0.8 {r['n_cells_ge_threshold']:2d}/16  "
              f"err {r['body_scale_error']:.3f}  ({len(r['steps_evaluated'])} steps)")


if __name__ == "__main__":
    main()
