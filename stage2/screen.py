#!/usr/bin/env python3
"""Create a visual and quantitative Stage-2 screening set from locked manual test outputs.

This is a read-only review of the NEST manual test trajectories.  It does not
run Stage 2 or write controls.  Each candidate is passed through the same
native-20 -> 81-frame temporal bridge logic (with anatomical optimization
disabled solely to make the screening inexpensive).  The rendered sheets show
the 20 native Stage-1 poses, with frame zero restored to the observed pose as
the production bridge does.
"""
from __future__ import annotations

import html
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                      # repository root
OUT = Path(__file__).resolve().parent                            # stage2/: versioned shortlist.csv
LOCAL = OUT / "run" / "screen"                                    # gallery, full screen table (not versioned)
ART = ROOT / "analysis/artifacts/test_partA"
sys.path.insert(0, str(ROOT / "src"))

from dmtbaby.pose.processing import AnatomyLBFGSConfig, process_native20_prediction
from dmtbaby.pose.records import PredictionPoseSequence
from dmtbaby.pose.schema import BODY20_SCOPE


CAPTION_ORDER = ["still/none", "1 limb", "2 limbs", "3-4 limbs"]
REPLICA_NAMES = {
    "v7_manual_s42@002600": "manual replica A",
    "v7_manual_s43@004400": "manual replica B",
    "v7_manual_s44@003000": "manual replica C",
}


def pct_rank_low(values: pd.Series) -> pd.Series:
    """One for the lowest error and zero for the highest error."""
    return 1.0 - values.rank(method="average", pct=True)


def bridge_screen(xy: np.ndarray, visible: np.ndarray, pose0: np.ndarray, visible0: np.ndarray) -> dict[str, float | int | bool | str]:
    """Exercise the locked temporal bridge path without its costly anatomy solve."""
    ref_xy = np.full((62, 2), np.nan, dtype=np.float32)
    ref_visible = np.zeros(62, dtype=bool)
    ref_xy[:20] = pose0
    ref_visible[:20] = visible0
    ref_xy[~ref_visible] = np.nan
    anchored = xy.copy()
    anchored[0] = pose0
    logits = np.where(visible, 20.0, -20.0).astype(np.float32)
    logits[0] = np.where(visible0, 20.0, -20.0)
    try:
        output, diagnostics = process_native20_prediction(
            PredictionPoseSequence(anchored, logits, joint_scope=BODY20_SCOPE),
            ref_xy,
            ref_visible,
            anatomy_config=AnatomyLBFGSConfig(enabled=False),
        )
        finite = np.isfinite(output.xy_norm[:, :20]).all(axis=-1)
        return {
            "bridge_ok": bool(finite[0, visible0].all()),
            "bridge_rejected_outliers": int(diagnostics.rejected_outliers),
            "bridge_interpolated_points": int(diagnostics.interpolated_points),
            "bridge_visible_fraction_81": float(finite.mean()),
            "bridge_error": "",
        }
    except Exception as exc:  # retain a row for every candidate
        return {
            "bridge_ok": False,
            "bridge_rejected_outliers": -1,
            "bridge_interpolated_points": -1,
            "bridge_visible_fraction_81": 0.0,
            "bridge_error": f"{type(exc).__name__}: {exc}",
        }


def native_motion_metrics(xy: np.ndarray, visible: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    valid = visible & np.isfinite(xy).all(axis=-1)
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    step = delta[pair_valid]
    displacement = np.linalg.norm(xy - xy[:1], axis=-1)
    displacement[~valid] = np.nan
    bone_cvs = []
    for a, b in edges:
        mask = valid[:, a] & valid[:, b]
        lengths = np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)[mask]
        if len(lengths) >= 3 and np.nanmean(lengths) > 1e-6:
            bone_cvs.append(float(np.nanstd(lengths) / np.nanmean(lengths)))
    joint_max_displacement = np.array([
        np.nanmax(displacement[:, joint]) if np.isfinite(displacement[:, joint]).any() else np.nan
        for joint in range(displacement.shape[1])
    ])
    return {
        "native_visible_fraction": float(valid.mean()),
        "native_max_step": float(np.max(step)) if len(step) else np.nan,
        "native_p95_step": float(np.percentile(step, 95)) if len(step) else np.nan,
        "native_median_step": float(np.median(step)) if len(step) else np.nan,
        "motion_median_displacement": float(np.nanmedian(joint_max_displacement)),
        "motion_max_displacement": float(np.nanmax(displacement)),
        "bone_length_cv": float(np.mean(bone_cvs)) if bone_cvs else np.nan,
    }


def choose_gallery(rows: pd.DataFrame, limit: int = 24) -> pd.DataFrame:
    """Six high-quality examples per caption type, prioritising patient coverage."""
    eligible = rows[rows.bridge_suitable].sort_values("screen_score", ascending=False)
    chosen: list[int] = []
    used_clips: set[str] = set()
    for caption_type in CAPTION_ORDER:
        candidates = eligible[eligible.caption_type_manual == caption_type]
        selected_for_type: list[int] = []
        observed_patients: set[str] = set()
        for index, row in candidates.iterrows():
            if row.clip_id in used_clips or row.patient_id in observed_patients:
                continue
            selected_for_type.append(index); used_clips.add(row.clip_id); observed_patients.add(row.patient_id)
        for index, row in candidates.iterrows():
            if len(selected_for_type) >= 6:
                break
            if row.clip_id not in used_clips:
                selected_for_type.append(index); used_clips.add(row.clip_id)
        chosen.extend(selected_for_type[:6])
        if len(chosen) >= limit:
            break
    return eligible.loc[chosen].copy().sort_values(["caption_type_manual", "screen_score"], ascending=[True, False])


def draw_sheet(row: pd.Series, xy: np.ndarray, visible: np.ndarray, pose0: np.ndarray, visible0: np.ndarray, edges: np.ndarray, destination: Path) -> None:
    xy = xy.copy(); visible = visible.copy()
    xy[0] = pose0; visible[0] = visible0
    fig, axes = plt.subplots(4, 5, figsize=(11.0, 8.8))
    fig.subplots_adjust(left=0.035, right=0.965, top=0.84, bottom=0.07, wspace=0.28, hspace=0.33)
    for frame, ax in enumerate(axes.flat):
        active = visible[frame] & np.isfinite(xy[frame]).all(axis=-1)
        for a, b in edges:
            if active[a] and active[b]:
                ax.plot(xy[frame, [a, b], 0], xy[frame, [a, b], 1], color="#243447", lw=1.5, solid_capstyle="round")
        ax.scatter(xy[frame, active, 0], xy[frame, active, 1], c="#117a8b", s=11, zorder=3, linewidths=0)
        ax.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"frame {frame:02d}" + (" (observed)" if frame == 0 else ""), fontsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_color("#c6cbd0"); spine.set_linewidth(0.65)
    fig.suptitle(
        f"{row.clip_id} · {row.patient_id} · {row.caption_type_manual} · {row.manual_replica}\n"
        f"{textwrap.fill(str(row.prompt), width=120)}",
        fontsize=10,
        y=0.975,
    )
    fig.text(0.5, -0.01,
             f"Bridge screen: {'PASS' if row.bridge_suitable else 'REVIEW'} | "
             f"outliers removed {int(row.bridge_rejected_outliers)} | gaps filled {int(row.bridge_interpolated_points)} | "
             f"max native step {row.native_max_step:.3f}",
             ha="center", fontsize=8.5)
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    gallery = LOCAL / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    for old_sheet in gallery.glob("*.png"):
        old_sheet.unlink()
    (gallery / "index.html").unlink(missing_ok=True)
    teacher = np.load(ART / "teacher/teacher.npz", allow_pickle=False)
    clip_lookup = pd.read_csv(ART / "teacher/clips.csv").set_index("clip_id")
    teacher_index = {clip_id: i for i, clip_id in enumerate(teacher["ids"].astype(str))}
    edges = teacher["edges"]
    candidates: list[dict[str, object]] = []
    trajectories: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for run_dir in sorted(ART.glob("v7_manual_s*@*")):
        replica = REPLICA_NAMES[run_dir.name]
        metrics = pd.read_csv(run_dir / "validate/per_clip.csv")
        for trajectory_path in sorted((run_dir / "validate").glob("trajectories_seed*.npz")):
            data = np.load(trajectory_path, allow_pickle=False)
            seed = int(trajectory_path.stem.removeprefix("trajectories_seed"))
            metric_rows = metrics[metrics.seed == seed].set_index("clip_id")
            for i, clip_id in enumerate(data["ids"].astype(str)):
                metric = metric_rows.loc[clip_id]
                teacher_i = teacher_index[clip_id]
                xy = data["xy_norm"][i].astype(np.float32)
                visible = data["render_visible"][i].astype(bool)
                pose0 = teacher["pose0"][teacher_i]
                visible0 = teacher["visible0"][teacher_i]
                native = native_motion_metrics(xy, visible, edges)
                bridge = bridge_screen(xy, visible, pose0, visible0)
                row = {
                    "candidate_id": f"{replica[-1]}_{seed}_{clip_id}",
                    "manual_replica": replica,
                    "seed": seed,
                    "clip_id": clip_id,
                    "patient_id": clip_lookup.loc[clip_id, "patient_id"],
                    "caption_type_manual": metric.caption_type_manual,
                    "prompt": metric.prompt,
                    "projected_geometry_error": metric.projected_geometry_error,
                    "body_scale_error": metric.body_scale_error,
                    "first_step_speed_error": metric.first_step_speed_error,
                    "acceleration_error": metric.acceleration_error,
                    "amplitude_log_error": metric.amplitude_log_error,
                    "outside_fraction": metric.outside_fraction,
                    "visibility_f1": metric.visibility_f1,
                    **native,
                    **bridge,
                }
                candidates.append(row)
                trajectories[(replica, seed, clip_id)] = (xy, visible, pose0, visible0)

    screen = pd.DataFrame(candidates)
    metric_quality = pd.concat([
        pct_rank_low(screen.projected_geometry_error),
        pct_rank_low(screen.body_scale_error),
        pct_rank_low(screen.first_step_speed_error),
        pct_rank_low(screen.acceleration_error),
        pct_rank_low(screen.bone_length_cv),
        pct_rank_low(screen.native_max_step),
    ], axis=1).mean(axis=1)
    command_quality = pd.concat([
        pct_rank_low(screen.amplitude_log_error),
        pct_rank_low(screen.native_p95_step),
    ], axis=1).mean(axis=1)
    bridge_clean = ((screen.bridge_rejected_outliers == 0) & (screen.bridge_interpolated_points == 0)).astype(float)
    screen["screen_score"] = 100 * (0.65 * metric_quality + 0.20 * command_quality + 0.15 * bridge_clean)
    screen["bridge_suitable"] = (
        screen.bridge_ok
        & (screen.bridge_rejected_outliers == 0)
        & (screen.bridge_interpolated_points == 0)
        & (screen.outside_fraction == 0)
        & (screen.native_visible_fraction >= 0.95)
    )
    screen = screen.sort_values("screen_score", ascending=False).reset_index(drop=True)
    screen.to_csv(LOCAL / "candidate_screen.csv", index=False)

    selected = choose_gallery(screen)
    selected.insert(0, "gallery_rank", range(1, len(selected) + 1))
    selected.to_csv(OUT / "shortlist.csv", index=False)
    cards = []
    for _, row in selected.iterrows():
        name = f"{int(row.gallery_rank):02d}_{row.candidate_id}.png"
        xy, visible, pose0, visible0 = trajectories[(row.manual_replica, int(row.seed), row.clip_id)]
        draw_sheet(row, xy, visible, pose0, visible0, edges, gallery / name)
        cards.append((name, row))

    summary = {
        "n_candidates": int(len(screen)),
        "n_bridge_suitable": int(screen.bridge_suitable.sum()),
        "n_shortlisted": int(len(selected)),
        "caption_type_counts_shortlist": selected.caption_type_manual.value_counts().to_dict(),
        "patient_counts_shortlist": selected.patient_id.value_counts().to_dict(),
        "screening": {
            "format": "20-frame Body20 NEST manual test trajectory",
            "bridge_path": "locked native20-to-81 temporal bridge with anatomy solve disabled for screening",
            "pass": "bridge executes; no temporal outliers rejected; no gaps filled; no reported outside points; >=95% native visible joints",
            "limitation": "A pass establishes bridge readiness, not visual quality of the final Stage-2 video.",
        },
    }
    (LOCAL / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = selected[["gallery_rank", "clip_id", "patient_id", "caption_type_manual", "manual_replica", "seed", "screen_score", "native_max_step", "motion_median_displacement", "prompt"]].copy()
    table.screen_score = table.screen_score.map("{:.1f}".format)
    table.native_max_step = table.native_max_step.map("{:.3f}".format)
    table.motion_median_displacement = table.motion_median_displacement.map("{:.3f}".format)
    rows_html = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    cards_html = "\n".join(
        f'<article><a href="{html.escape(name)}"><img src="{html.escape(name)}" loading="lazy"></a>'
        f'<p><b>{int(row.gallery_rank):02d}. {html.escape(row.clip_id)}</b><br>{html.escape(row.caption_type_manual)} · {html.escape(row.patient_id)}<br>{html.escape(row.prompt)}</p></article>'
        for name, row in cards
    )
    (gallery / "index.html").write_text(f"""<!doctype html><meta charset=\"utf-8\">
<title>NEST manual test trajectory review</title>
<style>body{{font-family:system-ui;margin:24px;color:#1d2935}} table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccd3d9;padding:5px;text-align:left}}th{{background:#eef3f5}}section{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:20px;margin-top:24px}}article{{border:1px solid #ccd3d9;padding:8px}}img{{width:100%;height:auto}}p{{margin:6px 2px 1px;font-size:13px}}</style>
<h1>NEST manual test trajectories: Stage-2 screening review</h1>
<p>{summary['n_bridge_suitable']} of {summary['n_candidates']} candidates pass the deterministic bridge screen. Each sheet displays all 20 native frames. Frame 00 is the observed reference pose restored by the production bridge.</p>
<p>Pass criteria: bridge executes, no temporal points rejected or gap-filled, no reported outside points, and at least 95% native visible joints. This establishes control-readiness, not final Stage-2 video quality.</p>
<table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in table.columns)}</tr></thead><tbody>{rows_html}</tbody></table>
<section>{cards_html}</section>
""")

    (LOCAL / "README.md").write_text(
        "# Manual test trajectory review for Stage 2\n\n"
        "`gallery/index.html` is the visual starting point. It contains 24 diverse, bridge-ready candidates, each rendered as a 4 × 5 contact sheet of all native Stage-1 frames. `shortlist.csv` identifies those examples; `candidate_screen.csv` records all 1,278 generated manual test trajectories and their reproducible screening fields.\n\n"
        "The screen uses the locked Body20 bridge temporal path: reference anchoring, 20-to-81 resampling, temporal outlier rejection, and short-gap filling. The anatomical solver is disabled only during screening; candidates pass when this path runs without rejected points or filled gaps, with no reported outside points and at least 95% visible native joints. A pass means the pose control is ready to render for Stage 2; it does not evaluate the generated Stage-2 video.\n\n"
        "Run again with:\n\n```bash\nsource scripts/env.sh\n\"$PY\" stage2/screen.py\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
