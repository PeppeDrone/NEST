"""Single command-line surface for the coordinate-to-UniAnimate cascade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _root_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Project root.")


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and describe work without writing any artifact.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmtbaby",
        description="Joint-coordinate diffusion → DWPose RGB → UniAnimate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Run hard DGX-Spark readiness gates.")
    _add_root(preflight)
    preflight.add_argument("--min-free-gib", type=float, default=100.0)
    preflight.add_argument("--min-memory-gib", type=float, default=90.0)
    preflight.add_argument(
        "--allow-cpu-onnx",
        action="store_true",
        help="Permit CPU ONNX only for an explicitly requested smoke check.",
    )
    preflight.add_argument("--skip-onnx-probe", action="store_true")
    preflight.add_argument("--verify-model-hashes", action="store_true")
    preflight.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    preflight.set_defaults(handler=_cmd_preflight)

    extract = subparsers.add_parser(
        "extract-keypoints",
        help="Extract atomic native-20 canonical teacher trajectories.",
    )
    _add_root(extract)
    _add_dry_run(extract)
    extract.add_argument("--manifest", default="Data/processed/dataset_manifest.csv")
    extract.add_argument("--output-root", default="Data2/dwpose_woface_62_v1")
    extract.add_argument(
        "--detector-model",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/yolox_l.onnx",
    )
    extract.add_argument(
        "--pose-model",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/dw-ll_ucoco_384.onnx",
    )
    extract.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    extract.add_argument("--allow-cpu-smoke", action="store_true")
    extract.add_argument("--det-threshold", type=float, default=0.3)
    extract.add_argument("--pose-threshold", type=float, default=0.3)
    extract.add_argument("--clip-id", action="append", default=[])
    extract.add_argument("--max-samples", type=int)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument(
        "--matched-prompts",
        default="Data/processed/pairs_stage1_intersection_manifest.csv",
    )
    extract.add_argument(
        "--split-manifest", default="reproducibility/patient_split_v1.json"
    )
    extract.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    extract.set_defaults(handler=_cmd_extract)

    train1 = subparsers.add_parser("train-stage1", help="Train coordinate diffusion Stage 1.")
    _add_root(train1)
    _add_dry_run(train1)
    train1.add_argument("--stage1-mode", choices=("coordinate", "legacy_binary_mask"), default="coordinate")
    train1.add_argument("--teacher-manifest", default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl")
    train1.add_argument("--output-dir", required=True)
    train1.add_argument("--prompt-source", choices=("manual", "automatic"), default="manual")
    train1.add_argument("--joint-scope", choices=("body20", "whole62"), default="whole62")
    train1.add_argument("--body20-checkpoint")
    train1.add_argument("--resume")
    train1.add_argument("--device", default="cuda")
    train1.add_argument("--seed", type=int, default=42)
    train1.add_argument("--microbatch", type=int, default=16)
    train1.add_argument("--gradient-accumulation", type=int, default=4)
    train1.add_argument("--max-steps", type=int)
    train1.add_argument("--validation-every", type=int, default=500)
    train1.add_argument("--patience-validations", type=int, default=20)
    train1.add_argument("--log-every", type=int, default=1)
    train1.add_argument("--val-example-count", type=int, default=4)
    train1.add_argument("--local-files-only", action="store_true")
    train1.add_argument(
        "--patient-split-manifest",
        default="reproducibility/patient_split_v1.json",
    )
    train1.add_argument(
        "--smoke-test",
        action="store_true",
        help="Explicitly permit reduced steps/batches without satisfying publication gates.",
    )
    train1.add_argument(
        "--improved-moving",
        action="store_true",
        help=(
            "Moving-labelled recipe that matches labelled joints frame by frame, "
            "saves a checkpoint and refreshes loss_terms.png every 200 steps, "
            "and keeps best.pt by the labelled-limb angle-range ratio on the "
            "validation strips. Patience is 5 validations. Does not overwrite "
            "existing Stage-1 checkpoints."
        ),
    )
    train1.add_argument(
        "--time-local-motion",
        dest="time_local_motion",
        action="store_true",
        help=(
            "Weight each labelled joint's coordinate error by how far the teacher "
            "has moved from frame 0, and replace the max-min amplitude term with "
            "a per-frame elbow/knee angle loss."
        ),
    )
    train1.add_argument(
        "--model-v2",
        dest="model_v2",
        action="store_true",
        help=(
            "Stage-1 2.0 backbone: frame 0 is cross-attention context only, "
            "invalid frame-0 joints are masked, each head has its own skeleton "
            "bias including contralateral limbs, limb moving/amplitude/speed "
            "tokens are concatenated with CLIP, and the diffusion timestep "
            "scales each block. Warm-start from a v1 --init-checkpoint copies "
            "matching weights and leaves the new parameters initialized."
        ),
    )
    train1.add_argument(
        "--regression",
        action="store_true",
        help=(
            "Train the v2 backbone to predict clean coordinates in one forward "
            "pass. Disables the noise schedule and DDIM sampler. Implies --model-v2."
        ),
    )
    train1.add_argument(
        "--skeleton-loss",
        dest="skeleton_loss",
        action="store_true",
        help=(
            "v3 recipe: v2 backbone from scratch on the moving clips. The loss is "
            "diffusion, all 20 joint coordinates, visibility, and bone length, "
            "each at weight 1. No elbow/knee angle, velocity, or acceleration. "
            "best.pt is the lowest validation loss, with patience 20. Uses one "
            "microbatch of 64 instead of four microbatches of 16."
        ),
    )
    train1.add_argument(
        "--model-v4",
        dest="model_v4",
        action="store_true",
        help=(
            "v4 recipe: v2 regression backbone from scratch on the moving clips. "
            "The head predicts four elbow/knee flexions over 19 frames and "
            "rebuilds XY by forward kinematics from the observed bone lengths. "
            "Loss is labelled per-frame angle plus unlabelled stillness. "
            "best.pt is the lowest labelled angle MAE. 10,000 steps, patience 20. "
            "Implies --regression and --model-v2. No warm-start, diffusion, "
            "or bone/velocity/acceleration terms."
        ),
    )
    train1.add_argument(
        "--uniform-loss",
        dest="uniform_loss",
        action="store_true",
        help=(
            "Set every Stage-1 loss multiplier to 1, including bone length, "
            "velocity, acceleration, confidence, and labelled-joint terms. "
            "High/fast prompt grades are not upweighted. Cannot be combined "
            "with --drop-bone-velocity-accel."
        ),
    )
    train1.add_argument(
        "--drop-bone-velocity-accel",
        dest="drop_bone_velocity_acceleration",
        action="store_true",
        help=(
            "Remove bone length, velocity, acceleration, labelled velocity, and "
            "labelled acceleration from the Stage-1 loss. Projected limb length "
            "changes when the infant moves in depth."
        ),
    )
    train1.add_argument(
        "--motion-recipe",
        action="store_true",
        help=(
            "Train and validate on moving-labelled clips and add labelled "
            "amplitude/velocity/acceleration losses. Writes a new --output-dir; "
            "does not overwrite existing Stage-1 checkpoints."
        ),
    )
    train1.add_argument(
        "--moving-labelled-only",
        action="store_true",
        help="Restrict train/val sampling to clips with at least one moving limb.",
    )
    train1.add_argument(
        "--labelled-motion-loss",
        action="store_true",
        help="Add amplitude/velocity/acceleration losses on prompt-marked moving joints.",
    )
    train1.add_argument(
        "--annotation-csv",
        help=(
            "Manual annotation CSV used by --motion-recipe. Defaults to "
            "Data/processed/manual_annotations_normalized.csv when the recipe is on."
        ),
    )
    train1.add_argument(
        "--init-checkpoint",
        help=(
            "Warm-start model and EMA weights from an existing same-scope checkpoint. "
            "Does not restore optimizer or RNG; use a new --output-dir."
        ),
    )
    train1.add_argument(
        "legacy_argv",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded only when --stage1-mode legacy_binary_mask.",
    )
    train1.set_defaults(handler=_cmd_train_stage1)

    predict1 = subparsers.add_parser("predict-stage1", help="Sample native-20 joint coordinates.")
    _add_root(predict1)
    _add_dry_run(predict1)
    predict1.add_argument("--stage1-mode", choices=("coordinate", "legacy_binary_mask"), default="coordinate")
    predict1.add_argument("--teacher-manifest", default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl")
    predict1.add_argument("--checkpoint", required=True)
    predict1.add_argument("--output-root", required=True)
    predict1.add_argument("--prompt-source", choices=("manual", "automatic"), default="manual")
    predict1.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    predict1.add_argument("--clip-id", action="append", default=[])
    predict1.add_argument("--device", default="cuda")
    predict1.add_argument("--seed", type=int, default=42)
    predict1.add_argument("--steps", type=int, default=50)
    predict1.add_argument("--cfg-scale", type=float, default=1.5)
    predict1.add_argument("--microbatch", type=int, default=16)
    predict1.add_argument("--local-files-only", action="store_true")
    predict1.add_argument(
        "--smoke-test",
        action="store_true",
        help="Explicitly permit CPU/reduced prediction for tests only.",
    )
    predict1.add_argument(
        "--overwrite",
        action="store_true",
        help="Archive stale per-sample predictions before replacement.",
    )
    predict1.add_argument("legacy_argv", nargs=argparse.REMAINDER)
    predict1.set_defaults(handler=_cmd_predict_stage1)

    render = subparsers.add_parser(
        "render-controls", help="Postprocess predictions and render exact DWPose RGB controls."
    )
    _add_root(render)
    _add_dry_run(render)
    render.add_argument("--prediction-manifest", required=True)
    render.add_argument("--teacher-manifest", default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl")
    render.add_argument("--output-root", required=True)
    render.add_argument("--unianimate-dir", default="external/UniAnimate-DiT")
    render.add_argument("--postprocess", choices=("constrained", "raw"), default="constrained")
    render.add_argument("--persist", choices=("all", "test", "none"), default="test")
    render.add_argument("--clip-id", action="append", default=[])
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--jpeg-quality", type=int, default=95)
    render.add_argument(
        "--smoke-test",
        action="store_true",
        help="Explicitly permit a reduced bridge test without production disk gates.",
    )
    render.set_defaults(handler=_cmd_render_controls)

    train2 = subparsers.add_parser(
        "train-stage2-adapter", help="Fine-tune only DWPose embedding and rank-128 DiT LoRA."
    )
    _add_root(train2)
    _add_dry_run(train2)
    train2.add_argument("--manifest", required=True, help="Manual whole62 constrained training manifest.")
    train2.add_argument(
        "--patient-split-manifest",
        default="reproducibility/patient_split_v1.json",
    )
    train2.add_argument("--output-dir", required=True)
    train2.add_argument(
        "--official-adapter",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    )
    train2.add_argument("--resume")
    train2.add_argument("--unianimate-dir", default="external/UniAnimate-DiT")
    train2.add_argument(
        "--wan-dir",
        default="external/UniAnimate-DiT/Wan2.1-I2V-14B-720P",
    )
    train2.add_argument("--device", default="cuda")
    train2.add_argument("--seed", type=int, default=42)
    train2.set_defaults(handler=_cmd_train_stage2)

    infer2 = subparsers.add_parser("infer-stage2", help="Run strict batched UniAnimate inference.")
    _add_root(infer2)
    _add_dry_run(infer2)
    infer2.add_argument("--manifest", help="Strict Stage-2 JSONL requests for batched inference.")
    control_input = infer2.add_mutually_exclusive_group()
    control_input.add_argument(
        "--control-dir",
        help="Strict direct input containing exactly 0000.jpg through 0080.jpg.",
    )
    control_input.add_argument(
        "--pose-dir",
        help="Legacy name-only alias for --control-dir; strict 81-frame validation still applies.",
    )
    infer2.add_argument("--reference-pose", help="Direct-mode reference pose; defaults to CONTROL_DIR/pose.jpg.")
    infer2.add_argument("--reference-image", help="Direct-mode RGB reference image.")
    infer2.add_argument("--output", help="Direct-mode generated video path.")
    infer2.add_argument("--clip-id", default="direct")
    infer2.add_argument(
        "--prompt",
        default="a realistic NICU video of a newborn baby lying in a cradle",
    )
    infer2.add_argument("--negative-prompt")
    infer2.add_argument("--seed", type=int, default=0)
    infer2.add_argument("--height", type=int, default=832)
    infer2.add_argument("--width", type=int, default=480)
    infer2.add_argument("--num-inference-steps", type=int, default=25)
    infer2.add_argument("--cfg-scale", type=float, default=1.5)
    infer2.add_argument("--fps", type=int, default=15)
    infer2.add_argument("--quality", type=int, default=5)
    infer2.add_argument("--use-teacache", action="store_true")
    infer2.add_argument(
        "--persistent-dit-params",
        type=int,
        default=6_000_000_000,
        help="DiT parameters kept resident on CUDA; use 0 for the low-memory path.",
    )
    infer2.add_argument("--unianimate-dir", default="external/UniAnimate-DiT")
    infer2.add_argument("--wan-dir", default="external/UniAnimate-DiT/Wan2.1-I2V-14B-720P")
    infer2.add_argument(
        "--official-adapter",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
        help="Official rank-128 initializer and frozen reference-pose encoder.",
    )
    infer2.add_argument(
        "--adapter",
        help="Optional compact accepted adapter layered on the official initializer.",
    )
    infer2.add_argument("--device", default="cuda")
    infer2.add_argument("--overwrite", action="store_true")
    infer2.set_defaults(handler=_cmd_infer_stage2)

    cascade = subparsers.add_parser(
        "run-cascade", help="Run Stage 1, bridge, then Stage 2 in separate processes."
    )
    _add_root(cascade)
    _add_dry_run(cascade)
    cascade.add_argument("--manifest", required=True)
    cascade.add_argument("--runs-root", default="Runs")
    cascade.add_argument("--run-id", required=True)
    cascade.add_argument("--stage1-checkpoint", required=True)
    cascade.add_argument(
        "--stage2-official-adapter",
        default="external/UniAnimate-DiT/models/UniAnimate-DiT/UniAnimate-Wan2.1-14B-Lora-12000.ckpt",
    )
    cascade.add_argument(
        "--stage2-adapter",
        help="Compact best/last adapter; required only with --stage2-mode adapted.",
    )
    cascade.add_argument("--prompt-source", choices=("manual", "automatic"), default="manual")
    cascade.add_argument("--joint-scope", choices=("whole62", "body20"), default="whole62")
    cascade.add_argument("--postprocess", choices=("constrained", "raw"), default="constrained")
    cascade.add_argument("--stage2-mode", choices=("frozen", "adapted"), default="frozen")
    cascade.add_argument("--stage1-mode", choices=("coordinate", "legacy_binary_mask"), default="coordinate")
    cascade.add_argument("--seed", type=int, default=42)
    cascade.add_argument("--skip-preflight", action="store_true")
    cascade.set_defaults(handler=_cmd_run_cascade)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate named cascade conditions.")
    _add_root(evaluate)
    _add_dry_run(evaluate)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bootstrap-draws", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.set_defaults(handler=_cmd_evaluate)

    plot_curves = subparsers.add_parser(
        "plot-stage1-curves",
        help="Compare manual vs automatic Stage-1 train/val loss curves.",
    )
    _add_root(plot_curves)
    plot_curves.add_argument(
        "--manual-dir",
        default="Runs/models/stage1_manual_body20",
        help="Manual Stage-1 run directory (contains metrics.jsonl).",
    )
    plot_curves.add_argument(
        "--automatic-dir",
        default="Runs/models/stage1_automatic_body20",
        help="Automatic Stage-1 run directory (contains metrics.jsonl).",
    )
    plot_curves.add_argument(
        "--output",
        default="Runs/figures/stage1_manual_vs_automatic_body20_curves.png",
        help="Destination image path (.png/.pdf).",
    )
    plot_curves.add_argument(
        "--train-smooth-window",
        type=int,
        default=50,
        help="Rolling mean window over train loss (1 disables smoothing).",
    )
    plot_curves.set_defaults(handler=_cmd_plot_stage1_curves)

    manuscript = subparsers.add_parser(
        "plot-stage1-manuscript",
        help="Write Stage-1 teacher/pred strips, error plots, and keypoint videos.",
    )
    _add_root(manuscript)
    manuscript.add_argument(
        "--prediction-root",
        required=True,
        help="Root from predict-stage1 (contains predictions/*.npz).",
    )
    manuscript.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    manuscript.add_argument(
        "--output-dir",
        default="Runs/figures/stage1_manual_body20_manuscript",
    )
    manuscript.add_argument(
        "--unianimate-dir",
        default="external/UniAnimate-DiT",
    )
    manuscript.add_argument(
        "--clip-id",
        action="append",
        default=[],
        help="Clip id to include (repeatable). Defaults to the manuscript set.",
    )
    manuscript.add_argument("--fps", type=float, default=4.0)
    manuscript.add_argument("--confidence-threshold", type=float, default=0.5)
    manuscript.set_defaults(handler=_cmd_plot_stage1_manuscript)

    manuscript_report = subparsers.add_parser(
        "stage1-manuscript-report",
        help=(
            "One-shot Stage-1 manuscript package: predict 10 train + 10 val + 10 test "
            "(defaults), write figures/videos, and mean±std error tables."
        ),
    )
    _add_root(manuscript_report)
    manuscript_report.add_argument(
        "--checkpoint",
        default="Runs/models/stage1_manual_body20/best.pt",
    )
    manuscript_report.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    manuscript_report.add_argument(
        "--prediction-root",
        default="Runs/predictions/manual20_manuscript_report",
    )
    manuscript_report.add_argument(
        "--output-dir",
        default="Runs/figures/stage1_manual_body20_manuscript_report",
    )
    manuscript_report.add_argument(
        "--unianimate-dir",
        default="external/UniAnimate-DiT",
    )
    manuscript_report.add_argument("--prompt-source", choices=("manual", "automatic"), default="manual")
    manuscript_report.add_argument("--n-train", type=int, default=10)
    manuscript_report.add_argument("--n-val", type=int, default=10)
    manuscript_report.add_argument("--n-test", type=int, default=10)
    manuscript_report.add_argument("--device", default="cuda")
    manuscript_report.add_argument("--seed", type=int, default=42)
    manuscript_report.add_argument("--fps", type=float, default=4.0)
    manuscript_report.add_argument("--confidence-threshold", type=float, default=0.5)
    manuscript_report.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse existing predictions under --prediction-root.",
    )
    manuscript_report.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only compute/write error_summary.{json,md}.",
    )
    manuscript_report.set_defaults(handler=_cmd_stage1_manuscript_report)

    full_errors = subparsers.add_parser(
        "stage1-full-split-errors",
        help=(
            "Predict all train+val+test subjects with a Stage-1 checkpoint and write "
            "mean±std error artifacts for the full splits."
        ),
    )
    _add_root(full_errors)
    full_errors.add_argument(
        "--checkpoint",
        default="Runs/models/stage1_manual_body20/best.pt",
    )
    full_errors.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    full_errors.add_argument(
        "--prediction-root",
        default="Runs/predictions/manual20_full_splits",
    )
    full_errors.add_argument(
        "--output-dir",
        default="Runs/metrics/stage1_manual_body20_full_split_errors",
    )
    full_errors.add_argument(
        "--prompt-source", choices=("manual", "automatic"), default="manual"
    )
    full_errors.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to score (default: train,val,test).",
    )
    full_errors.add_argument("--device", default="cuda")
    full_errors.add_argument("--seed", type=int, default=42)
    full_errors.add_argument("--microbatch", type=int, default=16)
    full_errors.add_argument("--confidence-threshold", type=float, default=0.5)
    full_errors.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse existing predictions under --prediction-root.",
    )
    full_errors.set_defaults(handler=_cmd_stage1_full_split_errors)

    body20_compare = subparsers.add_parser(
        "plot-stage1-body20-comparison",
        help=(
            "Compare manual vs automatic body20 training curves and full-split "
            "errors (train/val/test), including limb-group and per-joint plots."
        ),
    )
    _add_root(body20_compare)
    body20_compare.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    body20_compare.add_argument(
        "--manual-checkpoint",
        default="Runs/models/stage1_manual_body20/best.pt",
    )
    body20_compare.add_argument(
        "--automatic-checkpoint",
        default="Runs/models/stage1_automatic_body20/best.pt",
    )
    body20_compare.add_argument(
        "--manual-prediction-root",
        default="Runs/predictions/manual20_full_splits",
    )
    body20_compare.add_argument(
        "--automatic-prediction-root",
        default="Runs/predictions/automatic20_full_splits",
    )
    body20_compare.add_argument(
        "--manual-metrics-dir",
        default="Runs/models/stage1_manual_body20",
    )
    body20_compare.add_argument(
        "--automatic-metrics-dir",
        default="Runs/models/stage1_automatic_body20",
    )
    body20_compare.add_argument(
        "--output-dir",
        default="Runs/figures/stage1_body20_manual_vs_automatic",
    )
    body20_compare.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to score (default: train,val,test).",
    )
    body20_compare.add_argument("--device", default="cuda")
    body20_compare.add_argument("--seed", type=int, default=42)
    body20_compare.add_argument("--microbatch", type=int, default=16)
    body20_compare.add_argument("--confidence-threshold", type=float, default=0.5)
    body20_compare.add_argument("--train-smooth-window", type=int, default=50)
    body20_compare.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse existing predictions under both prediction roots.",
    )
    body20_compare.set_defaults(handler=_cmd_plot_stage1_body20_comparison)

    limb_angles = subparsers.add_parser(
        "plot-stage1-limb-angles",
        help=(
            "Compare teacher/manual/automatic joint-angle amplitude (max−min) on "
            "annotation-positive limbs."
        ),
    )
    _add_root(limb_angles)
    limb_angles.add_argument(
        "--teacher-manifest",
        default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
    )
    limb_angles.add_argument(
        "--annotation-csv",
        default="Data/processed/manual_annotations_normalized.csv",
    )
    limb_angles.add_argument(
        "--manual-prediction-root",
        default="Runs/predictions/manual20_full_splits",
    )
    limb_angles.add_argument(
        "--automatic-prediction-root",
        default="Runs/predictions/automatic20_full_splits",
    )
    limb_angles.add_argument(
        "--output-dir",
        default="Runs/figures/stage1_body20_limb_angle_amplitude",
    )
    limb_angles.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits from the teacher manifest (default: train,val,test).",
    )
    limb_angles.add_argument("--confidence-threshold", type=float, default=0.5)
    limb_angles.set_defaults(handler=_cmd_plot_stage1_limb_angles)

    def _add_frozen_stage2_eval_args(parser: argparse.ArgumentParser) -> None:
        _add_root(parser)
        _add_dry_run(parser)
        parser.add_argument(
            "--teacher-manifest",
            default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl",
        )
        parser.add_argument(
            "--manual-bridge-manifest",
            default=(
                "Runs/cascades/manual20-constrained-frozen/bridge/control_manifest.jsonl"
            ),
        )
        parser.add_argument(
            "--automatic-prediction-manifest",
            default="Runs/predictions/automatic20_full_splits/prediction_manifest.jsonl",
        )
        parser.add_argument(
            "--output-root",
            default="Runs/stage2_frozen_eval_body20",
        )
        parser.add_argument(
            "--automatic-bridge-root",
            help="Bridge output root for automatic generated controls.",
        )
        parser.add_argument(
            "--splits",
            default="val,test",
            help="Comma-separated accepted teacher-manifest splits.",
        )
        subset = parser.add_mutually_exclusive_group(required=True)
        subset.add_argument(
            "--clip-id",
            action="append",
            default=[],
            help="Evaluate one accepted clip ID; repeat for an explicit subset.",
        )
        subset.add_argument(
            "--ids-file",
            help="Newline-delimited accepted clip IDs to evaluate as a fixed subset.",
        )
        subset.add_argument(
            "--all-accepted",
            action="store_true",
            help="Evaluate every successful teacher in --splits.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Keep only the first N ordered accepted clips after the ID filter.",
        )
        parser.add_argument("--num-inference-steps", type=int, default=50)
        parser.add_argument("--height", type=int, default=832)
        parser.add_argument("--width", type=int, default=480)
        parser.add_argument(
            "--persistent-dit-params",
            type=int,
            default=0,
            help="DiT parameters kept resident on CUDA; 0 is the GB10 low-memory path.",
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--oracle-subdir",
            default="pose_unianimate_test",
            help="On-disk oracle DWPose folder under each processed sample.",
        )
        parser.add_argument(
            "--postprocess",
            choices=("constrained", "raw"),
            default="constrained",
        )
        parser.add_argument("--unianimate-dir", default="external/UniAnimate-DiT")
        parser.add_argument("--overwrite", action="store_true")

    build_frozen = subparsers.add_parser(
        "build-stage2-frozen-eval",
        help="Build frozen Stage-2 eval manifests for body20 val+test sweeps.",
    )
    _add_frozen_stage2_eval_args(build_frozen)
    build_frozen.add_argument(
        "--skip-bridge",
        action="store_true",
        help="Assume automatic bridge controls already exist.",
    )
    build_frozen.set_defaults(handler=_cmd_build_stage2_frozen_eval)

    run_frozen = subparsers.add_parser(
        "run-stage2-frozen-eval",
        help="Render automatic controls, build manifests, and run frozen Stage-2.",
    )
    _add_frozen_stage2_eval_args(run_frozen)
    run_frozen.add_argument(
        "--skip-bridge",
        action="store_true",
        help="Skip automatic bridge rendering.",
    )
    run_frozen.add_argument(
        "--skip-infer",
        action="store_true",
        help="Build manifests only; do not run infer-stage2.",
    )
    run_frozen.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Do not build labeled 2x2 comparison MP4s; metric-input manifests are still written.",
    )
    run_frozen.set_defaults(handler=_cmd_run_stage2_frozen_eval)

    follow = subparsers.add_parser(
        "extract-stage2-pose",
        help="Re-extract DWPose from Stage-2 videos and compare to control images.",
    )
    _add_root(follow)
    _add_dry_run(follow)
    follow.add_argument(
        "--manifest",
        default="Runs/stage2_frozen_eval_body20_50clips/manifests/eval_metadata.jsonl",
        help="Frozen-eval eval_metadata.jsonl or metric_inputs.jsonl.",
    )
    follow.add_argument(
        "--output",
        default="Runs/evaluation/stage2_followthrough.json",
    )
    follow.add_argument("--clip-id", action="append", default=[])
    follow.add_argument("--limit", type=int)
    follow.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    follow.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Permit CPU ONNX only for an explicitly requested smoke check.",
    )
    follow.add_argument("--bootstrap-draws", type=int, default=2000)
    follow.add_argument("--seed", type=int, default=42)
    follow.set_defaults(handler=_cmd_extract_stage2_pose)
    return parser


def _cmd_preflight(args: argparse.Namespace) -> int:
    from .preflight import run_preflight

    report = run_preflight(
        _root_path(args.root),
        min_free_gib=args.min_free_gib,
        min_available_memory_gib=args.min_memory_gib,
        require_ort_cuda=not args.allow_cpu_onnx,
        verify_model_hashes=args.verify_model_hashes,
        probe_onnx_models=not args.skip_onnx_probe,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for check in report.checks:
            marker = "PASS" if check.ok else "FAIL" if check.required else "WARN"
            print(f"[{marker}] {check.name}: {check.detail}")
        print(f"ready={str(report.ready).lower()}")
    return 0 if report.ready else 2


def _cmd_extract(args: argparse.Namespace) -> int:
    from .dataset import build_matched_teacher_manifest, extract_keypoint_dataset

    statuses = extract_keypoint_dataset(
        project_root=args.root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        detector_model=args.detector_model,
        pose_model=args.pose_model,
        provider=args.provider,
        allow_cpu_smoke=args.allow_cpu_smoke,
        det_threshold=args.det_threshold,
        pose_threshold=args.pose_threshold,
        clip_ids=args.clip_id,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    counts: dict[str, int] = {}
    for status in statuses:
        counts[status.state] = counts.get(status.state, 0) + 1
    print(json.dumps({"selected": len(statuses), "states": counts}, sort_keys=True))
    is_full = not args.clip_id and args.max_samples is None
    if is_full and not args.dry_run:
        build_matched_teacher_manifest(
            project_root=args.root,
            intersection_manifest=args.matched_prompts,
            extraction_root=args.output_root,
            split_manifest=args.split_manifest,
            output_path=args.teacher_manifest,
            dataset_manifest=args.manifest,
        )
    return 1 if any(status.state == "error" for status in statuses) else 0


def _run_legacy(script: str, root: Path, forwarded: Sequence[str], dry_run: bool) -> int:
    import subprocess

    command = [sys.executable, str(root / script), *forwarded]
    if dry_run:
        print(json.dumps({"legacy_binary_mask_command": command}))
        return 0
    return subprocess.run(command, cwd=root, check=False).returncode


def _cmd_train_stage1(args: argparse.Namespace) -> int:
    root = _root_path(args.root)
    if args.stage1_mode == "legacy_binary_mask":
        return _run_legacy("CodeStage1/03_train_stage1.py", root, args.legacy_argv, args.dry_run)
    from .stage1_runtime import Stage1TrainRequest, train_stage1

    return train_stage1(Stage1TrainRequest.from_namespace(args, root))


def _cmd_predict_stage1(args: argparse.Namespace) -> int:
    root = _root_path(args.root)
    if args.stage1_mode == "legacy_binary_mask":
        return _run_legacy("CodeStage1/04_evaluate_stage1.py", root, args.legacy_argv, args.dry_run)
    from .stage1_runtime import Stage1PredictRequest, predict_stage1

    return predict_stage1(Stage1PredictRequest.from_namespace(args, root))


def _cmd_render_controls(args: argparse.Namespace) -> int:
    from .bridge_runtime import RenderControlsRequest, render_controls

    return render_controls(RenderControlsRequest.from_namespace(args, _root_path(args.root)))


def _cmd_train_stage2(args: argparse.Namespace) -> int:
    from .stage2.training import AdapterTrainRequest, train_adapter

    return train_adapter(AdapterTrainRequest.from_namespace(args, _root_path(args.root)))


def _cmd_infer_stage2(args: argparse.Namespace) -> int:
    from .orchestration import infer_stage2

    return infer_stage2(args, _root_path(args.root))


def _cmd_run_cascade(args: argparse.Namespace) -> int:
    if args.stage1_mode == "legacy_binary_mask":
        raise ValueError(
            "legacy_binary_mask is a Stage-1-only ablation and cannot enter the unchanged RGB DWPose pathway"
        )
    from .orchestration import run_cascade

    return run_cascade(args, _root_path(args.root))


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation_runtime import evaluate_manifest

    return evaluate_manifest(args, _root_path(args.root))


def _cmd_plot_stage1_curves(args: argparse.Namespace) -> int:
    from .stage1.metrics_plot import plot_manual_vs_automatic_curves

    root = _root_path(args.root)
    manual = Path(args.manual_dir)
    automatic = Path(args.automatic_dir)
    output = Path(args.output)
    if not manual.is_absolute():
        manual = root / manual
    if not automatic.is_absolute():
        automatic = root / automatic
    if not output.is_absolute():
        output = root / output
    written = plot_manual_vs_automatic_curves(
        manual,
        automatic,
        output_path=output,
        train_smooth_window=int(args.train_smooth_window),
    )
    print(json.dumps({"wrote": str(written)}, sort_keys=True))
    return 0


def _cmd_plot_stage1_manuscript(args: argparse.Namespace) -> int:
    from .stage1.manuscript_figures import (
        DEFAULT_MANUSCRIPT_CLIPS,
        generate_manuscript_figures,
    )

    root = _root_path(args.root)

    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path)

    clip_ids = tuple(args.clip_id) if args.clip_id else DEFAULT_MANUSCRIPT_CLIPS
    manifest = generate_manuscript_figures(
        clip_ids=clip_ids,
        prediction_root=_resolve(args.prediction_root),
        teacher_manifest=_resolve(args.teacher_manifest),
        output_dir=_resolve(args.output_dir),
        unianimate_dir=_resolve(args.unianimate_dir),
        confidence_threshold=float(args.confidence_threshold),
        fps=float(args.fps),
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


def _cmd_stage1_manuscript_report(args: argparse.Namespace) -> int:
    from .stage1.manuscript_figures import run_stage1_manuscript_report

    report = run_stage1_manuscript_report(
        root=_root_path(args.root),
        checkpoint=args.checkpoint,
        teacher_manifest=args.teacher_manifest,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        prompt_source=args.prompt_source,
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        n_test=int(args.n_test),
        unianimate_dir=args.unianimate_dir,
        device=args.device,
        seed=int(args.seed),
        fps=float(args.fps),
        confidence_threshold=float(args.confidence_threshold),
        skip_predict=bool(args.skip_predict),
        skip_figures=bool(args.skip_figures),
    )
    # Print the manuscript numbers prominently.
    for summary in report["splits"]:
        print(
            f"{summary['split']}: mean={summary['mean_error']:.6f} "
            f"std={summary['std_error']:.6f} n={summary['n_clips']}"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


def _cmd_plot_stage1_limb_angles(args: argparse.Namespace) -> int:
    from .stage1.limb_angle_analysis import run_limb_angle_amplitude_report
    from .stage1.manuscript_figures import list_split_clip_ids

    root = _root_path(args.root)
    teacher_manifest = Path(args.teacher_manifest)
    if not teacher_manifest.is_absolute():
        teacher_manifest = root / teacher_manifest
    splits = tuple(part.strip() for part in str(args.splits).split(",") if part.strip())
    clip_ids: list[str] = []
    for split in splits:
        clip_ids.extend(list_split_clip_ids(teacher_manifest, split=split))
    report = run_limb_angle_amplitude_report(
        root=root,
        teacher_manifest=teacher_manifest,
        annotation_csv=args.annotation_csv,
        manual_prediction_root=args.manual_prediction_root,
        automatic_prediction_root=args.automatic_prediction_root,
        output_dir=args.output_dir,
        clip_ids=tuple(sorted(set(clip_ids))),
        confidence_threshold=float(args.confidence_threshold),
    )
    for key, path in report["figures"].items():
        print(f"{key}: {path}")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


def _cmd_plot_stage1_body20_comparison(args: argparse.Namespace) -> int:
    from .stage1.comparison_plots import run_stage1_body20_comparison_report

    splits = tuple(
        part.strip() for part in str(args.splits).split(",") if part.strip()
    )
    report = run_stage1_body20_comparison_report(
        root=_root_path(args.root),
        teacher_manifest=args.teacher_manifest,
        manual_checkpoint=args.manual_checkpoint,
        automatic_checkpoint=args.automatic_checkpoint,
        manual_prediction_root=args.manual_prediction_root,
        automatic_prediction_root=args.automatic_prediction_root,
        manual_metrics_dir=args.manual_metrics_dir,
        automatic_metrics_dir=args.automatic_metrics_dir,
        output_dir=args.output_dir,
        splits=splits,
        device=args.device,
        seed=int(args.seed),
        microbatch=int(args.microbatch),
        confidence_threshold=float(args.confidence_threshold),
        skip_predict=bool(args.skip_predict),
        train_smooth_window=int(args.train_smooth_window),
    )
    for key, path in report["figures"].items():
        print(f"{key}: {path}")
    print(json.dumps({"manifest": report.get("grouped_error_summary_json")}, sort_keys=True))
    return 0


def _cmd_stage1_full_split_errors(args: argparse.Namespace) -> int:
    from .stage1.manuscript_figures import run_stage1_full_split_error_report

    splits = tuple(
        part.strip() for part in str(args.splits).split(",") if part.strip()
    )
    report = run_stage1_full_split_error_report(
        root=_root_path(args.root),
        checkpoint=args.checkpoint,
        teacher_manifest=args.teacher_manifest,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        prompt_source=args.prompt_source,
        splits=splits,
        device=args.device,
        seed=int(args.seed),
        confidence_threshold=float(args.confidence_threshold),
        skip_predict=bool(args.skip_predict),
        microbatch=int(args.microbatch),
    )
    for summary in report["splits"]:
        print(
            f"{summary['split']}: mean={summary['mean_error']:.6f} "
            f"std={summary['std_error']:.6f} n={summary['n_clips']}"
        )
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "split_counts": report["split_counts"],
                "error_summary_json": report["error_summary_json"],
                "error_summary_markdown": report["error_summary_markdown"],
                "per_clip_errors_jsonl": report["per_clip_errors_jsonl"],
                "splits": [
                    {
                        "split": row["split"],
                        "n_clips": row["n_clips"],
                        "mean_error": row["mean_error"],
                        "std_error": row["std_error"],
                    }
                    for row in report["splits"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_build_stage2_frozen_eval(args: argparse.Namespace) -> int:
    from .stage2.frozen_eval import FrozenStage2EvalRequest, build_frozen_stage2_eval

    request = FrozenStage2EvalRequest.from_namespace(args, _root_path(args.root))
    summary = build_frozen_stage2_eval(request)
    print(json.dumps(summary, sort_keys=True))
    return 0


def _cmd_extract_stage2_pose(args: argparse.Namespace) -> int:
    from .stage2.followthrough import run_stage2_followthrough

    report = run_stage2_followthrough(
        root=_root_path(args.root),
        manifest=args.manifest,
        output=args.output,
        clip_ids=tuple(args.clip_id),
        limit=args.limit,
        dry_run=bool(args.dry_run),
        provider=str(args.provider),
        allow_cpu_smoke=bool(args.allow_cpu_smoke),
        seed=int(args.seed),
        bootstrap_draws=int(args.bootstrap_draws),
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "n_selected": report.get("n_selected"),
                "n_ok": report.get("n_ok"),
                "n_error": report.get("n_error"),
                "dry_run": report.get("dry_run"),
                "conditions": sorted(report.get("conditions") or {}),
                "generated_vs_oracle": report.get("generated_vs_oracle"),
            },
            sort_keys=True,
        )
    )
    if report.get("dry_run"):
        return 0
    return 1 if int(report.get("n_error") or 0) else 0


def _cmd_run_stage2_frozen_eval(args: argparse.Namespace) -> int:
    from .stage2.frozen_eval import FrozenStage2EvalRequest, run_frozen_stage2_eval

    request = FrozenStage2EvalRequest.from_namespace(args, _root_path(args.root))
    summary = run_frozen_stage2_eval(request)
    print(json.dumps(summary, sort_keys=True))
    return max(
        int(summary.get("infer_exit_code", 0)),
        int(summary.get("artifact_exit_code", 0)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
