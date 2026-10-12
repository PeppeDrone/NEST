#!/usr/bin/env python3
"""Turn the shortlisted manual test trajectories into Stage-2 (UniAnimate) inputs.

    source scripts/env.sh
    PYTHONPATH=src "$PY" stage2/build_inputs.py [--steps 100] [--height 832 --width 480]

Inputs:  stage2/shortlist.csv (24 bridge-ready test trajectories, from stage2/screen.py)
         analysis/artifacts/test_partA/<model>/validate/trajectories_seed<seed>.npz
Steps:
  1. each trajectory -> checksummed dmtbaby prediction pose (body20, 20 native frames;
     visibility logits +-20 from the generated render_visible mask)      -> predictions/
  2. production structural bridge (constrained post-processing, anatomy solve on)
     -> 81 DWPose control JPEGs + pose.jpg per clip                       -> bridge/
  3. strict Stage-2 request manifest (manual prompt, reference still, frozen official adapter)
                                                                          -> stage2_requests.jsonl
Order: one example per caption type first (still, 1 limb, 2 limbs, 3-4 limbs), then the rest of
the shortlist by gallery rank, so the first videos cover every caption type.
Settings follow the frozen Stage-2 evaluation (dmtbaby/stage2/frozen_eval.py): manual prompt,
stable per-clip seed, cfg 1.5, 15 fps, 832x480 (not exceeded), only the step count is raised.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                      # repository root
OUT = Path(__file__).resolve().parent / "run"                    # predictions, bridge controls, videos (not versioned)
sys.path.insert(0, str(ROOT / "src"))

from dmtbaby.bridge_runtime import RenderControlsRequest, render_controls      # noqa: E402
from dmtbaby.io import atomic_write_jsonl, sha256_file, stable_seed            # noqa: E402
from dmtbaby.pose.io import write_prediction_pose                              # noqa: E402
from dmtbaby.pose.records import PredictionPoseSequence                        # noqa: E402
from dmtbaby.pose.schema import BODY20_SCOPE, SCHEMA_ID                        # noqa: E402
from dmtbaby.stage2.service import DEFAULT_NEGATIVE_PROMPT                     # noqa: E402

PAPER = ROOT
ART = PAPER / "analysis/artifacts/test_partA"
TEACHER_MANIFEST = PAPER / "data/dwpose_woface_62_v1/teacher_training_manifest.jsonl"
SHORTLIST = Path(__file__).resolve().parent / "shortlist.csv"
UNIANIMATE = ROOT / "external/UniAnimate-DiT"
MODELS = {"manual replica A": "v7_manual_s42@002600", "manual replica B": "v7_manual_s43@004400",
          "manual replica C": "v7_manual_s44@003000"}
FIRST = ("pz11_clip0045", "pz11_clip0037", "pz17_clip0042", "pz17_clip0025")   # still, 1, 2, 3-4 limbs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="keep only the first N clips of the ordered list")
    args = ap.parse_args()
    if args.height * args.width > 832 * 480:
        raise SystemExit("resolution above 832x480 is not allowed for this run")

    shortlist = pd.read_csv(SHORTLIST)
    rank = {c: i for i, c in enumerate(FIRST)}
    shortlist["order"] = [rank.get(c, len(FIRST) + r) for c, r in zip(shortlist.clip_id, shortlist.gallery_rank)]
    shortlist = shortlist.sort_values("order").reset_index(drop=True)
    if args.limit:
        shortlist = shortlist.head(args.limit)
    teachers = {json.loads(l)["clip_id"]: json.loads(l) for l in TEACHER_MANIFEST.read_text().splitlines() if l.strip()}

    # 1. prediction pose artifacts
    rows = []
    for _, r in shortlist.iterrows():
        model = MODELS[r.manual_replica]
        run, step = model.split("@")
        ckpt = PAPER / "runs" / run / "ema_snapshots" / f"step_{int(step):06d}.pt"
        z = np.load(ART / model / "validate" / f"trajectories_seed{int(r.seed)}.npz", allow_pickle=False)
        i = list(z["ids"].astype(str)).index(r.clip_id)
        xy = z["xy_norm"][i].astype(np.float32)                         # [20 frames, 20 joints, 2]
        visible = z["render_visible"][i].astype(bool)
        logits = np.where(visible, 20.0, -20.0).astype(np.float32)
        npz = OUT / "predictions" / f"{r.clip_id}.npz"
        if not npz.with_suffix(".json").exists():
            write_prediction_pose(npz, PredictionPoseSequence(xy, logits, joint_scope=BODY20_SCOPE), {
                "joint_schema": SCHEMA_ID, "joint_scope": BODY20_SCOPE, "checkpoint_sha256": sha256_file(ckpt),
                "sampling_seed": int(r.seed), "stage1_model": model, "source": "analysis/artifacts/test_partA",
                "prompt": r.prompt})
        teacher = teachers[r.clip_id]
        if teacher.get("split") != "test":
            raise SystemExit(f"{r.clip_id} is not a test clip")
        rows.append({"clip_id": r.clip_id, "split": "test", "prediction_npz": str(npz), "prompt_source": "manual",
                     "joint_scope": BODY20_SCOPE, "manual_prompt": r.prompt, "sampling_seed": int(r.seed),
                     "stage1_model": model, "caption_type": r.caption_type_manual, "gallery_rank": int(r.gallery_rank)})
    atomic_write_jsonl(OUT / "prediction_manifest.jsonl", rows)
    print(f"[1/3] {len(rows)} prediction artifacts", flush=True)

    # 2. structural bridge -> 81 DWPose controls per clip
    code = render_controls(RenderControlsRequest(
        root=ROOT, prediction_manifest=OUT / "prediction_manifest.jsonl", teacher_manifest=TEACHER_MANIFEST,
        output_root=OUT / "bridge", unianimate_dir=UNIANIMATE, postprocess="constrained", persist="all"))
    if code:
        raise SystemExit("bridge reported errors; see bridge/control_manifest.jsonl")
    bridge = {json.loads(l)["clip_id"]: json.loads(l) for l in (OUT / "bridge/control_manifest.jsonl").read_text().splitlines() if l.strip()}
    print(f"[2/3] bridge complete for {len(bridge)} clips", flush=True)

    # 3. Stage-2 requests (same conventions as the frozen evaluation; only the step count differs)
    requests = []
    for k, row in enumerate(rows, start=1):
        b, t = bridge[row["clip_id"]], teachers[row["clip_id"]]
        reference_image = (TEACHER_MANIFEST.parent / t["reference_image"]).resolve()
        if not reference_image.is_file():
            raise SystemExit(f"missing reference still: {reference_image}")
        requests.append({
            "clip_id": row["clip_id"], "control_dir": b["control_dir"], "reference_pose": b["reference_pose"],
            "reference_image": str(reference_image),
            "output": str(OUT / "videos" / f"{k:02d}_{row['clip_id']}" / "output.mp4"),
            "prompt": row["manual_prompt"], "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "seed": stable_seed(row["clip_id"], base_seed=args.seed), "height": args.height, "width": args.width,
            "num_frames": 81, "num_inference_steps": args.steps, "cfg_scale": 1.5, "fps": 15, "quality": 5,
            "use_teacache": False})
    atomic_write_jsonl(OUT / "stage2_requests.jsonl", requests)
    print(f"[3/3] {len(requests)} Stage-2 requests -> {OUT / 'stage2_requests.jsonl'} "
          f"({args.width}x{args.height}, {args.steps} steps)", flush=True)


if __name__ == "__main__":
    main()
