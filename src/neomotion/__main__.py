"""Independent NEST CLI: train, predict a new prompt, and evaluate checkpoints."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .config import V7Config


def build_parser():
    parser = argparse.ArgumentParser(description="Stage 1 NEST: prompt + observed pose -> 20-frame body20 motion")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train either experimental prompt source with identical teachers and recipe")
    predict = sub.add_parser("predict", help="Generate from a free prompt and a pose NPZ; no annotation lookup")
    evaluate = sub.add_parser("evaluate", help="Evaluate generated trajectories and optionally counterfactual commands")
    for p in (train, predict, evaluate):
        p.add_argument("--root", type=Path, default=Path("."))
        p.add_argument("--device", default="cuda")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--local-files-only", action="store_true")
    for p in (train, evaluate):
        p.add_argument("--teacher-manifest", default="Data2/dwpose_woface_62_v1/teacher_training_manifest.jsonl")
        p.add_argument("--patient-split-manifest", default="reproducibility/patient_split_v1.json")
        p.add_argument("--output-dir", required=True)
    train.add_argument("--config", default="configs/v7.yaml")
    train.add_argument("--prompt-source", choices=("manual", "automatic"), default="manual")
    train.add_argument("--resume", help="Resume this output directory's last.pt")
    train.add_argument("--seed", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--validate-every", type=int)
    train.add_argument("--overfit-clips", type=int, help="Explicit training-only small-set diagnostic; not a final experiment")
    train.add_argument("--smoke-test", action="store_true", help="Four steps with a tiny backbone and 8/4 train/val clips")
    train.add_argument("--dry-run", action="store_true")
    for p in (predict, evaluate):
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--weights", choices=("selected", "raw", "ema"), default="selected")
        p.add_argument("--steps", type=int)
        p.add_argument("--cfg-scale", type=float)
    predict.add_argument("--pose", required=True, help="NPZ with pose0_xy_norm, pose0_visible, image_size, timestamps; or teacher ref_* fields")
    predict.add_argument("--prompt", required=True)
    predict.add_argument("--output", required=True, help="New output .npz (plus JSON metadata and skeleton PNG)")
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--image-size", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    predict.add_argument("--fps", type=float, help="Native pose sampling rate when the input NPZ has no timestamps")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--max-clips", type=int, help="Explicit bounded diagnostic subset")
    evaluate.add_argument("--counterfactual", action="store_true", help="Matched-noise still/side/amplitude/speed command probes")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    root = args.root.resolve()
    def resolve(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root/path).resolve()
    if args.command == "train":
        from .training import train
        cfg = V7Config.load(resolve(args.config))
        overrides = {k: getattr(args, k) for k in ("seed", "max_steps") if getattr(args, k) is not None}
        if args.validate_every is not None:
            overrides["validate_every"] = args.validate_every
        cfg = replace(cfg, **overrides)
        result = train(root, resolve(args.teacher_manifest), resolve(args.patient_split_manifest), resolve(args.output_dir),
                       args.prompt_source, cfg, args.device, resolve(args.resume) if args.resume else None,
                       args.smoke_test, args.overfit_clips, args.dry_run, args.local_files_only)
    elif args.command == "predict":
        from .inference import load_generator, generate_motion
        from .data import file_sha256
        from .training import save_json, source_hash
        from .evaluation import plot_examples
        from .common.clip_text import FrozenCLIPTextConditioner
        output = resolve(args.output)
        if output.suffix != ".npz":
            raise ValueError("Prediction output must end in .npz")
        if any(output.with_suffix(s).exists() for s in (".npz", ".json", ".png")):
            raise FileExistsError(f"Choose a new prediction output: {output}")
        with np.load(resolve(args.pose), allow_pickle=False) as arrays:
            if "pose0_xy_norm" in arrays:
                pose, visible = arrays["pose0_xy_norm"], arrays["pose0_visible"]
                score = arrays["pose0_score"] if "pose0_score" in arrays else None
            else:
                pose, visible = arrays["ref_xy_norm"][:20], arrays["ref_visible"][:20]
                score = arrays["ref_score"][:20] if "ref_score" in arrays else None
            size = args.image_size if args.image_size is not None else arrays["image_size"] if "image_size" in arrays else None
            times = arrays["timestamps"] if "timestamps" in arrays else None
        if size is None:
            raise ValueError("Supply source-pixel --image-size WIDTH HEIGHT or image_size in the pose NPZ")
        if args.fps is not None:
            if not np.isfinite(args.fps) or args.fps <= 0:
                raise ValueError("--fps must be positive and finite")
            times = np.arange(20, dtype=np.float32)/args.fps
        if times is None:
            raise ValueError("Supply native --fps or timestamps in the pose NPZ")
        model, scales, cfg, payload = load_generator(resolve(args.checkpoint), args.device, args.weights)
        encoder = FrozenCLIPTextConditioner(local_files_only=args.local_files_only)
        prediction = generate_motion(model, scales, encoder, args.prompt, pose, visible, size, times, args.seed, score,
                                     cfg.diffusion_steps, cfg.inference_steps if args.steps is None else args.steps,
                                     cfg.cfg_scale if args.cfg_scale is None else args.cfg_scale)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **{k: v.detach().cpu().numpy() for k,v in prediction.items() if isinstance(v, torch.Tensor)})
        result = {"format": "dmtbaby.stage1.v7.prediction", "checkpoint_sha256": file_sha256(resolve(args.checkpoint)),
                  "inference_source_sha256": source_hash(), "checkpoint_step": payload["step"],
                  "weights": payload["loaded_weights"], "training_prompt_source": payload["provenance"]["prompt_source"],
                  "prompt": args.prompt, "seed": args.seed, "shape": [20,20,2], "joint_scope": "body20",
                  "coordinate_space": "normalized_image_unclipped", "timestamp_units": "seconds",
                  "image_size_order": ["width", "height"], "prompt_controls": prediction["prompt_controls"],
                  "outside_fraction": float((((prediction['xy_norm'][1:]<0)|(prediction['xy_norm'][1:]>1)).any(-1)).float().mean()),
                  "output": str(output)}
        save_json(output.with_suffix(".json"), result)
        # Use the same drawing code with the observed pose repeated in the reference row.
        examples = {k: prediction[k].detach().cpu()[None] for k in ("xy_norm","render_visible","timestamps","image_size")}
        examples.update(ids=["new_prompt"], teacher=torch.as_tensor(pose)[None,None].expand(1,20,20,2),
                        teacher_visible=torch.as_tensor(visible)[None,None].expand(1,20,20))
        figure = plot_examples(examples, max_clips=1)
        figure.axes[0].set_ylabel("starting pose")
        figure.savefig(output.with_suffix(".png"), dpi=140)
        import matplotlib.pyplot as plt
        plt.close(figure)
    else:
        from .runner import evaluate_checkpoint
        result = evaluate_checkpoint(resolve(args.checkpoint), resolve(args.teacher_manifest), resolve(args.patient_split_manifest),
            resolve(args.output_dir), args.split, args.weights, args.device, args.max_clips,
            args.counterfactual, args.local_files_only, args.steps, args.cfg_scale)
    if not (args.command == "train" and args.dry_run):
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"NEST error: {error}", file=sys.stderr)
        raise SystemExit(1)
