"""NEST training flow: per-limb motion terms, caption balancing and per-round EMA snapshots."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common.clip_text import FrozenCLIPTextConditioner
from .conditioning import encode_prompt_table, prompt_batch, parse_prompt
from .data import verify_split, load_split, tensor_batch, fit_training_normalization, balanced_weights, patient_subset
from .diffusion import cosine_schedule, training_loss, update_ema
from .evaluation import evaluate_model, plot_examples
from .inference import model_inputs
from .model import ResidualMotionDenoiser
from .motion import BOUND_NAMES, fit_label_bounds, control_balance
from .representation import reference_frame, encode_future


def atomic_checkpoint(path, payload):
    """Native PyTorch checkpoint only; readable metadata/metrics are saved separately."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train(root, manifest, split_path, output, prompt_source, cfg, device="cuda", resume=None,
          smoke_test=False, overfit_clips=None, dry_run=False, local_files_only=False):
    # INPUT NODE: checksum-verified teachers and immutable patient assignments.
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest, split_path = Path(manifest).resolve(), Path(split_path).resolve()
    if prompt_source not in ("manual", "automatic"):
        raise ValueError("prompt_source must be manual or automatic")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if smoke_test:
        cfg = replace(cfg, width=32, layers=1, heads=4, batch_size=2, accumulation=1,
                      max_steps=4, min_steps=0, warmup_steps=1, validate_every=2, log_every=1,
                      ema_burnin=1, diffusion_steps=20, inference_steps=4, validation_seeds=(31415,), prior_steps=0)
    if overfit_clips is not None and overfit_clips < 1:
        raise ValueError("overfit_clips must be positive")
    provenance = verify_split(manifest, split_path)
    provenance.update(prompt_source=prompt_source, source_sha256=source_hash(),
                      smoke_test=smoke_test, overfit_clips=overfit_clips,
                      torch_version=str(torch.__version__), device_type=torch.device(device).type, normalization_fit_split="train",
                      prompt_controls="selected_prompt_only", sampling="patient_then_teacher_activity_bin_times_control_class")
    if dry_run:
        result = {"config": cfg.to_dict(), "provenance": provenance, "output": str(output)}
        print(json.dumps(result, indent=2))
        return result
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory or --resume: {output}")
    if resume is not None and Path(resume).resolve() != output / "last.pt":
        raise ValueError("Resume from last.pt in the same output directory (the committed run boundary)")
    output.mkdir(parents=True, exist_ok=True)

    # DERIVED NODE: training-only scale statistics and shared sampling weights.
    started = time.monotonic()
    train_data = load_split(manifest, prompt_source, "train")
    full_train = train_data  # label bounds always use the full training split, even for subsets
    val_data = load_split(manifest, prompt_source, "val")
    if smoke_test:
        train_data = patient_subset(train_data, 8)
        val_data = patient_subset(val_data, 4)
    if overfit_clips is not None:
        train_data = patient_subset(train_data, overfit_clips)
    scales, activity, activity_threshold = fit_training_normalization(train_data, cfg.residual_scale_floor)
    weights = balanced_weights(train_data["patients"], activity, activity_threshold)
    # Since V6: temper the patient/activity law by control-class rarity so single-limb and
    # high/slow captions are not drowned out by whole-body high/fast ones.
    # NEST: the class captions come from cfg.sampling_caption_source (manual) for BOTH runs,
    # so manual and automatic training draw clips with identical probabilities.
    with manifest.open() as stream:
        records = {r["clip_id"]: r for r in map(json.loads, stream) if r.get("status") == "ok"}
    column = f"{cfg.sampling_caption_source}_prompt"
    sampling_prompts = [records[c][column] for c in train_data["ids"]]
    if cfg.sampling_caption_source == prompt_source and sampling_prompts != list(train_data["prompts"]):
        raise ValueError("Manifest captions disagree with the loaded training prompts")
    train_controls = torch.stack([parse_prompt(p)[0] for p in sampling_prompts])
    balance, class_counts = control_balance(train_controls, cfg.control_balance_power)
    weights = weights * balance
    weights = weights / weights.sum()
    sampling_sha256 = hashlib.sha256(json.dumps([train_data["ids"], [round(float(w), 12) for w in weights]]).encode()).hexdigest()
    provenance["sampling_caption_source"] = cfg.sampling_caption_source
    provenance["sampling_probabilities_sha256"] = sampling_sha256
    print(f"NEST draw law from {cfg.sampling_caption_source} captions, sha256={sampling_sha256[:16]}", flush=True)
    full_controls = torch.stack([parse_prompt(p)[0] for p in full_train["prompts"]])
    teacher_frame = reference_frame(full_train["pose0"], full_train["visible0"], full_train["image_size"])
    teacher_residual = encode_future(full_train["target"], full_train["valid"], teacher_frame)
    label_bounds, bound_counts = fit_label_bounds(teacher_residual, full_train["valid"], full_train["visible0"],
                                                  full_train["timestamps"], full_controls, cfg.label_quantile)
    print("NEST label bounds: " + ", ".join(f"{n}={float(v):.4f} (n={bound_counts[n]})" if torch.isfinite(v)
                                          else f"{n}=disabled (n={bound_counts[n]})"
                                          for n, v in zip(BOUND_NAMES, label_bounds)), flush=True)
    provenance["training_clip_ids"] = train_data["ids"]
    provenance["validation_clip_ids"] = val_data["ids"]
    provenance["prompt_sha256"] = hashlib.sha256(json.dumps(train_data["prompts"]+val_data["prompts"]).encode()).hexdigest()
    print(f"NEST {prompt_source}: train={len(train_data['ids'])}, val={len(val_data['ids'])}, device={device}", flush=True)
    # Frozen text cache is in-memory only. Its weights/revision are verified by the adapter.
    encoder = FrozenCLIPTextConditioner(local_files_only=local_files_only, torch_dtype=torch.float32)
    text_table = encode_prompt_table(train_data["prompts"]+val_data["prompts"], encoder, device)
    provenance["text_encoder"] = {"model_id": encoder.model_id, "revision": encoder.revision, "sha256": encoder.model_sha256}
    del encoder
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    model = ResidualMotionDenoiser(cfg.width, cfg.layers, cfg.heads, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(.9, .95), weight_decay=cfg.weight_decay)
    shadow = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
              for k, v in model.state_dict().items()}
    sampler_rng = torch.Generator().manual_seed(cfg.seed + 17)
    noise_rng = torch.Generator().manual_seed(cfg.seed + 29)
    scales = scales.to(device)
    label_bounds_device = label_bounds.to(device)
    schedule = cosine_schedule(cfg.diffusion_steps, device)
    step, patience, best_score = 0, 0, math.inf
    best_by_weight = {"raw": math.inf, "ema": math.inf}
    selected_weights = "raw"
    last_validation = None

    # RESUME NODE: exact recipe/data identity and independent RNG states.
    if resume is not None:
        payload = torch.load(Path(resume), map_location="cpu", weights_only=True)
        if payload.get("format") != "dmtbaby.stage1.v7" or payload["config"] != cfg.to_dict():
            raise ValueError("Resume requires the same NEST recipe/configuration")
        for key in ("teacher_manifest_sha256", "patient_split_sha256", "prompt_source", "prompt_sha256",
                    "source_sha256", "training_clip_ids", "validation_clip_ids", "smoke_test", "overfit_clips", "text_encoder", "torch_version", "device_type", "sampling_probabilities_sha256"):
            if payload["provenance"][key] != provenance[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
        if not torch.equal(payload["scales"], scales.cpu()):
            raise ValueError("Training normalization changed since checkpoint")
        if not torch.allclose(payload["label_bounds"], label_bounds.cpu(), rtol=0, atol=0, equal_nan=True):
            raise ValueError("NEST label bounds changed since checkpoint")
        model.load_state_dict(payload["model"])
        shadow = {k: v.to(device) for k, v in payload["ema"].items()}
        optimizer.load_state_dict(payload["optimizer"])
        step, patience, best_score = payload["step"], payload["patience"], payload["best_score"]
        best_by_weight, selected_weights = payload["best_by_weight"], payload["selected_weights"]
        last_validation = payload["last_validation"]
        sampler_rng.set_state(payload["sampler_rng"])
        noise_rng.set_state(payload["noise_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        # Remove log records after the committed checkpoint boundary after a failed run.
        for name in ("metrics.jsonl", "validation_per_clip.jsonl"):
            p = output / name
            if p.exists():
                rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                p.write_text("".join(json.dumps(r)+"\n" for r in rows if r["step"] <= step))
    save_json(output / "resolved_config.json", cfg.to_dict())
    save_json(output / "provenance.json", provenance)
    save_json(output / "normalization.json", {"axes": ["observed_or_unknown_start", "joint", "xy"],
              "rms_scales": scales.cpu().tolist(), "floor": cfg.residual_scale_floor,
              "activity_threshold": activity_threshold, "fit_clip_ids": train_data["ids"],
              "label_bounds": {n: float(v) if torch.isfinite(v) else None for n, v in zip(BOUND_NAMES, label_bounds)},
              "label_bound_counts": bound_counts, "label_quantile": cfg.label_quantile})
    save_json(output / "sampling.json", {"clip_ids": train_data["ids"], "patients": train_data["patients"],
              "teacher_activity": activity.tolist(), "probabilities": weights.tolist(),
              "control_class_counts": class_counts, "control_balance_power": cfg.control_balance_power,
              "sampling_caption_source": cfg.sampling_caption_source, "probabilities_sha256": sampling_sha256})

    def checkpoint():
        return {"format": "dmtbaby.stage1.v7", "step": step, "config": cfg.to_dict(), "provenance": provenance,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "ema": {k: v.detach().cpu().clone() for k, v in shadow.items()}, "scales": scales.cpu(),
                "label_bounds": label_bounds.cpu(),
                "optimizer": optimizer.state_dict(), "sampler_rng": sampler_rng.get_state(), "noise_rng": noise_rng.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "best_score": best_score, "best_by_weight": dict(best_by_weight), "selected_weights": selected_weights,
                "patience": patience, "last_validation": last_validation}

    # TRAINING EDGE: exact same clip sampling law, only selected prompts differ.
    for next_step in range(step + 1, cfg.max_steps + 1):
        step = next_step
        if cfg.prior_steps and step == cfg.prior_steps + 1:
            best_score, patience = math.inf, 0
            best_by_weight = {"raw": math.inf, "ema": math.inf}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = min(1., step / max(cfg.warmup_steps, 1))
        progress = max(0., (step-cfg.warmup_steps)/max(1, cfg.max_steps-cfg.warmup_steps))
        lr = cfg.learning_rate * warmup * .5 * (1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = {}
        for _ in range(cfg.accumulation):
            indices = torch.multinomial(weights, cfg.batch_size, replacement=True, generator=sampler_rng)
            batch = tensor_batch(train_data, indices, device)
            frame = reference_frame(batch["pose0"], batch["visible0"], batch["image_size"])
            target = encode_future(batch["target"], batch["valid"], frame, scales)
            text = prompt_batch(text_table, batch["prompts"], device)
            dropped = torch.rand(cfg.batch_size, generator=noise_rng).to(device) < cfg.condition_dropout
            if step <= cfg.prior_steps:
                dropped[:] = True
            timesteps = torch.randint(cfg.diffusion_steps, (cfg.batch_size,), generator=noise_rng).to(device)
            noise = torch.randn(target.shape, generator=noise_rng).to(device)
            branch = (~batch["visible0"]).long()
            state_scales = scales[branch, torch.arange(20, device=device)[None]]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with amp:
                motion = {"controls": text["controls"], "dropped": dropped, "bounds": label_bounds_device,
                          "magnitude_weight": cfg.magnitude_weight, "label_weight": cfg.label_weight, "eps": cfg.motion_eps}
                losses = training_loss(model, target, batch["valid"], model_inputs(batch, frame, text, dropped),
                                       schedule, timesteps, noise, cfg.visibility_weight, cfg.velocity_weight, state_scales,
                                       motion=motion)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Nonfinite NEST training loss at {step}")
            (losses["loss"]/cfg.accumulation).backward()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + float(value.detach())/cfg.accumulation
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
        optimizer.step()
        effective_decay = update_ema(shadow, model, step, cfg.ema_burnin, cfg.ema_decay)
        if step % cfg.log_every == 0 or step == 1:
            record = dict(kind="train", step=step, phase="prior" if step<=cfg.prior_steps else "conditional",
                          **totals, lr=lr, gradient_norm=gradient_norm, ema_decay=effective_decay)
            append_jsonl(output / "metrics.jsonl", record)
            print(f"step={step} loss={totals['loss']:.5f} coordinate={totals['coordinate']:.5f} "
                  f"magnitude={totals['magnitude']:.5f} label={totals['label']:.5f} ema_decay={effective_decay:.4f}", flush=True)

        # VALIDATION EDGE: free generation with fixed per-clip noise, raw AND EMA.
        if step % cfg.validate_every == 0 or step == cfg.max_steps:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            summaries, improved_weights = {}, []
            try:
                for kind, state in (("raw", raw), ("ema", shadow)):
                    model.load_state_dict(state)
                    summary, rows, examples = evaluate_model(model, scales, val_data, text_table, cfg.validation_seeds,
                        cfg.batch_size, cfg.diffusion_steps, cfg.inference_steps, cfg.cfg_scale, force_unconditional=step<=cfg.prior_steps)
                    summaries[kind] = summary
                    for row in rows:
                        append_jsonl(output / "validation_per_clip.jsonl", dict(step=step, weights=kind, **row))
                    score = summary["patient_mean"]["selection_score"]
                    if score < best_by_weight[kind]:
                        best_by_weight[kind] = score
                        improved_weights.append(kind)
                    example_dir = output / "examples" / f"step_{step:06d}"
                    example_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(example_dir / f"{kind}.npz", **{k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                                        for k, v in examples.items()})
                    fig = plot_examples(examples)
                    fig.savefig(example_dir / f"{kind}.png", dpi=140)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            finally:
                model.load_state_dict(raw)
            chosen = min(summaries, key=lambda k: summaries[k]["patient_mean"]["selection_score"])
            score = summaries[chosen]["patient_mean"]["selection_score"]
            improved = score < best_score
            if improved:
                best_score, selected_weights, patience = score, chosen, 0
            elif step >= max(cfg.min_steps, cfg.prior_steps):
                patience += 1
            phase = "prior" if step <= cfg.prior_steps else "conditional"
            last_validation = {"step": step, "phase": phase, "raw": summaries["raw"], "ema": summaries["ema"]}
            append_jsonl(output / "metrics.jsonl", dict(kind="validation", step=step, summaries=summaries,
                          best_score=best_score, selected_weights=selected_weights, improved=improved, patience=patience))
            payload = checkpoint()
            best_name = "best_prior" if phase == "prior" else "best"
            for kind in improved_weights:
                atomic_checkpoint(output / f"{best_name}_{kind}.pt", dict(payload, selected_weights=kind))
            if improved:
                atomic_checkpoint(output / f"{best_name}.pt", payload)
            atomic_checkpoint(output / "last.pt", payload)
            # NEST: inference-only EMA snapshot every validation round (~8 MB), so any step
            # (not only the selection winner) can be evaluated or used later.
            snapshot_dir = output / "ema_snapshots"
            snapshot_dir.mkdir(exist_ok=True)
            atomic_checkpoint(snapshot_dir / f"step_{step:06d}.pt",
                              {k: payload[k] for k in ("format", "step", "config", "provenance", "ema", "scales", "label_bounds")}
                              | {"selected_weights": "ema", "snapshot": True, "last_validation": last_validation})
            print(f"validation step={step} raw={summaries['raw']['patient_mean']['selection_score']:.5f} "
                  f"ema={summaries['ema']['patient_mean']['selection_score']:.5f} best={best_score:.5f} ({selected_weights})", flush=True)
            if patience >= cfg.patience:
                break

    summary = {"format": "dmtbaby.stage1.v7", "optimizer_steps": step, "best_score": best_score,
               "best_weights": selected_weights, "parameters": sum(p.numel() for p in model.parameters()),
               "elapsed_seconds": time.monotonic()-started, "smoke_test": smoke_test,
               "stopped_early": step < cfg.max_steps, "best_checkpoint": str(output / "best.pt"),
               "last_checkpoint": str(output / "last.pt")}
    save_json(output / "training_summary.json", summary)
    return summary
