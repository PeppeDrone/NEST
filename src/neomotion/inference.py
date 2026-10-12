"""Free-prompt inference and common batched sampling used by validation."""
from contextlib import nullcontext
import hashlib
from pathlib import Path

import torch

from .conditioning import prompt_batch, parse_prompt
from .config import V7Config
from .diffusion import cosine_schedule, sample_clean
from .model import ResidualMotionDenoiser
from .representation import reference_frame, decode_future


def model_inputs(batch, frame, text, dropped=None):
    return {"pose": frame["pose"], "visible": batch["visible0"], "score": batch["score0"],
            "timestamps": batch["timestamps"], **text,
            "dropped": torch.zeros(len(batch["pose0"]), dtype=torch.bool, device=batch["pose0"].device) if dropped is None else dropped}


def per_clip_noise(ids, seed, device):
    noise = []
    for clip in ids:
        # Fixed per-clip noise keeps validation samples directly comparable.
        value = int.from_bytes(hashlib.sha256(f"v5:{seed}:{clip}".encode()).digest()[:8], "big") % (2**63-1)
        noise.append(torch.randn(19, 20, 2, generator=torch.Generator().manual_seed(value)))
    return torch.stack(noise).to(device)


@torch.inference_mode()
def generate_batch(model, scales, batch, text, initial_noise, diffusion_steps=1000, inference_steps=50, cfg_scale=1., force_unconditional=False):
    frame = reference_frame(batch["pose0"], batch["visible0"], batch["image_size"])
    context = model_inputs(batch, frame, text)
    if force_unconditional:
        context["dropped"][:] = True
    device = batch["pose0"].device
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    was_training = model.training
    model.eval()
    try:
        with amp:
            state, future_logits = sample_clean(model, context, cosine_schedule(diffusion_steps, device), initial_noise,
                                                inference_steps, cfg_scale)
        future = decode_future(state, frame, scales)
    finally:
        model.train(was_training)
    # The returned trajectory is un-clipped so generation failures remain measurable.
    xy = torch.cat((batch["pose0"][:, None], future), 1)
    start_logits = torch.where(batch["visible0"], torch.logit(batch["score0"].clamp(1e-6, 1-1e-6)), -20.)
    logits = torch.cat((start_logits[:, None], future_logits), 1)
    visible = (logits >= 0) & torch.isfinite(xy).all(-1) & ((xy >= 0) & (xy <= 1)).all(-1)
    visible[:, 0] = batch["visible0"]
    return {"xy_norm": xy, "visibility_logit": logits, "render_visible": visible,
            "timestamps": batch["timestamps"], "image_size": batch["image_size"]}


@torch.inference_mode()
def generate_motion(model, scales, text_conditioner, prompt, pose0_xy, pose0_visible,
                    image_size, timestamps, seed=42, pose0_score=None,
                    diffusion_steps=1000, inference_steps=50, cfg_scale=1.):
    """Generate from ONLY a sentence, observed pose, geometry, timestamps, and noise.

    Single-pose API. Returns native 20-frame tensors and parser metadata. No
    teacher trajectory, clip ID, or annotation path is accepted or accessed.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Supply a nonempty motion prompt")
    device = next(model.parameters()).device
    pose0 = torch.as_tensor(pose0_xy, dtype=torch.float32, device=device)
    visible = torch.as_tensor(pose0_visible, dtype=torch.bool, device=device)
    score = visible.float() if pose0_score is None else torch.as_tensor(pose0_score, dtype=torch.float32, device=device)
    if score.shape != (20,) or not torch.isfinite(score[visible]).all() or ((score[visible] < 0) | (score[visible] > 1)).any():
        raise ValueError("Observed pose scores must be finite and in [0,1]")
    batch = {"pose0": pose0[None], "visible0": visible[None], "score0": torch.where(visible, score, 0)[None],
             "image_size": torch.as_tensor(image_size, dtype=torch.float32, device=device)[None],
             "timestamps": torch.as_tensor(timestamps, dtype=torch.float32, device=device)[None]}
    hidden, mask = text_conditioner.encode([prompt], device)
    codes, description = parse_prompt(prompt)
    text = {"text": hidden.float(), "text_mask": mask.bool(), "controls": codes[None].to(device)}
    noise = torch.randn(1, 19, 20, 2, generator=torch.Generator().manual_seed(seed)).to(device)
    result = generate_batch(model, scales.to(device), batch, text, noise, diffusion_steps, inference_steps, cfg_scale)
    return {**{k: v[0] for k, v in result.items()}, "prompt_controls": description, "control_codes": codes}


def load_generator(path, device="cpu", weights="selected"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "dmtbaby.stage1.v7":
        raise ValueError("Expected a NEST checkpoint; incompatible checkpoint format")
    cfg = V7Config(**payload["config"])
    choice = payload["selected_weights"] if weights == "selected" else weights
    if choice not in ("raw", "ema"):
        raise ValueError("weights must be selected, raw, or ema")
    model = ResidualMotionDenoiser(cfg.width, cfg.layers, cfg.heads, cfg.dropout)
    if choice == "raw" and "model" not in payload:
        raise ValueError("This is an EMA-only snapshot; use --weights ema or selected")
    model.load_state_dict(payload["model"] if choice == "raw" else payload["ema"])
    model.to(device).eval()
    return model, payload["scales"].to(device), cfg, dict(payload, loaded_weights=choice)
