"""Pixel-isotropic coordinates anchored to the observed pose; no future fitting."""
import torch
from torch import Tensor


def reference_frame(pose: Tensor, visible: Tensor, image_size: Tensor) -> dict[str, Tensor]:
    """Input [B,20,2], [B,20], [B,2] (width,height); return fixed torso transforms."""
    if pose.ndim != 3 or pose.shape[1:] != (20, 2) or visible.shape != pose.shape[:2]:
        raise ValueError("Expected body20 pose [B,20,2] and visibility [B,20]")
    if image_size.shape != (pose.shape[0], 2) or not torch.isfinite(image_size).all() or (image_size <= 0).any():
        raise ValueError("image_size must contain positive pixel width,height for every pose")
    if not visible.any(dim=1).all():
        raise ValueError("At least one observed joint is required for every starting pose")
    observed = pose[visible]
    if not torch.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
        raise ValueError("Observed starting coordinates must be finite and in [0,1]")
    xy = torch.where(visible[..., None], pose.float(), 0) * image_size[:, None]
    origin = (xy * visible[..., None]).sum(1) / visible.sum(1, keepdim=True)
    origin = torch.where(visible[:, 1, None], xy[:, 1], origin)
    hip_mask = visible[:, [8, 11]]
    hips = (xy[:, [8, 11]] * hip_mask[..., None]).sum(1) / hip_mask.sum(1, keepdim=True).clamp_min(1)
    torso = hips - xy[:, 1]
    length = torch.linalg.vector_norm(torso, dim=-1)
    minimum = image_size.amin(-1) * 0.05
    torso_ok = visible[:, 1] & hip_mask.any(-1) & (length >= minimum)
    y_axis = torso / length[:, None].clamp_min(1e-6)
    y_axis = torch.where(torso_ok[:, None], y_axis, y_axis.new_tensor([0., 1.]))
    x_axis = torch.stack((y_axis[:, 1], -y_axis[:, 0]), -1)
    rotation = torch.stack((x_axis, y_axis), -1)
    high = xy.masked_fill(~visible[..., None], -torch.inf).amax(1)
    low = xy.masked_fill(~visible[..., None], torch.inf).amin(1)
    fallback = torch.linalg.vector_norm(high-low, dim=-1) * 0.5
    shoulder_length = torch.linalg.vector_norm(xy[:, 2]-xy[:, 5], dim=-1)
    shoulder_ok = visible[:, 2] & visible[:, 5] & (shoulder_length >= minimum)
    fallback = torch.where(shoulder_ok, shoulder_length, fallback).clamp_min(minimum)
    scale = torch.where(torso_ok, length, fallback)
    canonical = torch.einsum("bji,bik->bjk", xy-origin[:, None], rotation) / scale[:, None, None]
    canonical = torch.where(visible[..., None], canonical, 0)
    return {"origin": origin, "rotation": rotation, "scale": scale, "image_size": image_size.float(),
            "pose": canonical, "visible": visible.bool(), "input_pose": pose.float(),
            "torso_anchor": torso_ok}


def encode_future(xy: Tensor, valid: Tensor, frame: dict[str, Tensor], scales: Tensor | None = None) -> Tensor:
    """Observed joints: displacement. Unobserved joints: canonical absolute position."""
    clean = torch.where(valid[..., None], xy.float(), 0) * frame["image_size"][:, None, None]
    q = torch.einsum("btji,bik->btjk", clean-frame["origin"][:, None, None], frame["rotation"])
    q = q / frame["scale"][:, None, None, None]
    state = q - frame["pose"][:, None]
    if scales is not None:
        branch = (~frame["visible"]).long()
        joint = torch.arange(20, device=xy.device)[None]
        state = state / scales[branch, joint][:, None]
    return torch.where(valid[..., None], state, 0)


def decode_future(state: Tensor, frame: dict[str, Tensor], scales: Tensor) -> Tensor:
    branch = (~frame["visible"]).long()
    joint = torch.arange(20, device=state.device)[None]
    q = state.float() * scales[branch, joint][:, None] + frame["pose"][:, None]
    pixels = torch.einsum("btjk,bik->btji", q, frame["rotation"]) * frame["scale"][:, None, None, None]
    return (pixels + frame["origin"][:, None, None]) / frame["image_size"][:, None, None]


def fit_residual_scales(states: Tensor, valid: Tensor, observed: Tensor, floor: float) -> Tensor:
    """Training-only zero-centred RMS, separately for residual and absolute branches."""
    scales = []
    for branch in [observed, ~observed]:
        mask = valid & branch[:, None]
        numerator = (states.square() * mask[..., None]).sum((0, 1))
        denominator = mask.sum((0, 1))[:, None].clamp_min(1)
        scales.append((numerator / denominator).sqrt().clamp_min(floor))
    return torch.stack(scales)


def masked_sample_mean(values: Tensor, mask: Tensor, weights: Tensor | None = None) -> Tensor:
    """Mean valid tokens per sample, THEN absolute sample weights; never renormalize weights."""
    if values.shape != mask.shape:
        raise ValueError("Values and mask must have the same shape")
    axes = tuple(range(1, values.ndim))
    count = mask.sum(axes)
    per_sample = torch.where(mask, values, 0).sum(axes) / count.clamp_min(1)
    if weights is not None:
        per_sample = per_sample * weights
    present = count > 0
    return (per_sample * present).sum() / present.sum().clamp_min(1)
