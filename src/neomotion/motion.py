"""Per-limb (V6+) motion measures, label-derived bounds, and control-class balancing.

All measures use the unscaled residual state: starting-pose body units in the
torso frame, on initially observed joints only (unobserved joints carry absolute
positions, not displacement). Limb order matches ``conditioning.LIMBS``.
"""
from collections import Counter

import torch
from torch.nn import functional as F

from .common.topology import BODY_LIMB_JOINT_INDICES
from .conditioning import LIMBS

BOUND_NAMES = ("amp_low_min", "amp_high_min", "speed_fast_min", "speed_slow_max", "still_amp_max")


def _limb_mask(device):
    mask = torch.zeros(4, 20, dtype=torch.bool, device=device)
    for i, limb in enumerate(LIMBS):
        mask[i, list(BODY_LIMB_JOINT_INDICES[limb])] = True
    return mask


def limb_motion(residual, valid, visible0, timestamps):
    """residual [B,19,20,2] body units -> amplitude, speed [B,4] and support [B,4]."""
    limbs = _limb_mask(residual.device)                                    # [4,20]
    norm = (residual.float().square().sum(-1) + 1e-8).sqrt()                # [B,19,20]
    amp_mask = (valid & visible0[:, None])[:, None] & limbs[None, :, None]  # [B,4,19,20]
    amp_count = amp_mask.sum((2, 3))
    amplitude = (norm[:, None] * amp_mask).sum((2, 3)) / amp_count.clamp_min(1)
    # Frame 0 residual is exactly zero for observed joints.
    full = torch.cat((torch.zeros_like(residual[:, :1]), residual.float()), 1)
    full_valid = torch.cat((visible0[:, None], valid & visible0[:, None]), 1)
    dt = torch.diff(timestamps.float(), dim=1)[:, :, None]
    step = (torch.diff(full, dim=1).square().sum(-1) + 1e-8).sqrt() / dt   # [B,19,20]
    step_mask = (full_valid[:, 1:] & full_valid[:, :-1])[:, None] & limbs[None, :, None]
    step_count = step_mask.sum((2, 3))
    speed = (step[:, None] * step_mask).sum((2, 3)) / step_count.clamp_min(1)
    return amplitude, speed, (amp_count > 0) & (step_count > 0)


def fit_label_bounds(residual, valid, visible0, timestamps, controls, quantile):
    """Teacher quantiles per parsed label class; returns tensor ordered as BOUND_NAMES.

    A label class with fewer than 5 training limbs (e.g. no amplitude grades in the
    automatic captions) gets NaN, which disables that bound in ``motion_losses``.
    """
    amplitude, speed, support = limb_motion(residual, valid, visible0, timestamps)
    known, moving, amp, spd = controls.unbind(-1)
    moving_known = (known == 1) & (moving == 1) & support
    groups = {
        "amp_low_min": (amplitude, moving_known & (amp == 1), quantile),
        "amp_high_min": (amplitude, moving_known & (amp == 2), quantile),
        "speed_fast_min": (speed, moving_known & (spd == 2), quantile),
        "speed_slow_max": (speed, moving_known & (spd == 1), 1 - quantile),
        "still_amp_max": (amplitude, (known == 1) & (moving == 0) & support, 1 - quantile),
    }
    values, counts = [], {}
    for name in BOUND_NAMES:
        source, mask, q = groups[name]
        values.append(torch.quantile(source[mask].double(), q).float() if mask.sum() >= 5
                      else torch.tensor(float("nan")))
        counts[name] = int(mask.sum())
    return torch.stack(values), counts


def motion_losses(clean, target, state_scales, valid, visible0, timestamps, controls, dropped, alpha_bar_t,
                  bounds, eps):
    """Per-limb magnitude matching and label hinge terms, both weighted by alpha_bar(t)."""
    pred_res = clean.float() * state_scales[:, None]
    true_res = target.float() * state_scales[:, None]
    pred_amp, pred_speed, support = limb_motion(pred_res, valid, visible0, timestamps)
    with torch.no_grad():
        true_amp, _, _ = limb_motion(true_res, valid, visible0, timestamps)
    w = alpha_bar_t.float()[:, None]                                        # [B,1]
    log = lambda x: torch.log(x + eps)
    error = (log(pred_amp) - log(true_amp)).abs()
    magnitude = (w * error * support).sum() / support.sum().clamp_min(1)

    known, moving, amp, spd = controls.unbind(-1)
    usable = support & ~dropped[:, None]
    moving_known = usable & (known == 1) & (moving == 1)
    # Disabled (NaN) bounds are replaced by a dummy value and masked out: no penalty, no count.
    on = {name: bool(torch.isfinite(bounds[i])) for i, name in enumerate(BOUND_NAMES)}
    b = {name: torch.nan_to_num(bounds[i], nan=1.0) for i, name in enumerate(BOUND_NAMES)}
    terms = [
        (moving_known & (amp == 1) & on["amp_low_min"], F.relu(log(b["amp_low_min"]) - log(pred_amp))),
        (moving_known & (amp == 2) & on["amp_high_min"], F.relu(log(b["amp_high_min"]) - log(pred_amp))),
        (moving_known & (spd == 2) & on["speed_fast_min"], F.relu(log(b["speed_fast_min"]) - log(pred_speed))),
        (moving_known & (spd == 1) & on["speed_slow_max"], F.relu(log(pred_speed) - log(b["speed_slow_max"]))),
        (usable & (known == 1) & (moving == 0) & on["still_amp_max"], F.relu(log(pred_amp) - log(b["still_amp_max"]))),
    ]
    total = sum((w * value * mask).sum() for mask, value in terms)
    count = sum(mask.sum() for mask, _ in terms)
    label = total / count.clamp_min(1) if isinstance(count, torch.Tensor) else total * 0
    return magnitude, label


def control_class(codes):
    """Clip-level balancing class: number of moving limbs and the set of their (amp, speed) grades."""
    moving = [(int(c[2]), int(c[3])) for c in codes if c[0] == 1 and c[1] == 1]
    return min(len(moving), 3), tuple(sorted(set(moving)))


def control_balance(controls, power, min_count=10):
    """Multiplicative per-clip factors (inverse class frequency)^power, plus class counts for logging.

    Classes with fewer than ``min_count`` clips are pooled per moving-limb count, so a
    single clip cannot receive a huge sampling weight.
    """
    classes = [control_class(c) for c in controls]
    counts = Counter(classes)
    classes = [c if counts[c] >= min_count else (c[0], (("rare",),)) for c in classes]
    counts = Counter(classes)
    factors = torch.tensor([counts[c] ** -power for c in classes], dtype=torch.float64)
    label = lambda k: f"{k[0]}|" + (",".join("".join(map(str, g)) for g in k[1]) or "-")
    return factors, {label(k): v for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}
