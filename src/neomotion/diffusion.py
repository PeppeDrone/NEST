"""Clean-state diffusion, absolute sample weighting, and warmed-up FP32 EMA."""
import math
import torch
from torch.nn import functional as F

from .motion import motion_losses
from .representation import masked_sample_mean


def cosine_schedule(steps, device):
    u = torch.linspace(0, 1, steps+1, dtype=torch.float64, device=device)
    alpha_bar = torch.cos((u+.008)/1.008 * math.pi/2).square()
    alpha_bar = (alpha_bar/alpha_bar[0])[1:].float()
    alpha_bar[-1] = 1e-6
    return alpha_bar


def training_loss(model, target, valid, model_inputs, alpha_bar, timesteps, noise,
                  visibility_weight, velocity_weight=0., state_scales=None, sample_weights=None, motion=None):
    """``motion``: optional dict(controls, dropped, bounds, magnitude_weight, label_weight, eps) for NEST terms."""
    alpha = alpha_bar[timesteps].sqrt()[:, None, None, None]
    sigma = (1-alpha_bar[timesteps]).sqrt()[:, None, None, None]
    noisy = alpha*target + sigma*noise
    clean, logits = model(noisy, timesteps, **model_inputs)
    coordinate = masked_sample_mean((clean.float()-target).square().mean(-1), valid, sample_weights)
    visibility = F.binary_cross_entropy_with_logits(logits.float(), valid.float(), reduction="none").mean((1, 2))
    visibility = visibility.mean() if sample_weights is None else (visibility*sample_weights).mean()
    velocity = coordinate * 0
    if velocity_weight:
        if state_scales is None:
            raise ValueError("Physical velocity loss requires state_scales")
        zeros = torch.zeros_like(target[:, :1])
        pred = torch.cat((zeros, clean.float()*state_scales[:, None]), 1)
        truth = torch.cat((zeros, target*state_scales[:, None]), 1)
        dt = torch.diff(model_inputs["timestamps"], dim=1)[:, :, None, None]
        vm = torch.cat((model_inputs["visible"][:, None], valid), 1)
        vm = vm[:, 1:] & vm[:, :-1]
        error = F.smooth_l1_loss(torch.diff(pred, dim=1)/dt, torch.diff(truth, dim=1)/dt, reduction="none", beta=.1).mean(-1)
        velocity = masked_sample_mean(error, vm, sample_weights)
    magnitude = label = coordinate * 0
    if motion is not None and (motion["magnitude_weight"] or motion["label_weight"]):
        if state_scales is None:
            raise ValueError("NEST motion losses require state_scales")
        magnitude, label = motion_losses(clean, target, state_scales, valid, model_inputs["visible"],
                                         model_inputs["timestamps"], motion["controls"], motion["dropped"],
                                         alpha_bar[timesteps], motion["bounds"], motion["eps"])
    loss = coordinate + visibility_weight*visibility + velocity_weight*velocity
    if motion is not None:
        loss = loss + motion["magnitude_weight"]*magnitude + motion["label_weight"]*label
    return {"loss": loss, "coordinate": coordinate, "visibility": visibility, "velocity": velocity,
            "magnitude": magnitude, "label": label}


@torch.no_grad()
def sample_clean(model, model_inputs, alpha_bar, initial_noise, steps=50, cfg_scale=1.):
    if not isinstance(steps, int) or not 1 <= steps <= len(alpha_bar) or not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("Invalid sampler steps or CFG scale")
    timesteps = torch.linspace(len(alpha_bar)-1, 0, steps, device=initial_noise.device).round().long()
    state = initial_noise.float().clone()
    for i, timestep in enumerate(timesteps):
        ts = timestep.expand(len(state))
        clean, logits = model(state, ts, **model_inputs)
        clean = clean.float()
        if cfg_scale != 1:
            uncond_inputs = dict(model_inputs, dropped=torch.ones_like(model_inputs["dropped"]))
            uncond = model(state, ts, **uncond_inputs)[0].float()
            clean = uncond + cfg_scale*(clean-uncond)
        if i == len(timesteps)-1:
            return clean, logits.float()
        a = alpha_bar[timestep].sqrt()
        s = (1-alpha_bar[timestep]).sqrt()
        eps = (state-a*clean) / s.clamp_min(1e-8)
        a_next = alpha_bar[timesteps[i+1]].sqrt()
        s_next = (1-alpha_bar[timesteps[i+1]]).sqrt()
        state = a_next*clean + s_next*eps
    raise AssertionError("No sampling steps")


@torch.no_grad()
def update_ema(shadow, model, step, burnin, decay):
    """Copies trained weights throughout burn-in; FP32 averaging ramps up afterward."""
    effective = 0. if step <= burnin else min(decay, (1+step-burnin)/(10+step-burnin))
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            shadow[name].mul_(effective).add_(value.detach().float(), alpha=1-effective)
        else:
            shadow[name].copy_(value)
    return effective
