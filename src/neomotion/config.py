"""Flat, versioned recipe shared by the two prompt-source experiments."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math

import yaml


@dataclass(frozen=True)
class V7Config:
    version: int = 7
    width: int = 128
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    batch_size: int = 16
    accumulation: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    max_steps: int = 20000
    warmup_steps: int = 200
    prior_steps: int = 1000
    validate_every: int = 200
    log_every: int = 20
    patience: int = 20
    min_steps: int = 2000
    ema_decay: float = 0.99
    ema_burnin: int = 100
    condition_dropout: float = 0.1
    diffusion_steps: int = 1000
    inference_steps: int = 50
    cfg_scale: float = 2.0
    visibility_weight: float = 0.1
    velocity_weight: float = 0.0
    residual_scale_floor: float = 0.05
    # Motion terms (introduced in V6), all measured per limb in starting-pose body units and
    # weighted by alpha_bar(t), so near-pure-noise steps are not forced off the mean.
    magnitude_weight: float = 0.2      # |log(E_pred+eps) - log(E_teacher+eps)| per limb
    label_weight: float = 0.5          # hinge bounds from the limb's parsed amplitude/speed/still label
    label_quantile: float = 0.25       # lower bounds use this teacher quantile, upper bounds 1 - it
    motion_eps: float = 0.02
    control_balance_power: float = 0.5  # 0 = V5 sampling; 1 = full inverse frequency of control class
    # Caption set that defines the control classes for the draw law. "manual" makes the
    # draw probabilities identical for the manual and automatic runs (NEST decision).
    sampling_caption_source: str = "manual"
    validation_seeds: tuple[int, ...] = (31415, 92653)
    seed: int = 42

    def __post_init__(self):
        if self.version != 7:
            raise ValueError("This recipe is NEST only")
        for name in ("width", "layers", "heads", "batch_size", "accumulation", "max_steps",
                     "validate_every", "log_every", "patience", "diffusion_steps", "inference_steps"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads or self.width < 4:
            raise ValueError("width must be >=4 and divisible by heads")
        if self.diffusion_steps < 2 or self.inference_steps > self.diffusion_steps:
            raise ValueError("invalid diffusion/inference step counts")
        for name in ("dropout", "condition_dropout", "ema_decay"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0,1)")
        if self.sampling_caption_source not in ("manual", "automatic"):
            raise ValueError("sampling_caption_source must be manual or automatic")
        if not 0 < self.label_quantile < 0.5:
            raise ValueError("label_quantile must be in (0, 0.5)")
        for name in ("learning_rate", "residual_scale_floor", "motion_eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "visibility_weight", "velocity_weight", "cfg_scale",
                     "magnitude_weight", "label_weight", "control_balance_power"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("warmup_steps", "prior_steps", "ema_burnin", "min_steps"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.prior_steps >= self.max_steps:
            raise ValueError("prior_steps must leave at least one conditional training step")
        if not self.validation_seeds or any(not isinstance(s, int) or s < 0 for s in self.validation_seeds):
            raise ValueError("validation_seeds must be nonempty nonnegative integers")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("NEST config must be a flat YAML mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown NEST config fields: {sorted(unknown)}")
        if "validation_seeds" in values:
            values["validation_seeds"] = tuple(values["validation_seeds"])
        return cls(**values)
