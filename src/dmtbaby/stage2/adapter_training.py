"""Small, dependency-light contract for UniAnimate adapter fine-tuning.

The project fine-tunes only two parameter families:

* rank-128 LoRA parameters inside ``pipe.dit``; and
* the dynamic RGB DWPose encoder, ``dwpose_embedding``.

The reference-pose encoder, ``randomref_embedding_pose``, is loaded from the
official adapter but remains frozen.  Torch is imported only by checkpoint and
optimizer entry points, so manifests and configuration can be inspected on a
CPU-only machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping


OFFICIAL_LORA_RANK = 128
DEFAULT_TARGET_MODULES = ("q", "k", "v", "o", "ffn.0", "ffn.2")
RESUME_FORMAT = "dmtbaby.stage2.resume"
RESUME_VERSION = 1


@dataclass(frozen=True)
class AdapterTrainingConfig:
    """Reproducible adapter-training policy for the Stage-2 model."""

    lora_rank: int = OFFICIAL_LORA_RANK
    lora_alpha: int = OFFICIAL_LORA_RANK
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    lora_lr: float = 1e-5
    pose_lr: float = 5e-5
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    precision: str = "bf16"
    deterministic: bool = True
    allow_tf32: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_offload: bool = True
    max_steps: int = 1000
    validation_every_steps: int = 250
    resume_every_steps: int = 100
    save_best: bool = True
    save_last: bool = True
    save_resume: bool = True
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    train_dwpose_embedding: bool = True
    train_reference_pose_encoder: bool = False

    def __post_init__(self) -> None:
        if self.lora_rank != OFFICIAL_LORA_RANK:
            raise ValueError(
                f"Stage-2 uses the official rank-{OFFICIAL_LORA_RANK} LoRA architecture; "
                f"got rank {self.lora_rank}."
            )
        if self.lora_alpha != OFFICIAL_LORA_RANK:
            raise ValueError(
                f"Stage-2 rank-{OFFICIAL_LORA_RANK} training requires lora_alpha="
                f"{OFFICIAL_LORA_RANK}; got {self.lora_alpha}."
            )
        if self.target_modules != DEFAULT_TARGET_MODULES:
            raise ValueError(
                "Stage-2 LoRA targets are locked to " + ",".join(DEFAULT_TARGET_MODULES) + "."
            )
        if not self.train_dwpose_embedding:
            raise ValueError("dwpose_embedding must remain trainable for the Stage-2 adapter phase.")
        if self.train_reference_pose_encoder:
            raise ValueError("randomref_embedding_pose is locked frozen by the Stage-2 training contract.")
        if self.lora_lr != 1e-5 or self.pose_lr != 5e-5:
            raise ValueError("Locked Stage-2 rates are lora_lr=1e-5 and pose_lr=5e-5.")
        if self.weight_decay != 0.0:
            raise ValueError("Stage-2 AdamW weight_decay is locked to 0.0.")
        if self.optimizer != "adamw":
            raise ValueError("Stage-2 optimizer is locked to AdamW.")
        if self.scheduler != "cosine" or self.warmup_steps != 100:
            raise ValueError("Stage-2 scheduling is locked to cosine with 100 warmup steps.")
        if self.batch_size != 1 or self.gradient_accumulation_steps != 4:
            raise ValueError("Stage-2 training is locked to batch_size=1 and gradient_accumulation_steps=4.")
        if self.precision != "bf16":
            raise ValueError("Stage-2 training precision is locked to BF16.")
        if not self.deterministic or self.allow_tf32:
            raise ValueError("Stage-2 training is locked to deterministic kernels with TF32 disabled.")
        if not self.gradient_checkpointing or not self.gradient_checkpointing_offload:
            raise ValueError("Stage-2 requires gradient checkpointing with CPU offload.")
        if self.max_steps != 1000:
            raise ValueError("Stage-2 max_steps is locked to 1000.")
        if self.validation_every_steps != 250 or self.resume_every_steps != 100:
            raise ValueError("Stage-2 validation/resume cadence is locked to 250/100 optimizer steps.")
        if not self.save_best or not self.save_last or not self.save_resume:
            raise ValueError("Stage-2 checkpoint policy requires best, last, and one rolling resume checkpoint.")
        if self.monitor_metric != "val_loss" or self.monitor_mode != "min":
            raise ValueError("Stage-2 checkpoint selection is locked to minimum val_loss.")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class TrainableParameterSummary:
    dit_lora: tuple[str, ...]
    dwpose_embedding: tuple[str, ...]

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.dit_lora + self.dwpose_embedding


@dataclass(frozen=True)
class AdapterTrainingInterface:
    """Framework-neutral façade used by a Lightning/Accelerate training loop.

    Constructing this object has no torch side effects.  Torch is imported only
    when ``optimizer`` or a path-based checkpoint method is actually invoked.
    """

    config: AdapterTrainingConfig = AdapterTrainingConfig()

    def prepare(self, model: Any) -> TrainableParameterSummary:
        return configure_trainable_parameters(model, self.config)

    def initialize_from_official(self, model: Any, checkpoint: str | Path | Mapping[str, Any]) -> Any:
        return load_official_adapter(model, checkpoint, self.config)

    def resume(self, model: Any, checkpoint: str | Path | Mapping[str, Any]) -> Any:
        return load_training_adapter(model, checkpoint, self.config)

    def optimizer(self, model: Any) -> Any:
        return create_adamw(model, self.config)

    def lr_scheduler(self, optimizer: Any) -> Any:
        return create_cosine_scheduler(optimizer, self.config)

    def save(
        self,
        path: str | Path,
        model: Any,
        *,
        step: int | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_training_adapter(
            path,
            model,
            self.config,
            step=step,
            extra_metadata=extra_metadata,
        )


def _component_in_name(name: str, component: str) -> bool:
    return component in name.split(".")


def is_dit_lora_parameter(name: str) -> bool:
    """Return whether ``name`` is a LoRA matrix attached inside the Wan DiT."""

    parts = name.split(".")
    has_dit = "dit" in parts and ("pipe" in parts or parts[0] == "dit")
    has_lora = any(part == "lora_A" or part == "lora_B" or part.startswith("lora_") for part in parts)
    return has_dit and has_lora


def is_targeted_dit_lora_parameter(name: str, target_modules: tuple[str, ...]) -> bool:
    if not is_dit_lora_parameter(name):
        return False
    module_name = name.split(".lora_", 1)[0]
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def is_dwpose_embedding_parameter(name: str) -> bool:
    return _component_in_name(name, "dwpose_embedding")


def is_reference_pose_parameter(name: str) -> bool:
    return _component_in_name(name, "randomref_embedding_pose")


def is_allowed_trainable_parameter(
    name: str,
    config: AdapterTrainingConfig | None = None,
) -> bool:
    config = config or AdapterTrainingConfig()
    return is_targeted_dit_lora_parameter(name, config.target_modules) or is_dwpose_embedding_parameter(name)


def _set_requires_grad(parameter: Any, enabled: bool) -> None:
    setter = getattr(parameter, "requires_grad_", None)
    if callable(setter):
        setter(enabled)
    else:
        parameter.requires_grad = enabled


def configure_trainable_parameters(
    model: Any,
    config: AdapterTrainingConfig | None = None,
    *,
    require_complete: bool = True,
) -> TrainableParameterSummary:
    """Freeze the full model, then enable only DiT LoRA and dynamic pose encoder."""

    config = config or AdapterTrainingConfig()
    lora_names: list[str] = []
    dwpose_names: list[str] = []
    for name, parameter in model.named_parameters():
        if is_reference_pose_parameter(name):
            enabled = False
        elif is_targeted_dit_lora_parameter(name, config.target_modules):
            enabled = True
            lora_names.append(name)
        elif is_dwpose_embedding_parameter(name) and config.train_dwpose_embedding:
            enabled = True
            dwpose_names.append(name)
        else:
            enabled = False
        _set_requires_grad(parameter, enabled)

    if require_complete and not lora_names:
        raise RuntimeError("No rank-128 DiT LoRA parameters were found on the training model.")
    if require_complete and not dwpose_names:
        raise RuntimeError("No dwpose_embedding parameters were found on the training model.")

    ranks = infer_lora_ranks({name: parameter for name, parameter in model.named_parameters() if name in lora_names})
    if lora_names and ranks != {config.lora_rank}:
        rendered = sorted(ranks) if ranks else "unknown"
        raise RuntimeError(f"Training model LoRA rank must be {config.lora_rank}; inferred {rendered}.")
    observed_targets = {
        target
        for target in config.target_modules
        if any(is_targeted_dit_lora_parameter(name, (target,)) for name in lora_names)
    }
    missing_targets = sorted(set(config.target_modules) - observed_targets)
    if require_complete and missing_targets:
        raise RuntimeError("Training model is missing LoRA target modules: " + ", ".join(missing_targets))

    reference_encoder = getattr(model, "randomref_embedding_pose", None)
    if reference_encoder is not None:
        requires_grad = getattr(reference_encoder, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad(False)
        evaluate = getattr(reference_encoder, "eval", None)
        if callable(evaluate):
            evaluate()

    summary = TrainableParameterSummary(tuple(lora_names), tuple(dwpose_names))
    assert_trainable_allowlist(model, config=config, require_complete=require_complete)
    return summary


def assert_trainable_allowlist(
    model: Any,
    config: AdapterTrainingConfig | None = None,
    *,
    require_complete: bool = True,
) -> TrainableParameterSummary:
    """Fail closed if any full-model or reference-encoder parameter is trainable."""

    config = config or AdapterTrainingConfig()
    lora_names: list[str] = []
    dwpose_names: list[str] = []
    forbidden: list[str] = []
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        if is_targeted_dit_lora_parameter(name, config.target_modules):
            lora_names.append(name)
        elif is_dwpose_embedding_parameter(name):
            dwpose_names.append(name)
        else:
            forbidden.append(name)
    if forbidden:
        raise RuntimeError("Forbidden trainable parameters: " + ", ".join(forbidden[:12]))
    if require_complete and (not lora_names or not dwpose_names):
        raise RuntimeError(
            "Trainable set is incomplete: both pipe.dit LoRA and dwpose_embedding parameters are required."
        )
    if lora_names:
        named = dict(model.named_parameters())
        ranks = infer_lora_ranks({name: named[name] for name in lora_names})
        if ranks != {config.lora_rank}:
            rendered = sorted(ranks) if ranks else "unknown"
            raise RuntimeError(f"Training model LoRA rank must be {config.lora_rank}; inferred {rendered}.")
    observed_targets = {
        target
        for target in config.target_modules
        if any(is_targeted_dit_lora_parameter(name, (target,)) for name in lora_names)
    }
    missing_targets = sorted(set(config.target_modules) - observed_targets)
    if require_complete and missing_targets:
        raise RuntimeError("Training model is missing LoRA target modules: " + ", ".join(missing_targets))
    return TrainableParameterSummary(tuple(lora_names), tuple(dwpose_names))


def optimizer_parameter_groups(model: Any, config: AdapterTrainingConfig | None = None) -> list[dict[str, Any]]:
    """Return named AdamW groups after enforcing the allowlist."""

    config = config or AdapterTrainingConfig()
    assert_trainable_allowlist(model, config=config)
    lora: list[Any] = []
    dwpose: list[Any] = []
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        if is_targeted_dit_lora_parameter(name, config.target_modules):
            lora.append(parameter)
        elif is_dwpose_embedding_parameter(name):
            dwpose.append(parameter)
    return [
        {"name": "dit_lora", "params": lora, "lr": config.lora_lr, "weight_decay": config.weight_decay},
        {"name": "dwpose_embedding", "params": dwpose, "lr": config.pose_lr, "weight_decay": config.weight_decay},
    ]


def create_adamw(model: Any, config: AdapterTrainingConfig | None = None) -> Any:
    """Create the optimizer; torch stays a lazy training-only dependency."""

    import torch  # type: ignore

    config = config or AdapterTrainingConfig()
    return torch.optim.AdamW(optimizer_parameter_groups(model, config))


def create_cosine_scheduler(optimizer: Any, config: AdapterTrainingConfig | None = None) -> Any:
    """Create the locked 100-step warmup + cosine decay schedule."""

    import torch  # type: ignore

    config = config or AdapterTrainingConfig()

    def scale(step: int) -> float:
        if step < config.warmup_steps:
            return float(step + 1) / float(config.warmup_steps)
        progress = min(
            1.0,
            float(step - config.warmup_steps) / float(config.max_steps - config.warmup_steps),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def infer_lora_ranks(state_dict: Mapping[str, Any]) -> set[int]:
    """Infer PEFT LoRA ranks without importing torch."""

    ranks: set[int] = set()
    for name, value in state_dict.items():
        if not is_dit_lora_parameter(name):
            continue
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) < 2:
            continue
        parts = name.split(".")
        if "lora_A" in parts:
            ranks.add(int(shape[0]))
        elif "lora_B" in parts:
            ranks.add(int(shape[1]))
    return ranks


def validate_official_adapter_state_dict(state_dict: Mapping[str, Any]) -> None:
    """Validate an official initialization checkpoint before loading it."""

    lora_keys = [name for name in state_dict if is_dit_lora_parameter(name)]
    dwpose_keys = [name for name in state_dict if is_dwpose_embedding_parameter(name)]
    reference_keys = [name for name in state_dict if is_reference_pose_parameter(name)]
    if not lora_keys:
        raise ValueError("Official adapter checkpoint contains no pipe.dit LoRA parameters.")
    if not dwpose_keys:
        raise ValueError("Official adapter checkpoint contains no dwpose_embedding parameters.")
    if not reference_keys:
        raise ValueError("Official adapter checkpoint contains no randomref_embedding_pose parameters.")
    ranks = infer_lora_ranks(state_dict)
    if ranks != {OFFICIAL_LORA_RANK}:
        rendered = sorted(ranks) if ranks else "unknown"
        raise ValueError(f"Expected official LoRA rank {OFFICIAL_LORA_RANK}; inferred {rendered}.")
    unmatched = [
        name
        for name in lora_keys
        if not is_targeted_dit_lora_parameter(name, DEFAULT_TARGET_MODULES)
    ]
    if unmatched:
        raise ValueError("Official adapter has unsupported LoRA targets: " + ", ".join(unmatched[:12]))
    missing_targets = [
        target
        for target in DEFAULT_TARGET_MODULES
        if not any(is_targeted_dit_lora_parameter(name, (target,)) for name in lora_keys)
    ]
    if missing_targets:
        raise ValueError("Official adapter is missing LoRA targets: " + ", ".join(missing_targets))


def validate_official_adapter_checkpoint(path: str | Path) -> tuple[str, ...]:
    """Memory-map and validate the official initializer before model allocation."""

    import torch  # type: ignore

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(f"Official Stage-2 adapter is missing or empty: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state_dict = _unwrap_checkpoint(payload)
    validate_official_adapter_state_dict(state_dict)
    return tuple(str(name) for name in state_dict)


def adapter_checkpoint_state_dict(
    model_or_state: Any,
    config: AdapterTrainingConfig | None = None,
) -> dict[str, Any]:
    """Extract only resumable trainable state; the frozen reference encoder is excluded."""

    config = config or AdapterTrainingConfig()
    state_dict = model_or_state if isinstance(model_or_state, Mapping) else model_or_state.state_dict()
    selected = {
        name: value
        for name, value in state_dict.items()
        if is_allowed_trainable_parameter(name, config)
    }
    if not any(is_dit_lora_parameter(name) for name in selected):
        raise ValueError("Adapter state contains no DiT LoRA parameters.")
    if not any(is_dwpose_embedding_parameter(name) for name in selected):
        raise ValueError("Adapter state contains no dwpose_embedding parameters.")
    if any(is_reference_pose_parameter(name) for name in selected):
        raise AssertionError("Frozen reference-pose parameters leaked into the adapter checkpoint.")
    return selected


def validate_training_adapter_keys(
    keys: Any,
    config: AdapterTrainingConfig | None = None,
) -> tuple[str, ...]:
    """Validate the names-only contract of a compact Stage-2 checkpoint."""

    config = config or AdapterTrainingConfig()
    names = tuple(str(name) for name in keys)
    if len(names) != len(set(names)):
        raise ValueError("Training adapter checkpoint contains duplicate parameter names.")
    forbidden = [name for name in names if not is_allowed_trainable_parameter(name, config)]
    if forbidden:
        raise ValueError("Training adapter contains forbidden keys: " + ", ".join(forbidden[:12]))
    if any(is_reference_pose_parameter(name) for name in names):
        raise ValueError("Training adapter must not contain frozen randomref_embedding_pose parameters.")
    if not any(is_dwpose_embedding_parameter(name) for name in names):
        raise ValueError("Training adapter contains no dwpose_embedding parameters.")
    for target in config.target_modules:
        if not any(is_targeted_dit_lora_parameter(name, (target,)) for name in names):
            raise ValueError(f"Training adapter contains no DiT LoRA parameters for target {target!r}.")
    return names


def validate_training_adapter_checkpoint(
    path: str | Path,
    config: AdapterTrainingConfig | None = None,
) -> tuple[str, ...]:
    """Memory-map and validate a compact adapted inference checkpoint."""

    import torch  # type: ignore

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Compact Stage-2 adapter is missing or empty: {checkpoint}"
        )
    payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    state_dict = _unwrap_checkpoint(payload)
    names = validate_training_adapter_keys(state_dict, config)
    ranks = infer_lora_ranks(state_dict)
    if ranks != {OFFICIAL_LORA_RANK}:
        rendered = sorted(ranks) if ranks else "unknown"
        raise ValueError(
            f"Compact Stage-2 adapter must retain rank {OFFICIAL_LORA_RANK}; "
            f"inferred {rendered}."
        )
    return names


def capture_training_rng_state() -> dict[str, Any]:
    """Capture all RNG streams required by an exact Stage-2 resume."""

    import torch  # type: ignore

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np  # type: ignore

        bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        state["numpy"] = {
            "bit_generator": str(bit_generator),
            "keys": torch.from_numpy(keys.copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_training_rng_state(state: Mapping[str, Any]) -> None:
    import torch  # type: ignore

    required = {"python", "torch_cpu"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError("Stage-2 resume RNG state is missing: " + ", ".join(missing))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Resume contains NumPy RNG state but NumPy is unavailable.") from exc
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping):
            raise ValueError("Stage-2 resume NumPy RNG state is malformed.")
        required_numpy = {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
        if not required_numpy.issubset(numpy_state):
            raise ValueError("Stage-2 resume NumPy RNG state is incomplete.")
        keys = numpy_state["keys"]
        if hasattr(keys, "cpu"):
            keys = keys.cpu().numpy()
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(keys, dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Resume contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_training_resume_payload(
    payload: Mapping[str, Any],
    *,
    config: AdapterTrainingConfig | None = None,
    dataset_manifest_sha256: str,
    patient_split_sha256: str,
    dataset_content_sha256: str,
    training_source_sha256: str,
    model_artifacts_sha256: str,
    runtime_environment_sha256: str,
    seed: int,
) -> int:
    """Fail closed on resume state, optimizer scope, and data fingerprints."""

    config = config or AdapterTrainingConfig()
    if payload.get("format") != RESUME_FORMAT or payload.get("version") != RESUME_VERSION:
        raise ValueError("Not a supported DMTBaby Stage-2 rolling resume checkpoint.")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Stage-2 resume has no mapping-valued state_dict.")
    names = validate_training_adapter_keys(state_dict, config)
    metadata = payload.get("dmtbaby_resume")
    if not isinstance(metadata, Mapping):
        raise ValueError("Stage-2 resume has no dmtbaby_resume metadata.")
    if metadata.get("training_config") != asdict(config):
        raise ValueError("Stage-2 resume training configuration does not exactly match this run.")
    expected_fingerprints = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "patient_split_sha256": patient_split_sha256,
        "dataset_content_sha256": dataset_content_sha256,
        "training_source_sha256": training_source_sha256,
        "model_artifacts_sha256": model_artifacts_sha256,
        "runtime_environment_sha256": runtime_environment_sha256,
    }
    actual_fingerprints = {
        name: metadata.get(name)
        for name in expected_fingerprints
    }
    if actual_fingerprints != expected_fingerprints:
        raise ValueError(
            f"Stage-2 resume dataset fingerprints do not match: expected {expected_fingerprints}, "
            f"got {actual_fingerprints}."
        )
    if metadata.get("seed") != seed:
        raise ValueError(f"Stage-2 resume seed must be {seed}; got {metadata.get('seed')!r}.")
    step = metadata.get("step")
    if not isinstance(step, int) or not 0 < step <= config.max_steps:
        raise ValueError(f"Stage-2 resume step must be in [1,{config.max_steps}]; got {step!r}.")
    if payload.get("global_step") != step:
        raise ValueError("Stage-2 resume global_step disagrees with dmtbaby_resume.step.")

    optimizer_states = payload.get("optimizer_states")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise ValueError("Stage-2 resume must contain exactly one AdamW optimizer state.")
    optimizer = optimizer_states[0]
    if not isinstance(optimizer, Mapping):
        raise ValueError("Stage-2 resume optimizer state is malformed.")
    parameter_groups = optimizer.get("param_groups")
    if not isinstance(parameter_groups, list) or len(parameter_groups) != 2:
        raise ValueError("Stage-2 resume optimizer must contain exactly two allowed parameter groups.")
    expected_lrs = {"dit_lora": config.lora_lr, "dwpose_embedding": config.pose_lr}
    actual_lrs: dict[str, float] = {}
    parameter_ids: list[Any] = []
    for group in parameter_groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("Stage-2 resume optimizer parameter group is malformed.")
        name = group.get("name")
        if name not in expected_lrs:
            raise ValueError(f"Stage-2 resume optimizer has forbidden group {name!r}.")
        actual_lrs[str(name)] = float(group.get("initial_lr", group.get("lr")))
        parameter_ids.extend(group["params"])
    if actual_lrs != expected_lrs:
        raise ValueError(f"Stage-2 resume optimizer rates must be {expected_lrs}; got {actual_lrs}.")
    if len(parameter_ids) != len(set(parameter_ids)) or len(parameter_ids) != len(names):
        raise ValueError(
            "Stage-2 resume optimizer parameter IDs do not exactly cover the allowed trainable state."
        )
    optimizer_values = optimizer.get("state")
    if not isinstance(optimizer_values, Mapping) or set(optimizer_values) != set(parameter_ids):
        raise ValueError("Stage-2 resume AdamW moments do not exactly cover allowed parameters.")
    for parameter_id, parameter_state in optimizer_values.items():
        if not isinstance(parameter_state, Mapping) or not {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }.issubset(parameter_state):
            raise ValueError(
                f"Stage-2 resume AdamW state is incomplete for parameter ID {parameter_id!r}."
            )

    schedulers = payload.get("lr_schedulers")
    if not isinstance(schedulers, list) or len(schedulers) != 1:
        raise ValueError("Stage-2 resume must contain exactly one cosine scheduler state.")
    scheduler_state = schedulers[0]
    if not isinstance(scheduler_state, Mapping) or scheduler_state.get("last_epoch") != step:
        raise ValueError("Stage-2 resume cosine scheduler progress disagrees with global_step.")
    if "grad_scaler" not in metadata:
        raise ValueError("Stage-2 resume must explicitly record scaler state (None for BF16).")
    rng_state = metadata.get("rng_state")
    if not isinstance(rng_state, Mapping) or not {"python", "torch_cpu"}.issubset(rng_state):
        raise ValueError("Stage-2 resume RNG state is incomplete.")
    sampler_state = metadata.get("sampler_state")
    if not isinstance(sampler_state, Mapping):
        raise ValueError("Stage-2 resume sampler state is missing.")
    required_sampler = {
        "generator_state",
        "permutation",
        "position",
        "epoch",
        "samples_yielded",
        "total_samples",
    }
    if not required_sampler.issubset(sampler_state):
        raise ValueError("Stage-2 resume sampler state is incomplete.")
    if not isinstance(sampler_state["position"], int) or sampler_state["position"] < 0:
        raise ValueError("Stage-2 resume sampler position is invalid.")
    expected_microbatches = step * config.gradient_accumulation_steps
    if sampler_state.get("samples_yielded") != expected_microbatches:
        raise ValueError(
            "Stage-2 resume sampler progress does not match optimizer/global-step progress."
        )
    total_microbatches = config.max_steps * config.gradient_accumulation_steps
    if sampler_state.get("total_samples") != total_microbatches:
        raise ValueError("Stage-2 resume sampler length does not match the locked training run.")
    if "loops" not in payload:
        raise ValueError("Stage-2 resume is missing Lightning loop progress state.")
    callbacks = payload.get("callbacks")
    if not isinstance(callbacks, Mapping):
        raise ValueError("Stage-2 resume is missing callback selection/progress state.")
    compact_state = callbacks.get("dmtbaby.stage2.CompactBestLast")
    rolling_state = callbacks.get("dmtbaby.stage2.RollingResume")
    if not isinstance(compact_state, Mapping) or not isinstance(rolling_state, Mapping):
        raise ValueError("Stage-2 resume is missing compact-best or rolling-resume callback state.")
    expected_validation_steps = tuple(
        range(config.validation_every_steps, step + 1, config.validation_every_steps)
    )
    actual_validation_steps = tuple(int(value) for value in compact_state.get("validation_steps", ()))
    if (
        compact_state.get("validation_runs") != len(expected_validation_steps)
        or actual_validation_steps != expected_validation_steps
    ):
        raise ValueError("Stage-2 resume validation-selection progress is inconsistent with global_step.")
    if rolling_state.get("last_saved_step") != step:
        raise ValueError("Stage-2 rolling callback progress disagrees with global_step.")
    return step


def read_training_resume(
    path: str | Path,
    *,
    config: AdapterTrainingConfig | None = None,
    dataset_manifest_sha256: str,
    patient_split_sha256: str,
    dataset_content_sha256: str,
    training_source_sha256: str,
    model_artifacts_sha256: str,
    runtime_environment_sha256: str,
    seed: int,
) -> Mapping[str, Any]:
    """Memory-map and validate a trusted local rolling resume checkpoint."""

    import torch  # type: ignore

    resume_path = Path(path).expanduser().resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"Stage-2 rolling resume checkpoint does not exist: {resume_path}")
    payload = torch.load(
        resume_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-2 resume checkpoint root must be a mapping.")
    validate_training_resume_payload(
        payload,
        config=config,
        dataset_manifest_sha256=dataset_manifest_sha256,
        patient_split_sha256=patient_split_sha256,
        dataset_content_sha256=dataset_content_sha256,
        training_source_sha256=training_source_sha256,
        model_artifacts_sha256=model_artifacts_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
        seed=seed,
    )
    return payload


def _unwrap_checkpoint(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a mapping.")
    candidate = payload.get("state_dict", payload)
    if not isinstance(candidate, Mapping):
        raise TypeError("Checkpoint 'state_dict' must be a mapping.")
    return candidate


def load_official_adapter(
    model: Any,
    checkpoint: str | Path | Mapping[str, Any],
    config: AdapterTrainingConfig | None = None,
) -> Any:
    """Load official rank-128 weights, then apply the frozen-reference policy."""

    if isinstance(checkpoint, (str, Path)):
        import torch  # type: ignore

        payload = torch.load(Path(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    else:
        payload = checkpoint
    state_dict = _unwrap_checkpoint(payload)
    validate_official_adapter_state_dict(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", None)
    unexpected = getattr(incompatible, "unexpected_keys", None)
    if missing is None and isinstance(incompatible, tuple) and len(incompatible) == 2:
        missing, unexpected = incompatible
    missing = tuple(missing or ())
    unexpected = tuple(unexpected or ())
    required_model_keys = {
        name
        for name, _ in model.named_parameters()
        if is_allowed_trainable_parameter(name, config) or is_reference_pose_parameter(name)
    }
    missing_adapter_keys = sorted(required_model_keys.intersection(missing))
    if missing_adapter_keys:
        raise ValueError(
            "Official adapter does not initialize every constructed LoRA/pose parameter: "
            + ", ".join(missing_adapter_keys[:12])
        )
    if unexpected:
        raise ValueError(
            "Official adapter has keys not accepted by the constructed rank-128 model: "
            + ", ".join(str(name) for name in unexpected[:12])
        )
    configure_trainable_parameters(model, config)
    return incompatible


def load_training_adapter(
    model: Any,
    checkpoint: str | Path | Mapping[str, Any],
    config: AdapterTrainingConfig | None = None,
) -> Any:
    """Resume the trainable-only checkpoint after official initialization."""

    if isinstance(checkpoint, (str, Path)):
        import torch  # type: ignore

        payload = torch.load(Path(checkpoint), map_location="cpu", mmap=True, weights_only=True)
    else:
        payload = checkpoint
    state_dict = dict(_unwrap_checkpoint(payload))
    config = config or AdapterTrainingConfig()
    validate_training_adapter_keys(state_dict, config)
    state_dict = adapter_checkpoint_state_dict(state_dict, config)
    incompatible = model.load_state_dict(state_dict, strict=False)
    configure_trainable_parameters(model, config)
    return incompatible


def save_training_adapter(
    path: str | Path,
    model: Any,
    config: AdapterTrainingConfig | None = None,
    *,
    step: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save a compact, upstream-loadable adapter state dict.

    UniAnimate's existing ``ModelManager.load_lora_v2`` expects checkpoint
    parameter names at the top level.  Reproducibility metadata and trainer
    state therefore live in the separate rolling ``resume.pt`` checkpoint,
    while ``best.ckpt`` and ``last.ckpt`` remain drop-in inference adapters.
    ``step`` and ``extra_metadata`` are accepted for interface compatibility;
    they are intentionally not embedded as non-tensor top-level entries.
    """

    import torch  # type: ignore

    config = config or AdapterTrainingConfig()
    assert_trainable_allowlist(model, config=config)
    del step, extra_metadata
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter_checkpoint_state_dict(model, config)

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
