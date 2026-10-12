"""Executable UniAnimate adapter training for the locked DMTBaby cascade."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

from ..io import fingerprint, sha256_file
from ..pose.schema import CANONICAL_RENDER_HEIGHT, CANONICAL_RENDER_WIDTH

from .adapter_dataset import (
    AdapterDatasetManifest,
    AdapterDatasetRecord,
    load_adapter_dataset_manifest,
    target_frame_paths,
)
from .adapter_training import (
    AdapterTrainingConfig,
    adapter_checkpoint_state_dict,
    assert_trainable_allowlist,
    configure_trainable_parameters,
    create_adamw,
    create_cosine_scheduler,
    capture_training_rng_state,
    load_official_adapter,
    optimizer_parameter_groups,
    read_training_resume,
    restore_training_rng_state,
    RESUME_FORMAT,
    RESUME_VERSION,
    save_training_adapter,
    validate_training_resume_payload,
    validate_training_adapter_keys,
    validate_official_adapter_checkpoint,
)
from .controls import validate_control_sequence
from .service import UniAnimateModelConfig


TRAIN_HEIGHT = 832
TRAIN_WIDTH = 480
CHECKPOINT_HEADROOM_BYTES = 50 * 1024**3
CHECKPOINT_PEAK_ADAPTER_UNITS = 8


@dataclass(frozen=True)
class AdapterRuntimeReport:
    """Auditable facts returned by a concrete training runtime."""

    best_checkpoint: Path
    last_checkpoint: Path
    resume_checkpoint: Path
    best_checkpoint_keys: tuple[str, ...]
    last_checkpoint_keys: tuple[str, ...]
    resume_checkpoint_keys: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    optimizer_group_lrs: Mapping[str, float]
    steps_completed: int
    validation_runs: int
    validation_steps: tuple[int, ...]
    best_val_loss: float


@dataclass(frozen=True)
class AdapterTrainingResult:
    output_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    resume_checkpoint: Path
    train_examples: int
    val_examples: int
    steps_completed: int
    validation_runs: int
    validation_steps: tuple[int, ...]
    best_val_loss: float
    dataset_manifest_sha256: str
    patient_split_sha256: str
    dataset_content_sha256: str
    training_source_sha256: str
    model_artifacts_sha256: str
    runtime_environment_sha256: str
    seed: int


@dataclass(frozen=True)
class AdapterTrainingFingerprints:
    dataset_manifest_sha256: str
    patient_split_sha256: str
    dataset_content_sha256: str
    training_source_sha256: str
    model_artifacts_sha256: str
    runtime_environment_sha256: str


AdapterRuntime = Callable[..., AdapterRuntimeReport]


def validate_training_device(device: str) -> int | None:
    """Validate a single CUDA device string; return its optional index."""

    value = str(device).strip().lower()
    if value == "cuda":
        return None
    if value.startswith("cuda:") and value[5:].isdigit():
        return int(value[5:])
    raise ValueError(
        "Stage-2 adapter training is CUDA-only; --device must be 'cuda' or 'cuda:<index>'."
    )


def training_source_fingerprint(root: str | Path, config: UniAnimateModelConfig) -> str:
    """Hash local Stage-2 and checked-out upstream Python sources."""

    project_root = Path(root).expanduser().resolve()
    entries: list[tuple[str, str]] = []
    stage2_root = project_root / "dmtbaby" / "stage2"
    for path in sorted(stage2_root.rglob("*.py")) if stage2_root.is_dir() else ():
        entries.append((f"project/{path.relative_to(project_root)}", sha256_file(path)))
    for relative in ("dmtbaby/io.py", "dmtbaby/pose/schema.py"):
        path = project_root / relative
        if path.is_file():
            entries.append((f"project/{relative}", sha256_file(path)))
    upstream_root = Path(config.unianimate_dir).resolve()
    for path in sorted((upstream_root / "diffsynth").rglob("*.py")):
        entries.append((f"unianimate/{path.relative_to(upstream_root)}", sha256_file(path)))
    train_source = upstream_root / "examples" / "unianimate_wan" / "train_unianimate_wan.py"
    if train_source.is_file():
        entries.append(("unianimate/examples/unianimate_wan/train_unianimate_wan.py", sha256_file(train_source)))
    if not entries:
        raise FileNotFoundError("No Stage-2/upstream Python sources were found for reproducibility fingerprinting.")
    return fingerprint(entries)


def training_model_artifacts_fingerprint(config: UniAnimateModelConfig) -> str:
    """Hash the official initializer, frozen Wan weights, and tokenizers."""

    artifacts = validate_training_artifacts(config)
    wan_root = Path(config.wan_dir).resolve()
    unianimate_root = Path(config.unianimate_dir).resolve()
    entries: list[tuple[str, str]] = []
    for index, path in enumerate(artifacts):
        resolved = path.resolve()
        if resolved.suffix == ".py":
            continue
        if resolved == Path(config.lora_path).resolve():
            role = "official_adapter"
        elif resolved.is_relative_to(wan_root):
            role = f"wan/{resolved.relative_to(wan_root)}"
        elif resolved.is_relative_to(unianimate_root):
            role = f"unianimate/{resolved.relative_to(unianimate_root)}"
        else:
            role = f"artifact-{index}/{resolved.name}"
        entries.append((role, sha256_file(resolved)))
    return fingerprint(entries)


def training_runtime_environment_fingerprint(device: str) -> str:
    """Hash runtime versions and the selected CUDA device before allocation."""

    import platform

    try:
        import lightning as pl  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not fingerprint the pinned Stage-2 torch/Lightning/NumPy runtime: "
            f"{exc!r}"
        ) from exc
    device_index = validate_training_device(device)
    selected_index = torch.cuda.current_device() if device_index is None and torch.cuda.is_available() else device_index
    cuda_device: Mapping[str, Any] | None = None
    if selected_index is not None and torch.cuda.is_available() and selected_index < torch.cuda.device_count():
        properties = torch.cuda.get_device_properties(selected_index)
        cuda_device = {
            "index": int(selected_index),
            "name": str(properties.name),
            "capability": tuple(torch.cuda.get_device_capability(selected_index)),
            "total_memory": int(properties.total_memory),
        }
    return fingerprint(
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version() or 0),
            "lightning": str(pl.__version__),
            "numpy": str(np.__version__),
            "cuda_device": cuda_device,
        }
    )


def compute_training_fingerprints(
    manifest: AdapterDatasetManifest,
    root: str | Path,
    config: UniAnimateModelConfig,
) -> AdapterTrainingFingerprints:
    """Compute the exact data/code/model/runtime identity used for resume."""

    return AdapterTrainingFingerprints(
        dataset_manifest_sha256=manifest.manifest_sha256,
        patient_split_sha256=manifest.split_manifest_sha256,
        dataset_content_sha256=manifest.dataset_content_sha256,
        training_source_sha256=training_source_fingerprint(root, config),
        model_artifacts_sha256=training_model_artifacts_fingerprint(config),
        runtime_environment_sha256=training_runtime_environment_fingerprint(str(config.device)),
    )


def validate_training_images(
    manifest: AdapterDatasetManifest,
    *,
    image_module: Any | None = None,
) -> None:
    """Decode all aligned media and enforce canonical RGB DWPose controls."""

    if image_module is None:
        try:
            from PIL import Image as image_module  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required to preflight Stage-2 training media.") from exc

    def inspect(path: Path, *, role: str, expected_format: str | None = None, pose: bool = False) -> None:
        try:
            with image_module.open(path) as image:
                mode = str(image.mode)
                size = tuple(image.size)
                image_format = str(image.format or "").upper()
                image.verify()
        except Exception as exc:
            raise RuntimeError(f"Could not decode Stage-2 {role} image {path}: {exc}") from exc
        if expected_format is not None and image_format != expected_format:
            raise ValueError(
                f"Stage-2 {role} image {path} must be {expected_format}; got {image_format or 'unknown'}."
            )
        if pose and (mode != "RGB" or size != (CANONICAL_RENDER_WIDTH, CANONICAL_RENDER_HEIGHT)):
            raise ValueError(
                f"Stage-2 {role} image {path} must be RGB "
                f"{CANONICAL_RENDER_WIDTH}x{CANONICAL_RENDER_HEIGHT}; "
                f"got mode={mode}, size={size}."
            )
        if not pose and (len(size) != 2 or min(size) <= 0):
            raise ValueError(f"Stage-2 {role} image {path} has invalid size {size}.")

    for record in manifest.records:
        controls = validate_control_sequence(record.control_dir, record.reference_pose)
        inspect(controls.reference_pose, role="reference-pose", expected_format="JPEG", pose=True)
        for path in controls.frames:
            inspect(path, role="DWPose control", expected_format="JPEG", pose=True)
        inspect(record.reference_image, role="reference")
        for path in target_frame_paths(record):
            inspect(path, role="target", expected_format="PNG")


def required_training_artifacts(config: UniAnimateModelConfig) -> tuple[Path, ...]:
    """Return every external source/model artifact needed before allocation."""

    wan = Path(config.wan_dir)
    return (
        Path(config.unianimate_dir) / "examples" / "unianimate_wan" / "train_unianimate_wan.py",
        Path(config.unianimate_dir) / "diffsynth" / "__init__.py",
        Path(config.lora_path),
        wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        wan / "models_t5_umt5-xxl-enc-bf16.pth",
        wan / "Wan2.1_VAE.pth",
        wan / "google" / "umt5-xxl" / "special_tokens_map.json",
        wan / "google" / "umt5-xxl" / "spiece.model",
        wan / "google" / "umt5-xxl" / "tokenizer.json",
        wan / "google" / "umt5-xxl" / "tokenizer_config.json",
        *(wan / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors" for index in range(1, 8)),
    )


def validate_training_artifacts(config: UniAnimateModelConfig) -> tuple[Path, ...]:
    artifacts = required_training_artifacts(config)
    missing = [path for path in artifacts if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Stage-2 adapter training artifacts are missing:\n{rendered}")
    return artifacts


def required_checkpoint_free_bytes(
    output_dir: str | Path,
    official_adapter: str | Path,
) -> int:
    """Estimate free space needed for best/last plus atomic rolling resume.

    One compact adapter is one unit.  AdamW's two moment tensors make the
    rolling checkpoint approximately three units, and atomic replacement can
    temporarily retain both old and new rolling checkpoints.  The peak is thus
    1 (best) + 1 (last) + 3 (old resume) + 3 (new resume) = 8 units.  Existing
    canonical artifacts are already accounted for by filesystem free space and
    are subtracted, which keeps interrupted runs resumable.
    """

    destination = Path(output_dir).expanduser().resolve()
    adapter = Path(official_adapter).expanduser().resolve()
    if not adapter.is_file() or adapter.stat().st_size <= 0:
        raise FileNotFoundError(f"Official Stage-2 adapter is missing or empty: {adapter}")
    peak_bytes = CHECKPOINT_PEAK_ADAPTER_UNITS * adapter.stat().st_size
    existing_bytes = sum(
        path.stat().st_size
        for path in (destination / "best.ckpt", destination / "last.ckpt", destination / "resume.pt")
        if path.is_file()
    )
    return max(0, peak_bytes - existing_bytes) + CHECKPOINT_HEADROOM_BYTES


def _validate_runtime_report(
    report: AdapterRuntimeReport,
    output_dir: Path,
    config: AdapterTrainingConfig,
) -> None:
    expected_paths = {
        "best": (output_dir / "best.ckpt").resolve(),
        "last": (output_dir / "last.ckpt").resolve(),
        "resume": (output_dir / "resume.pt").resolve(),
    }
    actual_paths = {
        "best": Path(report.best_checkpoint).resolve(),
        "last": Path(report.last_checkpoint).resolve(),
        "resume": Path(report.resume_checkpoint).resolve(),
    }
    if actual_paths != expected_paths:
        raise RuntimeError(
            "Stage-2 runtime must emit only canonical best.ckpt, last.ckpt, and resume.pt paths; "
            f"got {actual_paths}."
        )
    for label, path in actual_paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Stage-2 {label} checkpoint is missing or empty: {path}")

    best_keys = validate_training_adapter_keys(report.best_checkpoint_keys, config)
    last_keys = validate_training_adapter_keys(report.last_checkpoint_keys, config)
    resume_keys = validate_training_adapter_keys(report.resume_checkpoint_keys, config)
    trainable_keys = validate_training_adapter_keys(report.trainable_parameter_names, config)
    if not (set(best_keys) == set(last_keys) == set(resume_keys)):
        raise RuntimeError("Best, last, and resume Stage-2 checkpoints have different parameter-key contracts.")
    if set(best_keys) != set(trainable_keys):
        raise RuntimeError("Stage-2 checkpoints do not exactly cover every and only trainable parameter.")
    expected_lrs = {"dit_lora": config.lora_lr, "dwpose_embedding": config.pose_lr}
    actual_lrs = {name: float(value) for name, value in report.optimizer_group_lrs.items()}
    if actual_lrs != expected_lrs:
        raise RuntimeError(f"Stage-2 optimizer groups must be {expected_lrs}; got {actual_lrs}.")
    if report.steps_completed != config.max_steps:
        raise RuntimeError(
            f"Stage-2 training stopped at {report.steps_completed} steps; expected {config.max_steps}."
        )
    expected_validation_steps = tuple(
        range(config.validation_every_steps, config.max_steps + 1, config.validation_every_steps)
    )
    if (
        report.validation_runs != len(expected_validation_steps)
        or tuple(report.validation_steps) != expected_validation_steps
        or not math.isfinite(report.best_val_loss)
    ):
        raise RuntimeError(
            "Stage-2 validation cadence is incomplete: expected optimizer steps "
            f"{expected_validation_steps}, got {tuple(report.validation_steps)}."
        )


def train_adapter(
    dataset_manifest: str | Path,
    patient_split_manifest: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path | None = None,
    model_config: UniAnimateModelConfig | None = None,
    training_config: AdapterTrainingConfig | None = None,
    seed: int = 42,
    dataloader_num_workers: int = 0,
    resume_checkpoint: str | Path | None = None,
    runtime: AdapterRuntime | None = None,
) -> AdapterTrainingResult:
    """Train the Stage-2 adapter and return its validated best/last artifacts.

    The default runtime is the checked-out UniAnimate one-stage Wan trainer.
    ``runtime`` is an injection seam for unit tests and alternative launchers;
    its report is checked against the same allowlist and checkpoint contract.
    """

    config = training_config or AdapterTrainingConfig()
    models = model_config or UniAnimateModelConfig()
    validate_training_device(str(models.device))
    if dataloader_num_workers != 0:
        raise ValueError("Exact Stage-2 sampler resume requires dataloader_num_workers=0.")
    if seed < 0:
        raise ValueError("seed cannot be negative")

    manifest = load_adapter_dataset_manifest(
        dataset_manifest,
        patient_split_manifest,
        root=root,
        validate_files=True,
    )
    destination = Path(output_dir).expanduser().resolve()
    resume_path = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint else None
    existing = [destination / "best.ckpt", destination / "last.ckpt", destination / "resume.pt"]
    present = [path for path in existing if path.exists()]
    if resume_path is None and present:
        raise FileExistsError(
            "Refusing to overwrite an existing Stage-2 adapter run: "
            + ", ".join(str(path) for path in present)
        )
    if resume_path is not None:
        expected_resume = (destination / "resume.pt").resolve()
        if resume_path != expected_resume:
            raise ValueError(
                f"Exact resume must continue in the same output directory from {expected_resume}; got {resume_path}."
            )
        if not resume_path.is_file():
            raise FileNotFoundError(f"Stage-2 rolling resume checkpoint does not exist: {resume_path}")
    destination.mkdir(parents=True, exist_ok=True)

    runtime_fn = runtime
    source_sha256 = "runtime-injected"
    model_sha256 = "runtime-injected"
    environment_sha256 = "runtime-injected"
    if runtime_fn is None:
        from ..io import require_free_disk

        require_free_disk(
            destination,
            100.0,
            operation="production Stage-2 adapter training",
        )
        validate_training_artifacts(models)
        validate_official_adapter_checkpoint(models.lora_path)
        validate_training_images(manifest)
        free_bytes = shutil.disk_usage(destination).free
        required_free = required_checkpoint_free_bytes(destination, models.lora_path)
        if free_bytes < required_free:
            raise RuntimeError(
                "Stage-2 best/last/rolling-resume checkpointing needs "
                f"{required_free / 1024**3:.1f} GiB additional free space for this run state; "
                f"found {free_bytes / 1024**3:.1f} GiB at {destination}."
            )
        project_root = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
        fingerprints = compute_training_fingerprints(manifest, project_root, models)
        source_sha256 = fingerprints.training_source_sha256
        model_sha256 = fingerprints.model_artifacts_sha256
        environment_sha256 = fingerprints.runtime_environment_sha256
        if resume_path is not None:
            payload = read_training_resume(
                resume_path,
                config=config,
                dataset_manifest_sha256=manifest.manifest_sha256,
                patient_split_sha256=manifest.split_manifest_sha256,
                dataset_content_sha256=manifest.dataset_content_sha256,
                training_source_sha256=source_sha256,
                model_artifacts_sha256=model_sha256,
                runtime_environment_sha256=environment_sha256,
                seed=int(seed),
            )
            del payload
        runtime_fn = _run_unianimate_training

    report = runtime_fn(
        manifest=manifest,
        output_dir=destination,
        model_config=models,
        training_config=config,
        seed=int(seed),
        dataloader_num_workers=int(dataloader_num_workers),
        resume_checkpoint=resume_path,
        training_source_sha256=source_sha256,
        model_artifacts_sha256=model_sha256,
        runtime_environment_sha256=environment_sha256,
    )
    if not isinstance(report, AdapterRuntimeReport):
        raise TypeError("Stage-2 runtime must return AdapterRuntimeReport.")
    _validate_runtime_report(report, destination, config)
    return AdapterTrainingResult(
        output_dir=destination,
        best_checkpoint=Path(report.best_checkpoint).resolve(),
        last_checkpoint=Path(report.last_checkpoint).resolve(),
        resume_checkpoint=Path(report.resume_checkpoint).resolve(),
        train_examples=len(manifest.train),
        val_examples=len(manifest.val),
        steps_completed=report.steps_completed,
        validation_runs=report.validation_runs,
        validation_steps=tuple(report.validation_steps),
        best_val_loss=report.best_val_loss,
        dataset_manifest_sha256=manifest.manifest_sha256,
        patient_split_sha256=manifest.split_manifest_sha256,
        dataset_content_sha256=manifest.dataset_content_sha256,
        training_source_sha256=source_sha256,
        model_artifacts_sha256=model_sha256,
        runtime_environment_sha256=environment_sha256,
        seed=int(seed),
    )


def _load_upstream_training_module(unianimate_dir: Path) -> Any:
    source = unianimate_dir / "examples" / "unianimate_wan" / "train_unianimate_wan.py"
    module_name = "_dmtbaby_unianimate_adapter_training"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    root_value = str(unianimate_dir)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import UniAnimate training API from {source}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(
            "UniAnimate training dependencies failed to import. Use the pinned dmtbaby environment; "
            f"source={source}; error={exc!r}"
        ) from exc
    sys.modules[module_name] = module
    return module


def _frame_dataset_class(torch: Any, np: Any, Image: Any) -> type:
    class AdapterFrameDataset(torch.utils.data.Dataset):
        def __init__(self, records: tuple[AdapterDatasetRecord, ...]) -> None:
            self.records = records
            self._resampling = getattr(Image, "Resampling", Image).BILINEAR

        def __len__(self) -> int:
            return len(self.records)

        def _rgb_array(self, path: Path) -> Any:
            try:
                with Image.open(path) as image:
                    resized = image.convert("RGB").resize((TRAIN_WIDTH, TRAIN_HEIGHT), self._resampling)
                    return np.asarray(resized, dtype=np.uint8).copy()
            except Exception as exc:
                raise RuntimeError(f"Could not decode Stage-2 training image {path}: {exc}") from exc

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            controls = validate_control_sequence(record.control_dir, record.reference_pose)
            target_arrays = [self._rgb_array(path) for path in target_frame_paths(record)]
            control_arrays = [self._rgb_array(path) for path in controls.frames]
            target_video = torch.stack(
                [torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1.0) for array in target_arrays],
                dim=1,
            )
            dwpose_data = torch.stack(
                [torch.from_numpy(array).permute(2, 0, 1) for array in control_arrays],
                dim=1,
            )
            return {
                "text": record.prompt,
                "video": target_video,
                "path": record.clip_id,
                "first_frame": torch.from_numpy(self._rgb_array(record.reference_image)),
                "dwpose_data": dwpose_data,
                "random_ref_dwpose_data": torch.from_numpy(self._rgb_array(record.reference_pose)),
            }

    return AdapterFrameDataset


def _stateful_random_sampler_class(torch: Any) -> type:
    class StatefulRandomSampler(torch.utils.data.Sampler):
        """One virtual epoch with resumable, repeatedly shuffled permutations.

        Keeping all 4,000 microbatches in one virtual epoch prevents Lightning
        from flushing a partial accumulation window at physical dataset epoch
        boundaries.  Validation therefore lands exactly at optimizer steps
        250, 500, 750, and 1,000.
        """

        def __init__(self, data_source: Any, seed: int, total_samples: int) -> None:
            if len(data_source) <= 0:
                raise ValueError("Stage-2 training dataset cannot be empty.")
            if total_samples <= 0:
                raise ValueError("Stage-2 sampler total_samples must be positive.")
            self.data_source = data_source
            self.generator = torch.Generator().manual_seed(int(seed))
            self.total_samples = int(total_samples)
            self.samples_yielded = 0
            self.permutation: Any | None = None
            self.position = 0
            self.epoch = 0

        def __len__(self) -> int:
            return self.total_samples

        def __iter__(self):
            length = len(self.data_source)
            while self.samples_yielded < self.total_samples:
                if self.permutation is None or self.position >= length:
                    self.permutation = torch.randperm(length, generator=self.generator)
                    self.position = 0
                    self.epoch += 1
                index = int(self.permutation[self.position])
                self.position += 1
                self.samples_yielded += 1
                yield index

        def state_dict(self) -> dict[str, Any]:
            permutation = self.permutation
            if permutation is None:
                permutation = torch.empty(0, dtype=torch.int64)
            return {
                "generator_state": self.generator.get_state(),
                "permutation": permutation.clone(),
                "position": int(self.position),
                "epoch": int(self.epoch),
                "samples_yielded": int(self.samples_yielded),
                "total_samples": int(self.total_samples),
            }

        def load_state_dict(self, state: Mapping[str, Any]) -> None:
            required = {
                "generator_state",
                "permutation",
                "position",
                "epoch",
                "samples_yielded",
                "total_samples",
            }
            if not required.issubset(state):
                raise ValueError("Stage-2 resume sampler state is incomplete.")
            if int(state["total_samples"]) != self.total_samples:
                raise ValueError("Stage-2 resume sampler length does not match the locked training run.")
            permutation = state["permutation"]
            if not hasattr(permutation, "numel"):
                raise ValueError("Stage-2 resume sampler permutation is malformed.")
            if int(permutation.numel()) not in {0, len(self.data_source)}:
                raise ValueError("Stage-2 resume sampler permutation length does not match the training dataset.")
            position = int(state["position"])
            if not 0 <= position <= len(self.data_source):
                raise ValueError("Stage-2 resume sampler position is outside the training dataset.")
            samples_yielded = int(state["samples_yielded"])
            if not 0 <= samples_yielded <= self.total_samples:
                raise ValueError("Stage-2 resume sampler progress is outside the locked training run.")
            self.generator.set_state(state["generator_state"])
            self.permutation = permutation.clone() if int(permutation.numel()) else None
            self.position = position
            self.epoch = int(state["epoch"])
            self.samples_yielded = samples_yielded

    return StatefulRandomSampler


def _adapter_lightning_module_class(upstream: Any, torch: Any, Image: Any, config: AdapterTrainingConfig) -> type:
    class DMTBabyAdapterModule(upstream.LightningModelForTrain_onestage):
        """Upstream UniAnimate objective with a strict optimizer and validation path."""

        def _diffusion_loss(self, batch: Mapping[str, Any], *, conditioning_dropout: bool) -> Any:
            text = batch["text"][0]
            video = batch["video"]
            self.pipe_VAE.device = self.device
            dwpose_data = self.dwpose_embedding(
                (
                    torch.cat(
                        [batch["dwpose_data"][:, :, :1].repeat(1, 1, 3, 1, 1), batch["dwpose_data"]],
                        dim=2,
                    )
                    / 255.0
                ).to(self.device)
            )
            random_ref_dwpose_data = self.randomref_embedding_pose(
                (batch["random_ref_dwpose_data"] / 255.0)
                .to(torch.bfloat16)
                .to(self.device)
                .permute(0, 3, 1, 2)
            ).unsqueeze(2)

            with torch.no_grad():
                prompt_emb = self.pipe_VAE.encode_prompt(text)
                video = video.to(dtype=self.pipe_VAE.torch_dtype, device=self.pipe_VAE.device)
                latents = self.pipe_VAE.encode_video(video, **self.tiler_kwargs)[0]
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe_VAE.encode_image(first_frame, num_frames, height, width)

            drop_all = conditioning_dropout and random.random() < 0.05
            drop_image = conditioning_dropout and random.random() < 0.1
            if drop_all:
                dwpose_data = torch.zeros_like(dwpose_data)
                random_ref_dwpose_data = torch.zeros_like(random_ref_dwpose_data)

            latents = latents.unsqueeze(0).to(self.device)
            prompt_emb["context"] = prompt_emb["context"].to(self.device)
            if "clip_feature" in image_emb:
                image_emb["clip_feature"] = image_emb["clip_feature"].to(self.device)
                if drop_image:
                    image_emb["clip_feature"] = torch.zeros_like(image_emb["clip_feature"])
            if "y" in image_emb:
                if drop_image:
                    image_emb["y"] = torch.zeros_like(image_emb["y"])
                image_emb["y"] = image_emb["y"].to(self.device) + random_ref_dwpose_data

            condition = upstream.rearrange(dwpose_data, "b c f h w -> b (f h w) c").contiguous()
            self.pipe.device = self.device
            noise = torch.randn_like(latents)
            timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
            timestep = self.pipe.scheduler.timesteps[timestep_id].to(
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )
            extra_input = self.pipe.prepare_extra_input(latents)
            noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
            target = self.pipe.scheduler.training_target(latents, noise, timestep)
            prediction = self.pipe.denoising_model()(
                noisy_latents,
                timestep=timestep,
                **prompt_emb,
                **extra_input,
                **image_emb,
                use_gradient_checkpointing=config.gradient_checkpointing,
                use_gradient_checkpointing_offload=config.gradient_checkpointing_offload,
                add_condition=condition,
            )
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
            return loss * self.pipe.scheduler.training_weight(timestep)

        def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Any:
            del batch_idx
            loss = self._diffusion_loss(batch, conditioning_dropout=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
            return loss

        def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Any:
            devices: list[int] = []
            if self.device.type == "cuda":
                devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(10_000 + int(batch_idx))
                loss = self._diffusion_loss(batch, conditioning_dropout=False)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
            return loss.detach()

        def configure_optimizers(self) -> dict[str, Any]:
            configure_trainable_parameters(self, config)
            optimizer = create_adamw(self, config)
            scheduler = create_cosine_scheduler(optimizer, config)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

        def on_train_start(self) -> None:
            assert_trainable_allowlist(self, config=config)

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            # Lightning loop/optimizer state is retained only for rolling resume;
            # model state itself is always compact and allowlisted.
            manifest = self._dmtbaby_manifest
            sampler = self._dmtbaby_train_sampler
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            scaler_state = scaler.state_dict() if scaler is not None else None
            checkpoint["format"] = RESUME_FORMAT
            checkpoint["version"] = RESUME_VERSION
            checkpoint["state_dict"] = adapter_checkpoint_state_dict(self, config)
            checkpoint["dmtbaby_resume"] = {
                "training_config": asdict(config),
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "patient_split_sha256": manifest.split_manifest_sha256,
                "dataset_content_sha256": manifest.dataset_content_sha256,
                "training_source_sha256": self._dmtbaby_training_source_sha256,
                "model_artifacts_sha256": self._dmtbaby_model_artifacts_sha256,
                "runtime_environment_sha256": self._dmtbaby_runtime_environment_sha256,
                "seed": int(self._dmtbaby_seed),
                "step": int(self.trainer.global_step),
                "reference_pose_encoder_frozen": True,
                "rng_state": capture_training_rng_state(),
                "sampler_state": sampler.state_dict(),
                "grad_scaler": scaler_state,
            }

        def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            manifest = self._dmtbaby_manifest
            validate_training_resume_payload(
                checkpoint,
                config=config,
                dataset_manifest_sha256=manifest.manifest_sha256,
                patient_split_sha256=manifest.split_manifest_sha256,
                dataset_content_sha256=manifest.dataset_content_sha256,
                training_source_sha256=self._dmtbaby_training_source_sha256,
                model_artifacts_sha256=self._dmtbaby_model_artifacts_sha256,
                runtime_environment_sha256=self._dmtbaby_runtime_environment_sha256,
                seed=int(self._dmtbaby_seed),
            )
            metadata = checkpoint["dmtbaby_resume"]
            self._dmtbaby_train_sampler.load_state_dict(metadata["sampler_state"])
            restore_training_rng_state(metadata["rng_state"])

    return DMTBabyAdapterModule


def _compact_checkpoint_callback_class(
    pl: Any,
    config: AdapterTrainingConfig,
    output_dir: Path,
) -> type:
    class CompactBestLast(pl.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best_val_loss = math.inf
            self.last_val_loss = math.inf
            self.validation_runs = 0
            self.validation_steps: list[int] = []
            self.best_path = output_dir / "best.ckpt"
            self.last_path = output_dir / "last.ckpt"

        @property
        def state_key(self) -> str:
            return "dmtbaby.stage2.CompactBestLast"

        def state_dict(self) -> dict[str, Any]:
            return {
                "best_val_loss": float(self.best_val_loss),
                "last_val_loss": float(self.last_val_loss),
                "validation_runs": int(self.validation_runs),
                "validation_steps": tuple(self.validation_steps),
            }

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            self.best_val_loss = float(state_dict.get("best_val_loss", math.inf))
            self.last_val_loss = float(state_dict.get("last_val_loss", math.inf))
            self.validation_runs = int(state_dict.get("validation_runs", 0))
            self.validation_steps = [int(step) for step in state_dict.get("validation_steps", ())]

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get("val_loss")
            if metric is None:
                raise RuntimeError("UniAnimate validation completed without val_loss.")
            value = float(metric.detach().float().cpu() if hasattr(metric, "detach") else metric)
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite Stage-2 validation loss: {value}")
            self.validation_runs += 1
            self.validation_steps.append(int(trainer.global_step))
            self.last_val_loss = value
            save_training_adapter(self.last_path, pl_module, config)
            if value < self.best_val_loss:
                self.best_val_loss = value
                save_training_adapter(self.best_path, pl_module, config)

        def on_train_end(self, trainer: Any, pl_module: Any) -> None:
            if self.validation_runs < 1 or not self.best_path.is_file():
                raise RuntimeError("No validation-selected best Stage-2 adapter was produced.")
            save_training_adapter(self.last_path, pl_module, config)

    return CompactBestLast


def _rolling_resume_callback_class(pl: Any, config: AdapterTrainingConfig, output_dir: Path) -> type:
    class RollingResume(pl.Callback):
        """Atomically overwrite exactly one full-state resume checkpoint."""

        def __init__(self) -> None:
            super().__init__()
            self.path = output_dir / "resume.pt"
            self.last_saved_step = 0

        @property
        def state_key(self) -> str:
            return "dmtbaby.stage2.RollingResume"

        def state_dict(self) -> dict[str, int]:
            return {"last_saved_step": int(self.last_saved_step)}

        def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
            self.last_saved_step = int(state_dict.get("last_saved_step", 0))

        def _save(self, trainer: Any) -> None:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".resume.pt.",
                suffix=".tmp",
                dir=output_dir,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary.unlink(missing_ok=True)
            try:
                trainer.save_checkpoint(str(temporary), weights_only=False)
                if not temporary.is_file() or temporary.stat().st_size <= 0:
                    raise RuntimeError("Lightning did not produce a non-empty Stage-2 resume checkpoint.")
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

        def on_train_batch_end(
            self,
            trainer: Any,
            pl_module: Any,
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ) -> None:
            del pl_module, outputs, batch, batch_idx
            step = int(trainer.global_step)
            if (
                step > 0
                and step < config.max_steps
                and step != self.last_saved_step
                and step % config.resume_every_steps == 0
                and step % config.validation_every_steps != 0
            ):
                self.last_saved_step = step
                self._save(trainer)

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            if trainer.sanity_checking:
                return
            step = int(trainer.global_step)
            if step > 0 and step % config.resume_every_steps == 0:
                self.last_saved_step = step
                self._save(trainer)

        def on_train_end(self, trainer: Any, pl_module: Any) -> None:
            del pl_module
            step = int(trainer.global_step)
            if step != self.last_saved_step:
                self.last_saved_step = step
                self._save(trainer)

    return RollingResume


def _run_unianimate_training(
    *,
    manifest: AdapterDatasetManifest,
    output_dir: Path,
    model_config: UniAnimateModelConfig,
    training_config: AdapterTrainingConfig,
    seed: int,
    dataloader_num_workers: int,
    resume_checkpoint: Path | None,
    training_source_sha256: str,
    model_artifacts_sha256: str,
    runtime_environment_sha256: str,
) -> AdapterRuntimeReport:
    """Run the real checked-out UniAnimate one-stage training implementation."""

    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import lightning as pl  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Stage-2 adapter training requires torch, NumPy, Pillow, and Lightning from the pinned environment. "
            f"Import error: {exc!r}"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 adapter training requires CUDA, but torch.cuda.is_available() is False.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage-2 adapter training requires a CUDA device with BF16 support.")
    device_index = validate_training_device(str(model_config.device))
    if device_index is not None and device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested Stage-2 CUDA device {device_index}, but only {torch.cuda.device_count()} devices are visible."
        )

    torch.backends.cuda.matmul.allow_tf32 = training_config.allow_tf32
    torch.backends.cudnn.allow_tf32 = training_config.allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(training_config.deterministic)
    pl.seed_everything(seed, workers=True)
    upstream = _load_upstream_training_module(Path(model_config.unianimate_dir))
    Dataset = _frame_dataset_class(torch, np, Image)
    train_dataset = Dataset(manifest.train)
    val_dataset = Dataset(manifest.val)
    Sampler = _stateful_random_sampler_class(torch)
    total_microbatches = training_config.max_steps * training_config.gradient_accumulation_steps
    train_sampler = Sampler(train_dataset, seed, total_samples=total_microbatches)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=False,
        persistent_workers=dataloader_num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=False,
        persistent_workers=dataloader_num_workers > 0,
    )

    wan = Path(model_config.wan_dir)
    model_vae = upstream.LightningModelForDataProcess(
        text_encoder_path=str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
        image_encoder_path=str(wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        vae_path=str(wan / "Wan2.1_VAE.pth"),
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    )
    Model = _adapter_lightning_module_class(upstream, torch, Image, training_config)
    dit_paths = ",".join(
        str(wan / f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors")
        for index in range(1, 8)
    )
    model = Model(
        dit_path=dit_paths,
        learning_rate=training_config.lora_lr,
        train_architecture="lora",
        lora_rank=training_config.lora_rank,
        lora_alpha=training_config.lora_alpha,
        lora_target_modules=",".join(training_config.target_modules),
        init_lora_weights="kaiming",
        use_gradient_checkpointing=training_config.gradient_checkpointing,
        use_gradient_checkpointing_offload=training_config.gradient_checkpointing_offload,
        pretrained_lora_path=None,
        model_VAE=model_vae,
    )
    model.strict_loading = False
    model._dmtbaby_manifest = manifest
    model._dmtbaby_train_sampler = train_sampler
    model._dmtbaby_training_source_sha256 = training_source_sha256
    model._dmtbaby_model_artifacts_sha256 = model_artifacts_sha256
    model._dmtbaby_runtime_environment_sha256 = runtime_environment_sha256
    model._dmtbaby_seed = int(seed)
    incompatible = load_official_adapter(model, Path(model_config.lora_path), training_config)
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise RuntimeError(
            "Official UniAnimate adapter has keys not accepted by the constructed rank-128 model: "
            + ", ".join(unexpected[:12])
        )
    summary = assert_trainable_allowlist(model, config=training_config)
    groups = optimizer_parameter_groups(model, training_config)

    Callback = _compact_checkpoint_callback_class(pl, training_config, output_dir)
    checkpoint_callback = Callback()
    RollingCallback = _rolling_resume_callback_class(pl, training_config, output_dir)
    rolling_callback = RollingCallback()
    validation_interval_batches = (
        training_config.gradient_accumulation_steps * training_config.validation_every_steps
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1 if device_index is None else [device_index],
        strategy="auto",
        precision="bf16-mixed",
        max_steps=training_config.max_steps,
        accumulate_grad_batches=training_config.gradient_accumulation_steps,
        check_val_every_n_epoch=None,
        val_check_interval=validation_interval_batches,
        num_sanity_val_steps=1,
        callbacks=[checkpoint_callback, rolling_callback],
        enable_checkpointing=False,
        default_root_dir=str(output_dir),
        logger=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        deterministic=training_config.deterministic,
        benchmark=False,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )

    best_state = torch.load(
        checkpoint_callback.best_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    last_state = torch.load(
        checkpoint_callback.last_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(best_state, Mapping) or not isinstance(last_state, Mapping):
        raise RuntimeError("Stage-2 compact checkpoints must be raw mapping-valued state dicts.")
    best_keys = validate_training_adapter_keys(best_state, training_config)
    last_keys = validate_training_adapter_keys(last_state, training_config)
    del best_state, last_state
    resume_payload = read_training_resume(
        rolling_callback.path,
        config=training_config,
        dataset_manifest_sha256=manifest.manifest_sha256,
        patient_split_sha256=manifest.split_manifest_sha256,
        dataset_content_sha256=manifest.dataset_content_sha256,
        training_source_sha256=training_source_sha256,
        model_artifacts_sha256=model_artifacts_sha256,
        runtime_environment_sha256=runtime_environment_sha256,
        seed=int(seed),
    )
    resume_keys = tuple(resume_payload["state_dict"])
    del resume_payload
    return AdapterRuntimeReport(
        best_checkpoint=checkpoint_callback.best_path,
        last_checkpoint=checkpoint_callback.last_path,
        resume_checkpoint=rolling_callback.path,
        best_checkpoint_keys=best_keys,
        last_checkpoint_keys=last_keys,
        resume_checkpoint_keys=resume_keys,
        trainable_parameter_names=summary.all_names,
        optimizer_group_lrs={str(group["name"]): float(group["lr"]) for group in groups},
        steps_completed=int(trainer.global_step),
        validation_runs=int(checkpoint_callback.validation_runs),
        validation_steps=tuple(checkpoint_callback.validation_steps),
        best_val_loss=float(checkpoint_callback.best_val_loss),
    )
