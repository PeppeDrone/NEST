"""Root-CLI adapter for executable Stage-2 training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapter_dataset import load_adapter_dataset_manifest
from .adapter_runner import compute_training_fingerprints, train_adapter as _train_adapter
from .adapter_runner import validate_training_artifacts, validate_training_device, validate_training_images
from .adapter_training import AdapterTrainingConfig
from .adapter_training import read_training_resume, validate_official_adapter_checkpoint
from .service import UniAnimateModelConfig


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class AdapterTrainRequest:
    root: Path
    manifest: Path
    patient_split_manifest: Path
    output_dir: Path
    official_adapter: Path
    unianimate_dir: Path
    wan_dir: Path
    device: str = "cuda"
    seed: int = 42
    resume: Path | None = None
    dry_run: bool = False

    @classmethod
    def from_namespace(cls, args: Any, root: str | Path) -> "AdapterTrainRequest":
        project_root = Path(root).expanduser().resolve()
        resume_value = getattr(args, "resume", None)
        return cls(
            root=project_root,
            manifest=_resolve(project_root, args.manifest),
            patient_split_manifest=_resolve(
                project_root,
                getattr(args, "patient_split_manifest", "reproducibility/patient_split_v1.json"),
            ),
            output_dir=_resolve(project_root, args.output_dir),
            official_adapter=_resolve(project_root, args.official_adapter),
            unianimate_dir=_resolve(
                project_root,
                getattr(args, "unianimate_dir", "external/UniAnimate-DiT"),
            ),
            wan_dir=_resolve(
                project_root,
                getattr(args, "wan_dir", "external/UniAnimate-DiT/Wan2.1-I2V-14B-720P"),
            ),
            device=str(getattr(args, "device", "cuda")),
            seed=int(getattr(args, "seed", 42)),
            resume=_resolve(project_root, resume_value) if resume_value else None,
            dry_run=bool(getattr(args, "dry_run", False)),
        )


def train_adapter(request: AdapterTrainRequest) -> int:
    """Validate or execute Stage-2 training for the root CLI."""

    if not isinstance(request, AdapterTrainRequest):
        raise TypeError("train_adapter expects AdapterTrainRequest")
    config = AdapterTrainingConfig()
    models = UniAnimateModelConfig(
        unianimate_dir=request.unianimate_dir,
        wan_dir=request.wan_dir,
        lora_path=request.official_adapter,
        official_lora_path=request.official_adapter,
        device=request.device,
    )
    validate_training_device(request.device)
    if request.dry_run:
        manifest = load_adapter_dataset_manifest(
            request.manifest,
            request.patient_split_manifest,
            root=request.root,
            validate_files=True,
        )
        validate_training_artifacts(models)
        validate_official_adapter_checkpoint(models.lora_path)
        validate_training_images(manifest)
        fingerprints = compute_training_fingerprints(manifest, request.root, models)
        if request.resume is not None:
            payload = read_training_resume(
                request.resume,
                config=config,
                dataset_manifest_sha256=manifest.manifest_sha256,
                patient_split_sha256=manifest.split_manifest_sha256,
                dataset_content_sha256=manifest.dataset_content_sha256,
                training_source_sha256=fingerprints.training_source_sha256,
                model_artifacts_sha256=fingerprints.model_artifacts_sha256,
                runtime_environment_sha256=fingerprints.runtime_environment_sha256,
                seed=request.seed,
            )
            del payload
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "ready": True,
                    "control_variant": "manual-whole62-constrained",
                    "train_examples": len(manifest.train),
                    "val_examples": len(manifest.val),
                    "output_dir": str(request.output_dir),
                    "official_adapter": str(request.official_adapter),
                    "resume": str(request.resume) if request.resume else None,
                    "training_config": asdict(config),
                    "fingerprints": asdict(fingerprints),
                },
                sort_keys=True,
            )
        )
        return 0

    result = _train_adapter(
        request.manifest,
        request.patient_split_manifest,
        request.output_dir,
        root=request.root,
        model_config=models,
        training_config=config,
        seed=request.seed,
        dataloader_num_workers=0,
        resume_checkpoint=request.resume,
    )
    print(
        json.dumps(
            {
                "ready": True,
                "best_checkpoint": str(result.best_checkpoint),
                "last_checkpoint": str(result.last_checkpoint),
                "resume_checkpoint": str(result.resume_checkpoint),
                "steps_completed": result.steps_completed,
                "validation_runs": result.validation_runs,
                "validation_steps": result.validation_steps,
                "best_val_loss": result.best_val_loss,
                "train_examples": result.train_examples,
                "val_examples": result.val_examples,
                "dataset_manifest_sha256": result.dataset_manifest_sha256,
                "patient_split_sha256": result.patient_split_sha256,
                "dataset_content_sha256": result.dataset_content_sha256,
                "training_source_sha256": result.training_source_sha256,
                "model_artifacts_sha256": result.model_artifacts_sha256,
                "runtime_environment_sha256": result.runtime_environment_sha256,
                "seed": result.seed,
            },
            sort_keys=True,
        )
    )
    return 0
