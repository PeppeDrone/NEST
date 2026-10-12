"""Fail-closed manifest contract for Stage-2 adapter fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .controls import CASCADE_FRAME_COUNT, ControlSequence, validate_control_sequence


ADAPTER_DATASET_SCHEMA = "dmtbaby.stage2.adapter_dataset"
ADAPTER_DATASET_SCHEMA_VERSION = 1
CONTROL_VARIANT = "manual-whole62-constrained"
PROMPT_SOURCE = "manual"
JOINT_COUNT = 62
POSTPROCESSING = "constrained"
CONTROL_SOURCE = "stage1_prediction"
TARGET_FRAME_NAMES = tuple(f"{index:04d}.png" for index in range(CASCADE_FRAME_COUNT))


@dataclass(frozen=True)
class AdapterDatasetRecord:
    """One aligned target/control example authorized for adapter training."""

    clip_id: str
    split: str
    prompt: str
    control_dir: Path
    reference_pose: Path
    reference_image: Path
    target_frames_dir: Path
    control_variant: str = CONTROL_VARIANT
    prompt_source: str = PROMPT_SOURCE
    joint_count: int = JOINT_COUNT
    postprocessing: str = POSTPROCESSING
    control_source: str = CONTROL_SOURCE

    @property
    def patient_id(self) -> str:
        if "_clip" not in self.clip_id:
            raise ValueError(f"Invalid clip_id (missing '_clip'): {self.clip_id}")
        return self.clip_id.split("_clip", 1)[0]


@dataclass(frozen=True)
class AdapterDatasetManifest:
    path: Path
    split_manifest_path: Path
    manifest_sha256: str
    split_manifest_sha256: str
    dataset_content_sha256: str
    train: tuple[AdapterDatasetRecord, ...]
    val: tuple[AdapterDatasetRecord, ...]

    @property
    def records(self) -> tuple[AdapterDatasetRecord, ...]:
        return self.train + self.val


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_content_sha256(records: list[AdapterDatasetRecord]) -> str:
    """Hash every referenced byte in a path-independent, deterministic order."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.clip_id):
        controls = validate_control_sequence(record.control_dir, record.reference_pose)
        artifacts = (
            ("reference_pose", controls.reference_pose),
            ("reference_image", record.reference_image),
            *((f"control/{path.name}", path) for path in controls.frames),
            *((f"target/{path.name}", path) for path in target_frame_paths(record)),
        )
        for role, artifact in artifacts:
            label = f"{record.clip_id}\0{role}\0".encode("utf-8")
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            digest.update(artifact.stat().st_size.to_bytes(8, "big"))
            with artifact.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _read_records(path: Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                rows.append(row)
        return {}, rows

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid adapter dataset JSON {path}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Adapter dataset manifest must be a JSON object or JSONL file.")
    records = payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, Mapping) for row in records):
        raise ValueError("Adapter dataset JSON must contain a list-valued 'records' field.")
    return dict(payload), list(records)


def _resolve_path(value: Any, root: Path, field: str, clip_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {clip_id!r} has no non-empty {field!r} path.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_target_frames(directory: Path, clip_id: str) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Target frame directory for {clip_id} does not exist: {directory}")
    expected = set(TARGET_FRAME_NAMES)
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or len(actual) != CASCADE_FRAME_COUNT:
        details = [
            f"Target frames for {clip_id} must be exactly {CASCADE_FRAME_COUNT} PNGs "
            "named 0000.png through 0080.png."
        ]
        if missing:
            details.append(f"Missing: {', '.join(missing[:6])}.")
        if unexpected:
            details.append(f"Unexpected: {', '.join(unexpected[:6])}.")
        raise ValueError(" ".join(details))
    return tuple(directory / name for name in TARGET_FRAME_NAMES)


def _required_string(row: Mapping[str, Any], field: str, clip_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record {clip_id!r} requires non-empty string field {field!r}.")
    return value.strip()


def _validate_provenance(row: Mapping[str, Any], clip_id: str) -> None:
    expected: dict[str, Any] = {
        "control_variant": CONTROL_VARIANT,
        "prompt_source": PROMPT_SOURCE,
        "joint_count": JOINT_COUNT,
        "postprocessing": POSTPROCESSING,
        "control_source": CONTROL_SOURCE,
    }
    mismatches = [
        f"{field}={row.get(field)!r} (expected {value!r})"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"Record {clip_id!r} is not the locked {CONTROL_VARIANT} condition: "
            + "; ".join(mismatches)
        )


def _split_sets(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        values = payload.get(split)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Split manifest field {split!r} must be a list of clip IDs.")
        if len(values) != len(set(values)):
            raise ValueError(f"Split manifest {split!r} contains duplicate clip IDs.")
        sets[split] = set(values)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"Split manifest leaks clips across {left}/{right}: {sorted(overlap)[:6]}")
    patient_sets = {
        split: {clip_id.split("_clip", 1)[0] for clip_id in values}
        for split, values in sets.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(f"Split manifest leaks patients across {left}/{right}: {sorted(overlap)}")
    return sets


def load_adapter_dataset_manifest(
    manifest_path: str | Path,
    split_manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    validate_files: bool = True,
) -> AdapterDatasetManifest:
    """Load and validate manual/whole-62/constrained train and validation data.

    Training records must belong to the frozen training-patient partition and
    validation records to its validation-patient partition. Test clips are
    rejected outright. Every temporal target and control is exactly 81 frames.
    """

    path = Path(manifest_path).expanduser().resolve()
    split_path = Path(split_manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Adapter dataset manifest does not exist: {path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Patient split manifest does not exist: {split_path}")
    base = Path(root).expanduser().resolve() if root is not None else path.parent

    header, raw_records = _read_records(path)
    if header:
        if header.get("schema") != ADAPTER_DATASET_SCHEMA:
            raise ValueError(
                f"Adapter dataset schema must be {ADAPTER_DATASET_SCHEMA!r}; got {header.get('schema')!r}."
            )
        if header.get("schema_version") != ADAPTER_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Adapter dataset schema_version must be {ADAPTER_DATASET_SCHEMA_VERSION}."
            )

    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid patient split JSON {split_path}: {exc.msg}") from exc
    if not isinstance(split_payload, Mapping):
        raise ValueError("Patient split manifest must be a JSON object.")
    authorized = _split_sets(split_payload)

    allowed_fields = {
        "clip_id",
        "split",
        "prompt",
        "control_dir",
        "reference_pose",
        "reference_image",
        "target_frames_dir",
        "control_variant",
        "prompt_source",
        "joint_count",
        "postprocessing",
        "control_source",
    }
    records: list[AdapterDatasetRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_records):
        unknown = sorted(set(row) - allowed_fields)
        if unknown:
            raise ValueError(f"Adapter record {index} has unknown fields: {', '.join(unknown)}")
        clip_id = _required_string(row, "clip_id", f"row-{index}")
        if clip_id in seen:
            raise ValueError(f"Duplicate adapter record for clip_id {clip_id!r}.")
        seen.add(clip_id)
        split = _required_string(row, "split", clip_id)
        if split not in {"train", "val"}:
            raise ValueError(f"Record {clip_id!r} split must be 'train' or 'val'; got {split!r}.")
        if clip_id not in authorized[split]:
            actual = next((name for name, values in authorized.items() if clip_id in values), "absent")
            raise ValueError(
                f"Record {clip_id!r} claims split {split!r}, but frozen patient split says {actual!r}."
            )
        _validate_provenance(row, clip_id)

        control_dir = _resolve_path(row.get("control_dir"), base, "control_dir", clip_id)
        reference_pose = _resolve_path(row.get("reference_pose"), base, "reference_pose", clip_id)
        reference_image = _resolve_path(row.get("reference_image"), base, "reference_image", clip_id)
        target_frames_dir = _resolve_path(row.get("target_frames_dir"), base, "target_frames_dir", clip_id)
        prompt = _required_string(row, "prompt", clip_id)
        if validate_files:
            controls: ControlSequence = validate_control_sequence(control_dir, reference_pose)
            reference_pose = controls.reference_pose
            if not reference_image.is_file():
                raise FileNotFoundError(f"Reference image for {clip_id} does not exist: {reference_image}")
            _validate_target_frames(target_frames_dir, clip_id)
        records.append(
            AdapterDatasetRecord(
                clip_id=clip_id,
                split=split,
                prompt=prompt,
                control_dir=control_dir,
                reference_pose=reference_pose,
                reference_image=reference_image,
                target_frames_dir=target_frames_dir,
                control_variant=CONTROL_VARIANT,
                prompt_source=PROMPT_SOURCE,
                joint_count=JOINT_COUNT,
                postprocessing=POSTPROCESSING,
                control_source=CONTROL_SOURCE,
            )
        )

    train = tuple(record for record in records if record.split == "train")
    val = tuple(record for record in records if record.split == "val")
    if not train:
        raise ValueError("Adapter dataset has no authorized training-patient records.")
    if not val:
        raise ValueError("Adapter dataset has no validation-patient records for best-checkpoint selection.")
    overlap = {record.patient_id for record in train} & {record.patient_id for record in val}
    if overlap:
        raise ValueError(f"Adapter dataset leaks patients across train/val: {sorted(overlap)}")

    return AdapterDatasetManifest(
        path=path,
        split_manifest_path=split_path,
        manifest_sha256=_sha256(path),
        split_manifest_sha256=_sha256(split_path),
        dataset_content_sha256=_dataset_content_sha256(records) if validate_files else "unvalidated",
        train=train,
        val=val,
    )


def target_frame_paths(record: AdapterDatasetRecord) -> tuple[Path, ...]:
    """Return a record's already-validated target ordering."""

    return tuple(record.target_frames_dir / name for name in TARGET_FRAME_NAMES)
