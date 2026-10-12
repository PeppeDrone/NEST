"""Checksum-verified, transactional NPZ + JSON pose artifact IO."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from .pose_records import PredictionPoseSequence, TeacherPoseSequence
from .pose_schema import (
    BODY20_SCOPE,
    NATIVE_STAGE1_FRAMES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    WHOLE62_SCOPE,
)


FORMAT_ID = "dmtbaby.pose"
FORMAT_VERSION = 1
PREDICTION_SCOPES = frozenset({BODY20_SCOPE, WHOLE62_SCOPE})

TEACHER_NPZ_KEYS = frozenset(
    {
        "raw_xy_norm",
        "aligned_xy_norm",
        "raw_score",
        "detector_valid",
        "render_visible",
        "ref_xy_norm",
        "ref_score",
        "ref_visible",
        "source_frame",
    }
)
PREDICTION_NPZ_KEYS = frozenset({"xy_norm", "visibility_logit"})


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for pose artifact IO") from exc
    return np


@dataclass(frozen=True)
class PoseArtifactPaths:
    npz: Path
    json: Path


def _paths(path: str | Path) -> PoseArtifactPaths:
    npz_path = Path(path)
    if npz_path.suffix.lower() != ".npz":
        raise ValueError(f"Pose artifact path must end in .npz: {npz_path}")
    return PoseArtifactPaths(npz=npz_path, json=npz_path.with_suffix(".json"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> str:
    np = _numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        checksum = _sha256(temporary)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return checksum
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_manifest(arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in sorted(arrays.items())
    }


def _validate_prediction_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(metadata or {})
    if value.get("joint_schema") != SCHEMA_ID:
        raise ValueError(f"prediction metadata 'joint_schema' must be {SCHEMA_ID!r}")
    joint_scope = value.get("joint_scope")
    if joint_scope not in PREDICTION_SCOPES:
        raise ValueError(
            f"prediction metadata 'joint_scope' must be one of {sorted(PREDICTION_SCOPES)}"
        )
    checkpoint = value.get("checkpoint_sha256")
    if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint) is None:
        raise ValueError("prediction metadata 'checkpoint_sha256' must be 64 lowercase hex digits")
    seed = value.get("sampling_seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("prediction metadata 'sampling_seed' must be an integer")
    value["sampling_seed"] = int(seed)
    return value


def _write(
    path: str | Path,
    kind: str,
    arrays: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> PoseArtifactPaths:
    paths = _paths(path)
    metadata_value = dict(metadata or {})
    try:
        json.dumps(metadata_value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pose metadata must be finite JSON data: {exc}") from exc
    # NPZ is committed first and JSON last. The JSON checksum is the commit
    # marker; readers reject an interrupted or mixed-generation pair.
    checksum = _atomic_npz(paths.npz, arrays)
    payload = {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "npz_file": paths.npz.name,
        "npz_sha256": checksum,
        "arrays": _array_manifest(arrays),
        "metadata": metadata_value,
    }
    try:
        _atomic_json(paths.json, payload)
    except Exception:
        # An NPZ without a matching JSON commit marker is deliberately
        # unreadable. Leave it in place for forensic recovery.
        raise
    return paths


def write_teacher_pose(
    path: str | Path,
    sequence: TeacherPoseSequence,
    metadata: dict[str, Any] | None = None,
) -> PoseArtifactPaths:
    np = _numpy()
    item = sequence.validated()
    arrays = {
        "raw_xy_norm": np.asarray(item.raw_xy_norm, dtype=np.float32),
        "aligned_xy_norm": np.asarray(item.aligned_xy_norm, dtype=np.float32),
        "raw_score": np.asarray(item.raw_score, dtype=np.float32),
        "detector_valid": np.asarray(item.detector_valid, dtype=np.bool_),
        "render_visible": np.asarray(item.render_visible, dtype=np.bool_),
        "ref_xy_norm": np.asarray(item.ref_xy_norm, dtype=np.float32),
        "ref_score": np.asarray(item.ref_score, dtype=np.float32),
        "ref_visible": np.asarray(item.ref_visible, dtype=np.bool_),
        "source_frame": np.asarray(item.source_frame, dtype=np.int32),
    }
    if frozenset(arrays) != TEACHER_NPZ_KEYS:  # pragma: no cover - maintenance guard
        raise RuntimeError("Internal teacher NPZ contract drift")
    return _write(path, "teacher", arrays, metadata)


def write_prediction_pose(
    path: str | Path,
    sequence: PredictionPoseSequence,
    metadata: dict[str, Any] | None = None,
) -> PoseArtifactPaths:
    np = _numpy()
    item = sequence.validated(expected_frames=NATIVE_STAGE1_FRAMES)
    metadata_value = _validate_prediction_metadata(metadata)
    if metadata_value["joint_scope"] != item.joint_scope:
        raise ValueError(
            "prediction metadata 'joint_scope' does not match the sequence"
        )
    arrays = {
        "xy_norm": np.asarray(item.xy_norm, dtype=np.float32),
        "visibility_logit": np.asarray(item.visibility_logit, dtype=np.float32),
    }
    return _write(path, "prediction", arrays, metadata_value)


def _metadata(paths: PoseArtifactPaths, expected_kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(paths.json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid pose metadata file {paths.json}: {exc}") from exc
    expected = {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "kind": expected_kind,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "npz_file": paths.npz.name,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Pose metadata {field!r} is {payload.get(field)!r}; expected {value!r}"
            )
    checksum = payload.get("npz_sha256")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("Pose metadata has no valid NPZ SHA-256")
    return payload


def _read_arrays(
    path: str | Path,
    expected_kind: str,
    expected_keys: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    np = _numpy()
    paths = _paths(path)
    mismatch: str | None = None
    # A writer may be between the NPZ and JSON commit. Re-read the commit
    # marker a few times without sleeping; persistent mismatch is corruption.
    for _ in range(3):
        payload = _metadata(paths, expected_kind)
        actual = _sha256(paths.npz)
        if actual == payload["npz_sha256"]:
            break
        mismatch = f"NPZ checksum mismatch for {paths.npz}"
    else:
        raise ValueError(mismatch or f"Could not read consistent artifact {paths.npz}")

    with np.load(paths.npz, allow_pickle=False) as archive:
        keys = frozenset(archive.files)
        if keys != expected_keys:
            raise ValueError(
                f"Unexpected NPZ keys: got {sorted(keys)}, expected {sorted(expected_keys)}"
            )
        arrays = {name: archive[name] for name in archive.files}
    declared = payload.get("arrays")
    if declared != _array_manifest(arrays):
        raise ValueError("NPZ array manifest does not match JSON metadata")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Pose metadata 'metadata' must be an object")
    return arrays, dict(metadata)


def read_teacher_pose(path: str | Path) -> tuple[TeacherPoseSequence, dict[str, Any]]:
    arrays, metadata = _read_arrays(path, "teacher", TEACHER_NPZ_KEYS)
    sequence = TeacherPoseSequence(**arrays).validated()
    return sequence, metadata


def read_prediction_pose(
    path: str | Path,
) -> tuple[PredictionPoseSequence, dict[str, Any]]:
    arrays, metadata = _read_arrays(path, "prediction", PREDICTION_NPZ_KEYS)
    metadata = _validate_prediction_metadata(metadata)
    sequence = PredictionPoseSequence(
        **arrays, joint_scope=metadata["joint_scope"]
    ).validated(expected_frames=NATIVE_STAGE1_FRAMES)
    return sequence, metadata
