"""Strict manifest/NPZ loader for coordinate-teacher Stage-1 datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .topology import joint_names


_PATH_COLUMNS = ("pose_npz", "pose_path", "npz_path")
_ID_COLUMNS = ("clip_id", "sample_id", "id")


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [dict(record) for record in csv.DictReader(stream)]
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"manifest line {line_number} is not an object")
                records.append(value)
        return records
    raise ValueError("teacher manifest must be CSV or JSONL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TeacherPoseDataset:
    """Load one frozen prompt source and normalized 20-frame teacher tracks.

    Canonical mode requires the checksum-committed JSON sidecar and exact
    nine-key teacher NPZ contract: ``raw_xy_norm``, ``aligned_xy_norm``,
    ``raw_score``, ``detector_valid``, ``render_visible``, ``ref_xy_norm``,
    ``ref_score``, ``ref_visible``, and ``source_frame``. The trajectory has
    shape ``[20,J,2]``; index zero is not trained and targets are 1..19.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        prompt_source: str,
        joint_count: int,
        split: str | None = None,
        data_root: str | Path | None = None,
        expected_frames: int = 20,
        verify_checksums: bool = False,
        require_joint_names: bool = False,
        allow_legacy_aliases: bool = False,
        annotation_csv: str | Path | None = None,
        moving_labelled_only: bool = False,
        high_amplitude_scale: float = 2.0,
        fast_speed_scale: float = 2.0,
    ) -> None:
        if prompt_source not in {"manual", "automatic"}:
            raise ValueError("prompt_source must be 'manual' or 'automatic'")
        if joint_count not in (20, 62):
            raise ValueError("joint_count must be 20 or 62")
        if expected_frames != 20:
            raise ValueError("Stage-1 teacher trajectories must contain 20 frames")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.manifest_path.parent
        )
        self.prompt_source = prompt_source
        self.prompt_column = f"{prompt_source}_prompt"
        self.joint_count = joint_count
        self.expected_frames = expected_frames
        self.verify_checksums = verify_checksums
        self.require_joint_names = require_joint_names
        self.allow_legacy_aliases = allow_legacy_aliases
        raw_records = _load_manifest(self.manifest_path)
        self.records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(raw_records):
            record_split = str(raw.get("split", "")).strip()
            if split is not None and record_split != split:
                continue
            sample_id = _first(raw, _ID_COLUMNS)
            relative_path = _first(raw, _PATH_COLUMNS)
            prompt = raw.get(self.prompt_column)
            if sample_id is None:
                raise ValueError(f"manifest record {index} has no sample identifier")
            sample_id = str(sample_id).strip()
            if sample_id in seen_ids:
                raise ValueError(f"duplicate teacher sample identifier {sample_id!r}")
            if relative_path is None:
                raise ValueError(f"manifest record {sample_id!r} has no pose NPZ path")
            if prompt is None or not str(prompt).strip():
                raise ValueError(
                    f"manifest record {sample_id!r} has no {self.prompt_column}"
                )
            pose_path = Path(str(relative_path)).expanduser()
            if not pose_path.is_absolute():
                pose_path = self.data_root / pose_path
            pose_path = pose_path.resolve()
            if not pose_path.is_file():
                raise FileNotFoundError(pose_path)
            seen_ids.add(sample_id)
            self.records.append(
                {
                    **raw,
                    "sample_id": sample_id,
                    "pose_path": pose_path,
                    "prompt": str(prompt).strip(),
                }
            )
        if moving_labelled_only and annotation_csv is None:
            raise ValueError("moving_labelled_only requires annotation_csv")
        self.annotation_csv: Path | None = None
        self.moving_labelled_only = bool(moving_labelled_only)
        if annotation_csv is not None:
            # Legacy annotation-CSV path (moving-only recipes); never used by NEST, which reads
            # prompts from the manifest only. Kept as an explicit error rather than dead imports.
            raise NotImplementedError("annotation_csv is not supported in the neomotion package")
            from .motion_labels import (  # noqa: F401  (unreachable, legacy)
                joint_motion_weights,
                limb_condition_codes,
                load_clip_motion_tables,
                tables_for_clip,
            )

            self.annotation_csv = Path(annotation_csv).expanduser().resolve()
            labels, grades = load_clip_motion_tables(self.annotation_csv)
            kept: list[dict[str, Any]] = []
            for record in self.records:
                clip_labels, clip_grades = tables_for_clip(
                    str(record["sample_id"]), labels, grades
                )
                mask, scale = joint_motion_weights(
                    clip_labels,
                    clip_grades,
                    joint_count=self.joint_count,
                    high_amplitude_scale=high_amplitude_scale,
                    fast_speed_scale=fast_speed_scale,
                )
                record["_moving_joint_mask"] = mask
                record["_motion_scale"] = scale
                record["_limb_features"] = limb_condition_codes(clip_labels, clip_grades)
                if self.moving_labelled_only and not any(clip_labels.values()):
                    continue
                kept.append(record)
            if self.moving_labelled_only:
                if not kept:
                    qualifier = f" for split {split!r}" if split is not None else ""
                    raise ValueError(
                        f"no moving-labelled teacher records remain{qualifier}"
                    )
                self.records = kept
        if not self.records:
            qualifier = f" for split {split!r}" if split is not None else ""
            raise ValueError(f"teacher manifest has no records{qualifier}")
        self.manifest_sha256 = _sha256(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np
        import torch

        record = self.records[index]
        path = record["pose_path"]
        declared_checksum = str(
            record.get("pose_sha256") or record.get("sha256") or ""
        ).strip().lower()
        if self.verify_checksums:
            if not declared_checksum:
                raise ValueError(f"teacher record {record['sample_id']!r} has no sha256")
            actual_checksum = _sha256(path)
            if actual_checksum != declared_checksum:
                raise ValueError(f"teacher checksum mismatch for {record['sample_id']!r}")

        artifact_metadata: dict[str, Any] = {}
        if not self.allow_legacy_aliases:
            # The JSON checksum is the transaction commit marker. Reading the
            # NPZ directly would accept interrupted writes, wrong schemas, and
            # archives with a merely compatible subset of the canonical keys.
            from .pose_io import read_teacher_pose
            from .pose_schema import JOINT_NAMES

            sequence, artifact_metadata = read_teacher_pose(path)
            coordinates = np.asarray(sequence.aligned_xy_norm, dtype=np.float32)
            confidence = np.asarray(sequence.raw_score, dtype=np.float32)
            valid = np.asarray(sequence.render_visible, dtype=np.bool_)
            reference_coordinates = np.asarray(sequence.ref_xy_norm, dtype=np.float32)
            reference_confidence = np.asarray(sequence.ref_score, dtype=np.float32)
            reference_valid = np.asarray(sequence.ref_visible, dtype=np.bool_)
            stored_names = tuple(JOINT_NAMES)
        else:
            # Explicit compatibility path for pre-canonical standalone NPZs.
            with np.load(path, allow_pickle=False) as archive:
                coordinate_key = "aligned_xy_norm"
                score_key = "raw_score"
                visible_key = "render_visible"
                reference_coordinate_key = "ref_xy_norm"
                reference_score_key = "ref_score"
                reference_visible_key = "ref_visible"
                if coordinate_key not in archive:
                    coordinate_key = "coords" if "coords" in archive else "coordinates"
                if score_key not in archive:
                    score_key = "scores" if "scores" in archive else "confidence"
                if visible_key not in archive:
                    visible_key = "valid"
                if (
                    coordinate_key not in archive
                    or score_key not in archive
                    or visible_key not in archive
                ):
                    raise ValueError(
                        f"{path} must contain ref_xy_norm/ref_score/ref_visible and "
                        "aligned_xy_norm/raw_score/render_visible"
                    )
                coordinates = np.asarray(archive[coordinate_key], dtype=np.float32)
                confidence = np.asarray(archive[score_key], dtype=np.float32)
                valid = np.asarray(archive[visible_key], dtype=np.bool_)
                if (
                    reference_coordinate_key in archive
                    and reference_score_key in archive
                    and reference_visible_key in archive
                ):
                    reference_coordinates = np.asarray(
                        archive[reference_coordinate_key], dtype=np.float32
                    )
                    reference_confidence = np.asarray(
                        archive[reference_score_key], dtype=np.float32
                    )
                    reference_valid = np.asarray(
                        archive[reference_visible_key], dtype=np.bool_
                    )
                else:
                    reference_coordinates = coordinates[0].copy()
                    reference_confidence = confidence[0].copy()
                    reference_valid = valid[0].copy()
                stored_names = None
                if "joint_names" in archive:
                    stored_names = tuple(
                        str(value) for value in archive["joint_names"].tolist()
                    )
                if "coordinate_space" in archive:
                    coordinate_space = str(
                        np.asarray(archive["coordinate_space"]).item()
                    )
                    if coordinate_space != "normalized_0_1":
                        raise ValueError(
                            f"unsupported coordinate_space {coordinate_space!r} in {path}"
                        )

        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(f"teacher coordinates in {path} must have shape [20,J,2]")
        frames, stored_joints, _ = coordinates.shape
        if stored_joints not in (20, 62):
            raise ValueError(f"teacher archive {path} must contain exactly 20 or 62 joints")
        if frames != self.expected_frames or stored_joints < self.joint_count:
            raise ValueError(
                f"teacher shape {coordinates.shape} cannot supply "
                f"[{self.expected_frames},{self.joint_count},2]"
            )
        if confidence.shape != (frames, stored_joints) or valid.shape != (frames, stored_joints):
            raise ValueError(f"teacher confidence/valid shapes do not match coordinates in {path}")
        if (
            reference_coordinates.shape != (stored_joints, 2)
            or reference_confidence.shape != (stored_joints,)
            or reference_valid.shape != (stored_joints,)
        ):
            raise ValueError(f"reference pose shapes do not match teacher joints in {path}")
        expected_names = joint_names(self.joint_count)
        if self.require_joint_names and stored_names is None:
            raise ValueError(f"teacher archive {path} has no joint_names")
        if stored_names is not None and stored_names[: self.joint_count] != expected_names:
            raise ValueError(f"teacher archive {path} has a noncanonical joint order")

        coordinates = coordinates[:, : self.joint_count].copy()
        confidence = confidence[:, : self.joint_count].copy()
        valid = valid[:, : self.joint_count].copy()
        reference_coordinates = reference_coordinates[: self.joint_count].copy()
        reference_confidence = reference_confidence[: self.joint_count].copy()
        reference_valid = reference_valid[: self.joint_count].copy()
        valid_values = coordinates[valid]
        if valid_values.size and (
            not np.isfinite(valid_values).all()
            or float(valid_values.min()) < 0.0
            or float(valid_values.max()) > 1.0
        ):
            raise ValueError(f"valid coordinates in {path} must be finite and normalized to [0,1]")
        valid_confidence = confidence[valid]
        if valid_confidence.size and (
            not np.isfinite(valid_confidence).all()
            or float(valid_confidence.min()) < 0.0
            or float(valid_confidence.max()) > 1.0
        ):
            raise ValueError(f"valid confidence in {path} must be finite and in [0,1]")
        coordinates[~valid] = 0.0
        confidence[~valid] = 0.0
        valid_reference_values = reference_coordinates[reference_valid]
        if valid_reference_values.size and (
            not np.isfinite(valid_reference_values).all()
            or float(valid_reference_values.min()) < 0.0
            or float(valid_reference_values.max()) > 1.0
        ):
            raise ValueError(
                f"visible reference coordinates in {path} must be finite and in [0,1]"
            )
        valid_reference_confidence = reference_confidence[reference_valid]
        if valid_reference_confidence.size and (
            not np.isfinite(valid_reference_confidence).all()
            or float(valid_reference_confidence.min()) < 0.0
            or float(valid_reference_confidence.max()) > 1.0
        ):
            raise ValueError(
                f"visible reference confidence in {path} must be finite and in [0,1]"
            )
        reference_coordinates[~reference_valid] = 0.0
        reference_confidence[~reference_valid] = 0.0

        coordinate_tensor = torch.from_numpy(coordinates)
        confidence_tensor = torch.from_numpy(confidence)
        valid_tensor = torch.from_numpy(valid)
        reference_coordinate_tensor = torch.from_numpy(reference_coordinates)
        reference_confidence_tensor = torch.from_numpy(reference_confidence)
        reference_valid_tensor = torch.from_numpy(reference_valid)
        sample = {
            "sample_id": record["sample_id"],
            "prompt": record["prompt"],
            "prompt_source": self.prompt_source,
            "pose_path": str(path),
            "teacher_artifact_metadata": artifact_metadata,
            "pose0_xy_norm": reference_coordinate_tensor,
            "pose0_raw_score": reference_confidence_tensor,
            "pose0_render_visible": reference_valid_tensor,
            "target_xy_norm": coordinate_tensor[1:],
            "target_raw_score": confidence_tensor[1:],
            "target_render_visible": valid_tensor[1:],
        }
        if "_moving_joint_mask" in record:
            sample["moving_joint_mask"] = torch.from_numpy(
                record["_moving_joint_mask"].copy()
            )
            sample["motion_scale"] = torch.from_numpy(record["_motion_scale"].copy())
        if "_limb_features" in record:
            sample["limb_features"] = torch.from_numpy(record["_limb_features"].copy())
        return sample

    def provenance(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "dmtbaby.stage1.teacher_npz",
            "schema_version": 1,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "prompt_source": self.prompt_source,
            "joint_count": self.joint_count,
            "frame_count": self.expected_frames,
            "sample_count": len(self),
            "canonical_npz_keys": [
                "raw_xy_norm",
                "ref_xy_norm",
                "ref_score",
                "ref_visible",
                "aligned_xy_norm",
                "raw_score",
                "detector_valid",
                "render_visible",
                "source_frame",
            ],
            "legacy_aliases_enabled": self.allow_legacy_aliases,
        }
        if self.annotation_csv is not None:
            payload["annotation_csv"] = str(self.annotation_csv)
            payload["moving_labelled_only"] = self.moving_labelled_only
        return payload


def collate_teacher_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    rows = list(samples)
    if not rows:
        raise ValueError("cannot collate an empty teacher batch")
    tensor_keys = (
        "pose0_xy_norm",
        "pose0_raw_score",
        "pose0_render_visible",
        "target_xy_norm",
        "target_raw_score",
        "target_render_visible",
    )
    result = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    for key in ("moving_joint_mask", "motion_scale", "limb_features"):
        if key in rows[0]:
            result[key] = torch.stack([row[key] for row in rows])
    for key in (
        "sample_id",
        "prompt",
        "prompt_source",
        "pose_path",
        "teacher_artifact_metadata",
    ):
        result[key] = [row[key] for row in rows]
    return result
