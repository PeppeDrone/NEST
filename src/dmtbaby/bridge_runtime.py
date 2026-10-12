"""Fingerprint-aware structural bridge from native-20 predictions to DWPose RGB."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    fingerprint,
    sha256_file,
)
from .manifests import source_fingerprint
from .pose.io import read_prediction_pose, read_teacher_pose
from .pose.processing import (
    expand_prediction_to_whole62,
    process_native20_prediction,
    resample_native20_to81,
)
from .pose.records import CanonicalPoseFrame, PredictionPoseSequence
from .pose.render import (
    UpstreamDrawPoseAdapter,
    load_strict_pose_directory,
    render_pose_directory,
)
from .pose.schema import (
    BODY20_SCOPE,
    NUM_BODY_JOINTS,
    NUM_JOINTS,
    SCHEMA_ID,
    UNIANIMATE_FRAMES,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"Missing clip_id at {path}:{line_number}")
            row = dict(row)
            row["clip_id"] = clip_id
            rows.append(row)
    identifiers = [str(row["clip_id"]) for row in rows]
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate clip_id values in {path}: {duplicates}")
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _artifact_path(row: Mapping[str, Any], base: Path, *names: str) -> Path:
    for name in names:
        value = row.get(name)
        if value:
            path = Path(str(value)).expanduser()
            return path.resolve() if path.is_absolute() else (base / path).resolve()
    raise ValueError(f"{row.get('clip_id', '<unknown>')}: missing one of {names}")


def _paired_hash(path: Path) -> str:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Incomplete NPZ+JSON artifact pair: {path}")
    return fingerprint(
        {
            "npz": sha256_file(path),
            "json": sha256_file(sidecar),
        }
    )


def _safe_id(value: str) -> str:
    rendered = value.replace("/", "_").replace(os.sep, "_")
    if rendered in {"", ".", ".."}:
        raise ValueError(f"Unsafe clip_id: {value!r}")
    return rendered


@dataclass(frozen=True)
class RenderControlsRequest:
    root: Path
    prediction_manifest: Path
    teacher_manifest: Path
    output_root: Path
    unianimate_dir: Path
    postprocess: str = "constrained"
    persist: str = "test"
    clip_ids: tuple[str, ...] = ()
    overwrite: bool = False
    jpeg_quality: int = 95
    smoke_test: bool = False
    dry_run: bool = False

    @classmethod
    def from_namespace(cls, args: Any, root: str | Path) -> "RenderControlsRequest":
        project_root = Path(root).expanduser().resolve()
        return cls(
            root=project_root,
            prediction_manifest=_resolve(project_root, args.prediction_manifest),
            teacher_manifest=_resolve(project_root, args.teacher_manifest),
            output_root=_resolve(project_root, args.output_root),
            unianimate_dir=_resolve(
                project_root,
                getattr(args, "unianimate_dir", "external/UniAnimate-DiT"),
            ),
            postprocess=str(args.postprocess),
            persist=str(args.persist),
            clip_ids=tuple(str(value) for value in args.clip_id),
            overwrite=bool(args.overwrite),
            jpeg_quality=int(args.jpeg_quality),
            smoke_test=bool(getattr(args, "smoke_test", False)),
            dry_run=bool(args.dry_run),
        )

    def validate(self) -> None:
        if self.postprocess not in {"raw", "constrained"}:
            raise ValueError("postprocess must be raw or constrained")
        if self.persist not in {"all", "test", "none"}:
            raise ValueError("persist must be all, test, or none")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must lie in [1,100]")
        if not self.prediction_manifest.is_file():
            raise FileNotFoundError(self.prediction_manifest)
        if not self.teacher_manifest.is_file():
            raise FileNotFoundError(self.teacher_manifest)
        if not (self.unianimate_dir / "run_align_pose.py").is_file():
            raise FileNotFoundError(self.unianimate_dir / "run_align_pose.py")


def _raw_bridge(
    source: PredictionPoseSequence,
    reference_xy: np.ndarray,
    reference_visible: np.ndarray,
) -> PredictionPoseSequence:
    """Anchor, resample, sigmoid-gate and clamp without structural cleanup."""

    item = expand_prediction_to_whole62(source)
    ref_xy = np.asarray(reference_xy, dtype=np.float32).copy()
    ref_visible = np.asarray(reference_visible, dtype=np.bool_).copy()
    if source.joint_scope == BODY20_SCOPE:
        ref_xy[NUM_BODY_JOINTS:] = np.nan
        ref_visible[NUM_BODY_JOINTS:] = False
    native_xy = item.xy_norm.copy()
    native_logits = item.visibility_logit.copy()
    native_xy[0] = np.nan
    native_xy[0, ref_visible] = ref_xy[ref_visible]
    native_logits[0] = -20.0
    native_logits[0, ref_visible] = 20.0
    native = PredictionPoseSequence(native_xy, native_logits).validated()
    output = resample_native20_to81(native)
    xy = output.xy_norm.copy()
    logits = output.visibility_logit.copy()
    probability = output.visibility_prob
    valid = (probability >= 0.5) & np.isfinite(xy).all(axis=-1)
    xy[valid] = np.clip(xy[valid], 0.0, 1.0)
    xy[~valid] = np.nan
    logits[~valid] = -20.0
    xy[0] = np.nan
    xy[0, ref_visible] = ref_xy[ref_visible]
    logits[0] = -20.0
    logits[0, ref_visible] = 20.0
    return PredictionPoseSequence(xy, logits).validated(
        expected_frames=UNIANIMATE_FRAMES
    )


def _reference(sequence: Any) -> CanonicalPoseFrame:
    # The public teacher contract intentionally exposes only ref_visible.  It
    # therefore acts as both availability and renderability for this frame.
    return CanonicalPoseFrame(
        xy_norm=sequence.ref_xy_norm,
        raw_score=sequence.ref_score,
        detector_valid=sequence.ref_visible,
        render_visible=sequence.ref_visible,
    ).validated()


def _renderer_fingerprint(request: RenderControlsRequest) -> str:
    files = (
        request.unianimate_dir / "run_align_pose.py",
        request.unianimate_dir / "dwpose/util.py",
    )
    return fingerprint({str(path): sha256_file(path) for path in files})


def _status_path(root: Path, clip_id: str) -> Path:
    return root / "status" / f"{_safe_id(clip_id)}.json"


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _control_is_complete(path: Path) -> bool:
    try:
        load_strict_pose_directory(path)
        return True
    except (OSError, ValueError):
        return False


def _control_tree_sha256(path: Path) -> str:
    load_strict_pose_directory(path)
    return fingerprint(
        [
            (name, sha256_file(path / name))
            for name in [*(f"{index:04d}.jpg" for index in range(UNIANIMATE_FRAMES)), "pose.jpg"]
        ]
    )


def _conditioned_is_complete(
    npz_path: Path, expected_fingerprint: str
) -> tuple[bool, str | None]:
    sidecar = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not sidecar.is_file():
        return False, None
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "xy_norm",
                "visibility_logit",
                "render_visible",
            }:
                return False, None
            xy = payload["xy_norm"]
            logits = payload["visibility_logit"]
            visible = payload["render_visible"]
        if xy.shape != (UNIANIMATE_FRAMES, NUM_JOINTS, 2) or xy.dtype != np.float32:
            return False, None
        if logits.shape != (UNIANIMATE_FRAMES, NUM_JOINTS) or logits.dtype != np.float32:
            return False, None
        if visible.shape != (UNIANIMATE_FRAMES, NUM_JOINTS) or visible.dtype != np.bool_:
            return False, None
        finite = np.isfinite(xy).all(axis=-1)
        if not finite[visible].all() or not np.isnan(xy[~visible]).all():
            return False, None
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            metadata.get("format") != "dmtbaby.conditioned_pose.v1"
            or metadata.get("fingerprint") != expected_fingerprint
            or metadata.get("npz_sha256") != sha256_file(npz_path)
        ):
            return False, None
        return True, _paired_hash(npz_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, None


def _backup_existing(path: Path, output_root: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    relative = path.relative_to(output_root)
    destination = output_root / "replaced" / stamp / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)


def _conditioned_arrays(sequence: PredictionPoseSequence) -> dict[str, Any]:
    visible = (
        np.asarray(sequence.visibility_prob >= 0.5, dtype=np.bool_)
        & np.isfinite(sequence.xy_norm).all(axis=-1)
    )
    xy = np.asarray(sequence.xy_norm, dtype=np.float32).copy()
    xy[~visible] = np.nan
    return {
        "xy_norm": xy,
        "visibility_logit": np.asarray(sequence.visibility_logit, dtype=np.float32),
        "render_visible": visible,
    }


def render_controls(request: RenderControlsRequest) -> int:
    request.validate()
    predictions = _read_jsonl(request.prediction_manifest)
    teacher_rows = _read_jsonl(request.teacher_manifest)
    teachers = {str(row["clip_id"]): row for row in teacher_rows}
    selected = set(request.clip_ids)
    if selected:
        unknown = selected - {str(row["clip_id"]) for row in predictions}
        if unknown:
            raise ValueError(f"Unknown requested clip_ids: {sorted(unknown)}")
        predictions = [row for row in predictions if str(row["clip_id"]) in selected]
    renderer_hash = _renderer_fingerprint(request)
    code_hash = source_fingerprint(request.root)
    renderer: UpstreamDrawPoseAdapter | None = None
    rows_out: list[dict[str, Any]] = []
    errors = 0

    if not request.dry_run and not request.smoke_test:
        from .io import require_free_disk

        require_free_disk(
            request.output_root,
            100.0,
            operation="production structural bridge",
        )

    for row in predictions:
        if not request.dry_run and not request.smoke_test:
            require_free_disk(
                request.output_root,
                50.0,
                operation="structural bridge headroom",
            )
        clip_id = str(row["clip_id"])
        split = str(row.get("split", teachers.get(clip_id, {}).get("split", "")))
        teacher_row = teachers.get(clip_id)
        if teacher_row is None:
            raise ValueError(f"{clip_id}: absent from teacher manifest")
        prediction_path = _artifact_path(
            row, request.prediction_manifest.parent, "prediction_npz", "artifact"
        )
        teacher_path = _artifact_path(
            teacher_row, request.teacher_manifest.parent, "pose_npz", "teacher_npz"
        )
        persist_control = request.persist == "all" or (
            request.persist == "test" and split == "test"
        )
        config = {
            "postprocess": request.postprocess,
            "persist": request.persist,
            "jpeg_quality": request.jpeg_quality,
            "schema": SCHEMA_ID,
        }
        expected = fingerprint(
            {
                "config": config,
                "code": code_hash,
                "renderer": renderer_hash,
                "prediction": _paired_hash(prediction_path),
                "teacher": _paired_hash(teacher_path),
            }
        )
        safe = _safe_id(clip_id)
        conditioned = request.output_root / "conditioned" / f"{safe}.npz"
        conditioned_json = conditioned.with_suffix(".json")
        control_dir = request.output_root / "controls" / safe
        status_path = _status_path(request.output_root, clip_id)
        existing = _read_status(status_path)
        conditioned_ok, conditioned_pair_sha256 = _conditioned_is_complete(
            conditioned, expected
        )
        control_tree_sha256 = None
        if persist_control and _control_is_complete(control_dir):
            control_tree_sha256 = _control_tree_sha256(control_dir)
        resume_ok = (
            existing is not None
            and existing.get("state") == "complete"
            and existing.get("fingerprint") == expected
            and conditioned_ok
            and existing.get("conditioned_pair_sha256") == conditioned_pair_sha256
            and (
                not persist_control
                or existing.get("control_tree_sha256") == control_tree_sha256
            )
        )
        propagated_source = {**teacher_row, **row}
        propagated = {
            key: propagated_source[key]
            for key in (
                "manual_prompt",
                "automatic_prompt",
                "prompt",
                "patient_id",
                "sampling_seed",
            )
            if key in propagated_source
        }
        for path_key in ("reference_image", "target_frames_dir", "source_video"):
            if row.get(path_key):
                propagated[path_key] = str(
                    _artifact_path(row, request.prediction_manifest.parent, path_key)
                )
            elif teacher_row.get(path_key):
                propagated[path_key] = str(
                    _artifact_path(teacher_row, request.teacher_manifest.parent, path_key)
                )
        output_row: dict[str, Any] = {
            "clip_id": clip_id,
            "split": split,
            "prediction_npz": str(prediction_path),
            "teacher_npz": str(teacher_path),
            "conditioned_npz": str(conditioned),
            "control_dir": str(control_dir) if persist_control else None,
            "reference_pose": str(control_dir / "pose.jpg") if persist_control else None,
            "postprocess": request.postprocess,
            "prompt_source": str(row.get("prompt_source", "")),
            "joint_scope": str(row.get("joint_scope", "")),
            "fingerprint": expected,
            **propagated,
        }
        if resume_ok:
            output_row["status"] = "resumed"
            rows_out.append(output_row)
            continue

        try:
            prediction, prediction_metadata = read_prediction_pose(prediction_path)
            teacher, teacher_metadata = read_teacher_pose(teacher_path)
            if request.postprocess == "constrained":
                sequence, diagnostics = process_native20_prediction(
                    prediction, teacher.ref_xy_norm, teacher.ref_visible
                )
                diagnostic_payload: dict[str, Any] = asdict(diagnostics)
            else:
                sequence = _raw_bridge(
                    prediction, teacher.ref_xy_norm, teacher.ref_visible
                )
                diagnostic_payload = {"mode": "raw"}
            if request.dry_run:
                output_row["status"] = "validated"
                rows_out.append(output_row)
                continue
            occupied = [conditioned, conditioned_json]
            if persist_control:
                occupied.append(control_dir)
            if any(path.exists() for path in occupied):
                if not request.overwrite:
                    raise FileExistsError(
                        f"{clip_id}: stale output exists; pass --overwrite to archive and replace it"
                    )
                for path in occupied:
                    _backup_existing(path, request.output_root)
            atomic_write_npz(conditioned, _conditioned_arrays(sequence))
            atomic_write_json(
                conditioned_json,
                {
                    "format": "dmtbaby.conditioned_pose.v1",
                    "clip_id": clip_id,
                    "schema": SCHEMA_ID,
                    "joint_scope": prediction.joint_scope,
                    "frames": UNIANIMATE_FRAMES,
                    "postprocess": request.postprocess,
                    "fingerprint": expected,
                    "npz_sha256": sha256_file(conditioned),
                    "prediction_metadata": prediction_metadata,
                    "teacher_metadata": teacher_metadata,
                    "diagnostics": diagnostic_payload,
                },
            )
            if persist_control:
                if renderer is None:
                    renderer = UpstreamDrawPoseAdapter(request.unianimate_dir)
                render_pose_directory(
                    control_dir,
                    sequence,
                    _reference(teacher),
                    renderer,
                    jpeg_quality=request.jpeg_quality,
                )
            conditioned_pair_sha256 = _paired_hash(conditioned)
            control_tree_sha256 = (
                _control_tree_sha256(control_dir) if persist_control else None
            )
            atomic_write_json(
                status_path,
                {
                    "clip_id": clip_id,
                    "state": "complete",
                    "fingerprint": expected,
                    "conditioned_pair_sha256": conditioned_pair_sha256,
                    "control_tree_sha256": control_tree_sha256,
                    "artifacts": {
                        "conditioned_npz": str(conditioned),
                        "conditioned_json": str(conditioned_json),
                        "control_dir": str(control_dir) if persist_control else None,
                    },
                },
            )
            output_row["status"] = "complete"
        except Exception as exc:
            errors += 1
            output_row["status"] = "error"
            output_row["error"] = f"{type(exc).__name__}: {exc}"
            if not request.dry_run:
                atomic_write_json(
                    status_path,
                    {
                        "clip_id": clip_id,
                        "state": "error",
                        "fingerprint": expected,
                        "error": output_row["error"],
                    },
                )
        rows_out.append(output_row)

    summary = {
        "selected": len(predictions),
        "errors": errors,
        "controls": sum(bool(row.get("control_dir")) for row in rows_out),
        "dry_run": request.dry_run,
    }
    if not request.dry_run:
        atomic_write_jsonl(request.output_root / "control_manifest.jsonl", rows_out)
        adapter_rows = []
        for row in rows_out:
            if (
                row.get("status") not in {"complete", "resumed"}
                or row.get("split") not in {"train", "val"}
                or row.get("prompt_source") != "manual"
                or row.get("joint_scope") != "whole62"
                or row.get("postprocess") != "constrained"
                or not row.get("control_dir")
            ):
                continue
            adapter_rows.append(
                {
                    "clip_id": row["clip_id"],
                    "split": row["split"],
                    "prompt": row.get("manual_prompt") or row.get("prompt"),
                    "control_dir": row["control_dir"],
                    "reference_pose": row["reference_pose"],
                    "reference_image": row.get("reference_image"),
                    "target_frames_dir": row.get("target_frames_dir"),
                    "control_variant": "manual-whole62-constrained",
                    "prompt_source": "manual",
                    "joint_count": 62,
                    "postprocessing": "constrained",
                    "control_source": "stage1_prediction",
                }
            )
        if adapter_rows:
            atomic_write_jsonl(
                request.output_root / "stage2_adapter_manifest.jsonl", adapter_rows
            )
    print(json.dumps(summary, sort_keys=True))
    return 1 if errors else 0
