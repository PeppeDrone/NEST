"""Validated in-memory records shared by extraction, training, and rendering.

The on-disk contract deliberately uses NaN for an unavailable coordinate and a
separate boolean mask for its meaning.  ``-1`` is reserved for the transient
dictionary handed to UniAnimate's unmodified DWPose renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import (
    BODY20_SCOPE,
    NATIVE_STAGE1_FRAMES,
    NUM_BODY_JOINTS,
    NUM_JOINTS,
    SCHEMA_ID,
    WHOLE62_SCOPE,
)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for pose records") from exc
    return np


def _bool_array(name: str, value: Any, shape: tuple[int, ...]) -> Any:
    np = _numpy()
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype != np.bool_:
        # Do not silently reinterpret arbitrary integer masks.
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must be boolean")
    return np.ascontiguousarray(array.astype(np.bool_, copy=False))


def _score_array(name: str, value: Any, shape: tuple[int, ...]) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all() or (array < 0.0).any() or (array > 1.0).any():
        raise ValueError(f"{name} must contain finite raw scores in [0,1]")
    return np.ascontiguousarray(array)


def _masked_xy(name: str, value: Any, mask: Any, shape: tuple[int, ...]) -> Any:
    """Require finite coordinates exactly where ``mask`` is true, NaN elsewhere."""

    np = _numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape + (2,):
        raise ValueError(f"{name} must have shape {shape + (2,)}, got {array.shape}")
    if np.isinf(array).any():
        raise ValueError(f"{name} contains Inf")
    coordinate_mask = np.asarray(mask, dtype=np.bool_)
    finite = np.isfinite(array)
    partially_defined = finite.any(axis=-1) & ~finite.all(axis=-1)
    if partially_defined.any():
        raise ValueError(f"{name} contains a half-defined (x/y) coordinate")
    if not finite[coordinate_mask].all():
        raise ValueError(f"{name} must be finite where its validity mask is true")
    if not np.isnan(array[~coordinate_mask]).all():
        raise ValueError(f"{name} must be NaN where its validity mask is false")
    return np.ascontiguousarray(array)


def _joint_count(scope: str) -> int:
    if scope == BODY20_SCOPE:
        return NUM_BODY_JOINTS
    if scope == WHOLE62_SCOPE:
        return NUM_JOINTS
    raise ValueError(
        f"joint_scope must be {BODY20_SCOPE!r} or {WHOLE62_SCOPE!r}, got {scope!r}"
    )


def _prediction_xy(value: Any, joints: int) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (joints, 2):
        raise ValueError(
            f"xy_norm must have shape [T,{joints},2], got {array.shape}"
        )
    if np.isinf(array).any():
        raise ValueError("xy_norm contains Inf")
    finite = np.isfinite(array)
    if (finite.any(axis=-1) & ~finite.all(axis=-1)).any():
        raise ValueError("xy_norm contains a half-defined (x/y) coordinate")
    return np.ascontiguousarray(array)


def _visibility_logits(value: Any, frames: int, joints: int) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=np.float32)
    expected = (frames, joints)
    if array.shape != expected:
        raise ValueError(f"visibility_logit must have shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("visibility_logit must be finite")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class CanonicalPoseFrame:
    """One extracted canonical no-face DWPose frame.

    ``detector_valid`` records whether a coordinate was returned at all;
    ``render_visible`` additionally applies the configured pose threshold.
    Raw confidence values are preserved unchanged.
    """

    xy_norm: Any
    raw_score: Any
    detector_valid: Any
    render_visible: Any
    schema_id: str = SCHEMA_ID

    def validated(self) -> "CanonicalPoseFrame":
        if self.schema_id != SCHEMA_ID:
            raise ValueError(f"Unexpected schema {self.schema_id!r}")
        valid = _bool_array("detector_valid", self.detector_valid, (NUM_JOINTS,))
        visible = _bool_array("render_visible", self.render_visible, (NUM_JOINTS,))
        if (visible & ~valid).any():
            raise ValueError("render_visible must be a subset of detector_valid")
        xy = _masked_xy("xy_norm", self.xy_norm, valid, (NUM_JOINTS,))
        score = _score_array("raw_score", self.raw_score, (NUM_JOINTS,))
        return CanonicalPoseFrame(xy, score, valid, visible, self.schema_id)


@dataclass(frozen=True)
class TeacherPoseSequence:
    """Canonical native-20 DWPose teacher artifact.

    Field names intentionally match the exact NPZ key contract.
    """

    raw_xy_norm: Any
    aligned_xy_norm: Any
    raw_score: Any
    detector_valid: Any
    render_visible: Any
    ref_xy_norm: Any
    ref_score: Any
    ref_visible: Any
    source_frame: Any
    schema_id: str = SCHEMA_ID

    def validated(self) -> "TeacherPoseSequence":
        np = _numpy()
        if self.schema_id != SCHEMA_ID:
            raise ValueError(f"Unexpected schema {self.schema_id!r}")
        temporal_shape = (NATIVE_STAGE1_FRAMES, NUM_JOINTS)
        detector_valid = _bool_array(
            "detector_valid", self.detector_valid, temporal_shape
        )
        render_visible = _bool_array(
            "render_visible", self.render_visible, temporal_shape
        )
        if (render_visible & ~detector_valid).any():
            raise ValueError("render_visible must be a subset of detector_valid")
        raw_xy = _masked_xy(
            "raw_xy_norm", self.raw_xy_norm, detector_valid, temporal_shape
        )
        aligned_xy = _masked_xy(
            "aligned_xy_norm", self.aligned_xy_norm, detector_valid, temporal_shape
        )
        raw_score = _score_array("raw_score", self.raw_score, temporal_shape)
        ref_visible = _bool_array("ref_visible", self.ref_visible, (NUM_JOINTS,))
        ref_xy = _masked_xy("ref_xy_norm", self.ref_xy_norm, ref_visible, (NUM_JOINTS,))
        ref_score = _score_array("ref_score", self.ref_score, (NUM_JOINTS,))

        source_frame = np.asarray(self.source_frame)
        if source_frame.shape != (NATIVE_STAGE1_FRAMES,):
            raise ValueError(
                f"source_frame must have shape ({NATIVE_STAGE1_FRAMES},), "
                f"got {source_frame.shape}"
            )
        if source_frame.dtype.kind not in "iu":
            raise ValueError("source_frame must contain integers")
        source_frame_i32 = source_frame.astype(np.int32, copy=False)
        if not np.array_equal(source_frame, source_frame_i32):
            raise ValueError("source_frame values do not fit int32 exactly")
        if (source_frame_i32 < 0).any():
            raise ValueError("source_frame values must be non-negative")
        if (np.diff(source_frame_i32) < 0).any():
            raise ValueError("source_frame values must be non-decreasing")

        return TeacherPoseSequence(
            raw_xy_norm=raw_xy,
            aligned_xy_norm=aligned_xy,
            raw_score=raw_score,
            detector_valid=detector_valid,
            render_visible=render_visible,
            ref_xy_norm=ref_xy,
            ref_score=ref_score,
            ref_visible=ref_visible,
            source_frame=np.ascontiguousarray(source_frame_i32),
            schema_id=self.schema_id,
        )


@dataclass(frozen=True)
class PredictionPoseSequence:
    """Stage-1 normalized coordinates and uncalibrated visibility logits."""

    xy_norm: Any
    visibility_logit: Any
    joint_scope: str = WHOLE62_SCOPE
    schema_id: str = SCHEMA_ID

    def validated(
        self,
        expected_frames: int | None = None,
        expected_scope: str | None = None,
    ) -> "PredictionPoseSequence":
        if self.schema_id != SCHEMA_ID:
            raise ValueError(f"Unexpected schema {self.schema_id!r}")
        joints = _joint_count(self.joint_scope)
        if expected_scope is not None and self.joint_scope != expected_scope:
            raise ValueError(
                f"Expected joint_scope {expected_scope!r}, got {self.joint_scope!r}"
            )
        xy = _prediction_xy(self.xy_norm, joints)
        frames = int(xy.shape[0])
        if expected_frames is not None and frames != expected_frames:
            raise ValueError(f"Expected {expected_frames} frames, got {frames}")
        logits = _visibility_logits(self.visibility_logit, frames, joints)
        return PredictionPoseSequence(xy, logits, self.joint_scope, self.schema_id)

    @property
    def visibility_prob(self) -> Any:
        """Stable sigmoid view for thresholding; logits remain the stored truth."""

        np = _numpy()
        logits = np.asarray(self.visibility_logit, dtype=np.float32)
        probability = np.empty_like(logits)
        nonnegative = logits >= 0
        probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exponential = np.exp(logits[~nonnegative])
        probability[~nonnegative] = exponential / (1.0 + exponential)
        return np.ascontiguousarray(probability)
