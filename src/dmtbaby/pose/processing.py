"""Temporal and anatomical processing for native-20 Stage-1 predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .records import PredictionPoseSequence
from .schema import (
    ANATOMICAL_EDGES,
    BODY20_SCOPE,
    CANONICAL_RENDER_HEIGHT,
    CANONICAL_RENDER_WIDTH,
    HAND_WRIST_ATTACHMENTS,
    LEFT_HAND_OFFSET,
    NATIVE_STAGE1_FRAMES,
    NUM_BODY_JOINTS,
    NUM_HAND_JOINTS,
    NUM_JOINTS,
    RIGHT_HAND_OFFSET,
    UNIANIMATE_FRAMES,
    WHOLE62_SCOPE,
)


NECK_INDEX = 1
_FORCED_VISIBLE_LOGIT = 20.0
_FORCED_INVISIBLE_LOGIT = -20.0


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for pose processing") from exc
    return np


def _probability_to_logit(probability: Any) -> Any:
    np = _numpy()
    value = np.asarray(probability, dtype=np.float32)
    epsilon = np.finfo(np.float32).eps
    value = np.clip(value, epsilon, 1.0 - epsilon)
    return np.ascontiguousarray(np.log(value) - np.log1p(-value), dtype=np.float32)


def _reference_arrays(reference_xy: Any, reference_visibility: Any) -> tuple[Any, Any]:
    np = _numpy()
    xy = np.asarray(reference_xy, dtype=np.float32)
    visible = np.asarray(reference_visibility)
    if xy.shape != (NUM_JOINTS, 2):
        raise ValueError(f"reference_xy must have shape ({NUM_JOINTS},2)")
    if visible.shape != (NUM_JOINTS,):
        raise ValueError(f"reference_visibility must have shape ({NUM_JOINTS},)")
    if visible.dtype != np.bool_ and not np.isin(visible, (0, 1)).all():
        raise ValueError("reference_visibility must be boolean")
    visible = visible.astype(np.bool_, copy=False)
    if np.isinf(xy).any():
        raise ValueError("reference_xy contains Inf")
    finite = np.isfinite(xy)
    if (finite.any(axis=-1) & ~finite.all(axis=-1)).any():
        raise ValueError("reference_xy contains a half-defined point")
    if not finite[visible].all():
        raise ValueError("reference_xy must be finite where reference_visibility is true")
    if not np.isnan(xy[~visible]).all():
        raise ValueError("reference_xy must be NaN where reference_visibility is false")
    if ((xy[visible] < 0.0) | (xy[visible] > 1.0)).any():
        raise ValueError("visible reference coordinates must lie in [0,1]")
    return np.ascontiguousarray(xy), np.ascontiguousarray(visible)


@dataclass(frozen=True)
class TemporalProcessingConfig:
    visibility_threshold: float = 0.5
    max_gap: int = 4
    savgol_window: int = 5
    savgol_poly: int = 2
    body_outlier_threshold: float = 0.12
    hand_outlier_threshold: float = 0.08
    outlier_window: int = 2


@dataclass(frozen=True)
class AnatomyLBFGSConfig:
    enabled: bool = True
    bone_weight: float = 0.25
    acceleration_weight: float = 0.05
    wrist_attachment_weight: float = 1.0
    reference_anchor_weight: float = 10.0
    max_iterations: int = 20
    history_size: int = 10
    ftol: float = 1e-9
    gtol: float = 1e-6
    require_success: bool = False


@dataclass(frozen=True)
class AnatomyDiagnostics:
    attempted: bool
    success: bool
    status: int
    iterations: int
    final_loss: float
    message: str


@dataclass(frozen=True)
class ProcessingDiagnostics:
    rejected_outliers: int
    interpolated_points: int
    anatomy: AnatomyDiagnostics


def expand_prediction_to_whole62(
    sequence: PredictionPoseSequence,
) -> PredictionPoseSequence:
    """Pad a body20 milestone prediction with explicitly invisible hands."""

    np = _numpy()
    item = sequence.validated()
    if item.joint_scope == WHOLE62_SCOPE:
        return item
    if item.joint_scope != BODY20_SCOPE:  # pragma: no cover - record validation guard
        raise ValueError(f"Unsupported joint_scope {item.joint_scope!r}")
    frames = item.xy_norm.shape[0]
    xy = np.full((frames, NUM_JOINTS, 2), np.nan, dtype=np.float32)
    logits = np.full(
        (frames, NUM_JOINTS), _FORCED_INVISIBLE_LOGIT, dtype=np.float32
    )
    xy[:, :NUM_BODY_JOINTS] = item.xy_norm
    logits[:, :NUM_BODY_JOINTS] = item.visibility_logit
    return PredictionPoseSequence(xy, logits, joint_scope=WHOLE62_SCOPE).validated()


def resample_native20_to81(sequence: PredictionPoseSequence) -> PredictionPoseSequence:
    """Linearly resample normalized coordinates and logits from 20 to 81."""

    np = _numpy()
    source = expand_prediction_to_whole62(sequence).validated(
        expected_frames=NATIVE_STAGE1_FRAMES,
        expected_scope=WHOLE62_SCOPE,
    )
    source_times = np.linspace(0.0, 1.0, NATIVE_STAGE1_FRAMES, dtype=np.float32)
    target_times = np.linspace(0.0, 1.0, UNIANIMATE_FRAMES, dtype=np.float32)
    xy = np.full((UNIANIMATE_FRAMES, NUM_JOINTS, 2), np.nan, dtype=np.float32)
    logits = np.empty((UNIANIMATE_FRAMES, NUM_JOINTS), dtype=np.float32)

    for joint in range(NUM_JOINTS):
        logits[:, joint] = np.interp(
            target_times, source_times, source.visibility_logit[:, joint]
        )
        valid = np.isfinite(source.xy_norm[:, joint]).all(axis=-1)
        count = int(valid.sum())
        if count == 0:
            continue
        if count == 1:
            xy[:, joint] = source.xy_norm[valid, joint][0]
            continue
        for coordinate in range(2):
            xy[:, joint, coordinate] = np.interp(
                target_times,
                source_times[valid],
                source.xy_norm[valid, joint, coordinate],
            )
    return PredictionPoseSequence(xy, logits).validated(
        expected_frames=UNIANIMATE_FRAMES
    )


def _reject_outliers(
    series: Any,
    valid: Any,
    threshold: float,
    radius: int,
) -> tuple[Any, Any, int]:
    np = _numpy()
    output = series.copy()
    output_valid = valid.copy()
    rejected = np.zeros((len(series),), dtype=np.bool_)
    if threshold <= 0 or radius <= 0 or int(valid.sum()) < 3:
        return output, output_valid, 0

    for frame in range(len(series)):
        if not output_valid[frame]:
            continue
        start = max(0, frame - radius)
        end = min(len(series), frame + radius + 1)
        neighbor_valid = output_valid[start:end].copy()
        neighbor_valid[frame - start] = False
        if int(neighbor_valid.sum()) < 2:
            continue
        neighbors = output[start:end][neighbor_valid]
        median = np.median(neighbors, axis=0)
        residual = float(np.linalg.norm(output[frame] - median))
        if residual <= threshold:
            continue

        previous = np.flatnonzero(output_valid[start:frame]) + start
        following = np.flatnonzero(output_valid[frame + 1 : end]) + frame + 1
        if len(previous) and len(following):
            left, right = int(previous[-1]), int(following[0])
            alpha = (frame - left) / float(max(right - left, 1))
            expected = (1.0 - alpha) * output[left] + alpha * output[right]
            interpolation_residual = float(np.linalg.norm(output[frame] - expected))
            jump_left = float(np.linalg.norm(output[frame] - output[left]))
            jump_right = float(np.linalg.norm(output[frame] - output[right]))
            bridge = float(np.linalg.norm(output[right] - output[left]))
            if (
                interpolation_residual > threshold
                and jump_left > threshold
                and jump_right > threshold
                and bridge <= threshold * 1.5
            ):
                rejected[frame] = True
        elif residual > threshold * 1.5 and int(neighbor_valid.sum()) >= 3:
            rejected[frame] = True

    output_valid[rejected] = False
    output[rejected] = np.nan
    return output, output_valid, int(rejected.sum())


def _fill_short_gaps(
    series: Any,
    probability: Any,
    valid: Any,
    max_gap: int,
    threshold: float,
) -> tuple[Any, Any, Any, int]:
    np = _numpy()
    output = series.copy()
    probabilities = probability.copy()
    output_valid = valid.copy()
    interpolated = 0
    frame = 0
    while frame < len(series):
        if output_valid[frame]:
            frame += 1
            continue
        start = frame
        while frame < len(series) and not output_valid[frame]:
            frame += 1
        end = frame - 1
        length = end - start + 1
        if length > max_gap:
            continue
        left = start - 1 if start > 0 and output_valid[start - 1] else None
        right = frame if frame < len(series) and output_valid[frame] else None
        # Interpolation requires a valid sample on both sides.  Replicating a
        # boundary value would fabricate a trajectory outside the observed
        # interval and is not the locked "interpolate short runs" behavior.
        if left is None or right is None:
            continue
        target = np.arange(start, end + 1)
        for coordinate in range(2):
            output[start : end + 1, coordinate] = np.interp(
                target,
                [left, right],
                [output[left, coordinate], output[right, coordinate]],
            )
        confidence = min(
            float(probabilities[left]), float(probabilities[right]), threshold
        )
        probabilities[start : end + 1] = confidence
        output_valid[start : end + 1] = True
        interpolated += length
    return output, probabilities, output_valid, interpolated


def _smooth_values(values: Any, window: int, poly: int) -> Any:
    np = _numpy()
    if len(values) < 3:
        return values
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3 or window <= poly:
        return values
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(values, window_length=window, polyorder=poly, mode="interp")
    except ImportError:  # pragma: no cover - SciPy-free runtime fallback
        padding = window // 2
        padded = np.pad(values, (padding, padding), mode="edge")
        kernel = np.ones((window,), dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid")


def clean_temporal_pose(
    sequence: PredictionPoseSequence,
    config: TemporalProcessingConfig = TemporalProcessingConfig(),
) -> tuple[PredictionPoseSequence, int, int]:
    """Reject spikes, fill short gaps, smooth, and clamp only active points."""

    np = _numpy()
    item = sequence.validated(
        expected_frames=UNIANIMATE_FRAMES, expected_scope=WHOLE62_SCOPE
    )
    xy = item.xy_norm.copy()
    probability = item.visibility_prob.copy()
    finite = np.isfinite(xy).all(axis=-1)
    valid = (probability >= config.visibility_threshold) & finite
    rejected_total = 0
    interpolated_total = 0

    for joint in range(NUM_JOINTS):
        threshold = (
            config.body_outlier_threshold
            if joint < NUM_BODY_JOINTS
            else config.hand_outlier_threshold
        )
        series, joint_valid, rejected = _reject_outliers(
            xy[:, joint], valid[:, joint], threshold, config.outlier_window
        )
        rejected_total += rejected
        probability[~joint_valid, joint] = np.minimum(
            probability[~joint_valid, joint],
            np.nextafter(
                np.float32(config.visibility_threshold), np.float32(0.0)
            ),
        )
        series, joint_probability, joint_valid, interpolated = _fill_short_gaps(
            series,
            probability[:, joint],
            joint_valid,
            config.max_gap,
            config.visibility_threshold,
        )
        interpolated_total += interpolated

        frame = 0
        while frame < len(series):
            if not joint_valid[frame]:
                frame += 1
                continue
            start = frame
            while frame < len(series) and joint_valid[frame]:
                frame += 1
            end = frame
            if end - start >= 3:
                for coordinate in range(2):
                    series[start:end, coordinate] = _smooth_values(
                        series[start:end, coordinate],
                        config.savgol_window,
                        config.savgol_poly,
                    )
        series[joint_valid] = np.clip(series[joint_valid], 0.0, 1.0)
        series[~joint_valid] = np.nan
        xy[:, joint] = series
        probability[:, joint] = joint_probability
        valid[:, joint] = joint_valid

    output = PredictionPoseSequence(xy, _probability_to_logit(probability))
    return output.validated(expected_frames=UNIANIMATE_FRAMES), rejected_total, interpolated_total


def _anchor_reference(
    sequence: PredictionPoseSequence,
    reference_xy: Any,
    reference_visibility: Any,
) -> PredictionPoseSequence:
    np = _numpy()
    item = sequence.validated(expected_scope=WHOLE62_SCOPE)
    ref_xy, ref_visible = _reference_arrays(reference_xy, reference_visibility)
    xy = item.xy_norm.copy()
    logits = item.visibility_logit.copy()
    xy[0] = np.nan
    xy[0, ref_visible] = ref_xy[ref_visible]
    logits[0, ref_visible] = _FORCED_VISIBLE_LOGIT
    logits[0, ~ref_visible] = _FORCED_INVISIBLE_LOGIT
    return PredictionPoseSequence(xy, logits).validated()


def refine_anatomy_lbfgs(
    sequence: PredictionPoseSequence,
    reference_xy: Any,
    reference_visibility: Any,
    temporal_config: TemporalProcessingConfig = TemporalProcessingConfig(),
    config: AnatomyLBFGSConfig = AnatomyLBFGSConfig(),
) -> tuple[PredictionPoseSequence, AnatomyDiagnostics]:
    """Refine relative bone lengths while preserving a moving neck trajectory."""

    np = _numpy()
    item = sequence.validated(
        expected_frames=UNIANIMATE_FRAMES, expected_scope=WHOLE62_SCOPE
    )
    ref_xy, ref_visible = _reference_arrays(reference_xy, reference_visibility)
    if not config.enabled:
        return item, AnatomyDiagnostics(False, True, 0, 0, 0.0, "disabled")
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SciPy is required for anatomical L-BFGS refinement") from exc

    visible = (
        (item.visibility_prob >= temporal_config.visibility_threshold)
        & np.isfinite(item.xy_norm).all(axis=-1)
    )
    if not visible.any():
        return item, AnatomyDiagnostics(
            False, True, 0, 0, 0.0, "no visible coordinates"
        )

    diagonal = float(
        (CANONICAL_RENDER_WIDTH**2 + CANONICAL_RENDER_HEIGHT**2) ** 0.5
    )
    scale = np.asarray(
        [CANONICAL_RENDER_WIDTH / diagonal, CANONICAL_RENDER_HEIGHT / diagonal],
        dtype=np.float64,
    )
    original = np.zeros_like(item.xy_norm, dtype=np.float64)
    original[visible] = np.asarray(item.xy_norm, dtype=np.float64)[visible] * scale
    reference = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    reference[ref_visible] = np.asarray(ref_xy, dtype=np.float64)[ref_visible] * scale
    data_weight = np.where(
        visible, np.clip(item.visibility_prob, 0.05, 1.0), 0.0
    ).astype(np.float64)

    bones: list[tuple[int, int, float]] = []
    for start, end in ANATOMICAL_EDGES:
        if ref_visible[start] and ref_visible[end]:
            length = float(np.linalg.norm(reference[start] - reference[end]))
            if length > 1e-8:
                bones.append((start, end, length))
    # Wrist coincidence is a topology constraint, not a reference-bone
    # length. It remains available whenever both predicted joints are visible,
    # even if the reference hand root itself was not detected.
    attachments = list(HAND_WRIST_ATTACHMENTS)

    active_indices = np.flatnonzero(visible.reshape(-1))

    def unpack(active: Any) -> Any:
        points = original.copy().reshape(-1, 2)
        points[active_indices] = active.reshape(-1, 2)
        return points.reshape(original.shape)

    def objective(active: Any) -> tuple[float, Any]:
        points = unpack(active)
        gradient = np.zeros_like(points)
        displacement = points - original
        gradient += 2.0 * data_weight[..., None] * displacement
        loss = float(np.sum(data_weight[..., None] * displacement**2))

        # A ratio to the reference length, rather than an absolute-pixel error,
        # gives short hand bones and long body bones comparable meaning.
        for start, end, reference_length in bones:
            active_edge = visible[:, start] & visible[:, end]
            if not active_edge.any():
                continue
            delta = points[:, start] - points[:, end]
            distance = np.linalg.norm(delta, axis=1)
            safe_distance = np.maximum(distance, 1e-8)
            weight = (
                np.minimum(data_weight[:, start], data_weight[:, end]) * active_edge
            )
            residual = distance / reference_length - 1.0
            loss += config.bone_weight * float(np.sum(weight * residual**2))
            edge_gradient = (
                2.0
                * config.bone_weight
                * weight
                * residual
                / (reference_length * safe_distance)
            )[:, None] * delta
            gradient[:, start] += edge_gradient
            gradient[:, end] -= edge_gradient

        for start, end in attachments:
            active_edge = visible[:, start] & visible[:, end]
            if not active_edge.any():
                continue
            delta = points[:, start] - points[:, end]
            weight = (
                np.minimum(data_weight[:, start], data_weight[:, end]) * active_edge
            )
            loss += config.wrist_attachment_weight * float(
                np.sum(weight[:, None] * delta**2)
            )
            attachment_gradient = (
                2.0 * config.wrist_attachment_weight * weight[:, None] * delta
            )
            gradient[:, start] += attachment_gradient
            gradient[:, end] -= attachment_gradient

        if len(points) >= 3 and config.acceleration_weight > 0:
            active_triplet = visible[:-2] & visible[1:-1] & visible[2:]
            weight = np.minimum(
                np.minimum(data_weight[:-2], data_weight[1:-1]), data_weight[2:]
            ) * active_triplet
            acceleration = points[2:] - 2.0 * points[1:-1] + points[:-2]
            weighted = weight[..., None] * acceleration
            loss += config.acceleration_weight * float(np.sum(weighted * acceleration))
            gradient[:-2] += 2.0 * config.acceleration_weight * weighted
            gradient[1:-1] -= 4.0 * config.acceleration_weight * weighted
            gradient[2:] += 2.0 * config.acceleration_weight * weighted

        frame_zero_anchor = visible[0] & ref_visible
        if frame_zero_anchor.any() and config.reference_anchor_weight > 0:
            delta = points[0, frame_zero_anchor] - reference[frame_zero_anchor]
            loss += config.reference_anchor_weight * float(np.sum(delta**2))
            gradient[0, frame_zero_anchor] += (
                2.0 * config.reference_anchor_weight * delta
            )
        active_gradient = gradient.reshape(-1, 2)[active_indices]
        return loss, active_gradient.ravel()

    initial = original.reshape(-1, 2)[active_indices].ravel()
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(config.max_iterations),
            "maxcor": int(config.history_size),
            "ftol": float(config.ftol),
            "gtol": float(config.gtol),
            "maxls": 20,
        },
    )
    if config.require_success and not result.success:
        raise RuntimeError(f"Anatomical L-BFGS failed: {result.message}")
    refined_scaled = unpack(np.asarray(result.x, dtype=np.float64))
    if not np.isfinite(refined_scaled[visible]).all():
        raise RuntimeError("Anatomical L-BFGS produced NaN or Inf")
    refined = refined_scaled / scale

    # Preserve the predicted root trajectory per frame. Translating all active
    # joints together leaves every refined relative bone length unchanged.
    for frame in range(UNIANIMATE_FRAMES):
        if visible[frame, NECK_INDEX]:
            root_delta = item.xy_norm[frame, NECK_INDEX] - refined[frame, NECK_INDEX]
            refined[frame, visible[frame]] += root_delta

    # The optimizer term brings the independently detected hand root close to
    # its body wrist. The bridge contract is exact on generated frames:
    # translate the complete visible hand so the two roots coincide while its
    # internal bone geometry is preserved. Frame zero is restored verbatim
    # from the observed detector output below, even if the two detectors differ.
    hand_ranges = (
        (
            HAND_WRIST_ATTACHMENTS[0],
            range(LEFT_HAND_OFFSET, LEFT_HAND_OFFSET + NUM_HAND_JOINTS),
        ),
        (
            HAND_WRIST_ATTACHMENTS[1],
            range(RIGHT_HAND_OFFSET, RIGHT_HAND_OFFSET + NUM_HAND_JOINTS),
        ),
    )
    for (body_wrist, hand_root), hand_indices in hand_ranges:
        for frame in range(1, UNIANIMATE_FRAMES):
            if not (visible[frame, body_wrist] and visible[frame, hand_root]):
                continue
            delta = refined[frame, body_wrist] - refined[frame, hand_root]
            active_hand = [index for index in hand_indices if visible[frame, index]]
            refined[frame, active_hand] += delta

    refined[visible] = np.clip(refined[visible], 0.0, 1.0)
    refined[~visible] = np.nan
    # The structural condition begins at the observed reference, exactly—not
    # merely close to it under an optimizer penalty.
    refined[0] = np.nan
    refined[0, ref_visible] = ref_xy[ref_visible]
    logits = item.visibility_logit.copy()
    logits[0, ref_visible] = _FORCED_VISIBLE_LOGIT
    logits[0, ~ref_visible] = _FORCED_INVISIBLE_LOGIT
    output = PredictionPoseSequence(refined.astype(np.float32), logits).validated(
        expected_frames=UNIANIMATE_FRAMES
    )
    diagnostics = AnatomyDiagnostics(
        attempted=True,
        success=bool(result.success),
        status=int(result.status),
        iterations=int(getattr(result, "nit", 0)),
        final_loss=float(result.fun),
        message=str(result.message),
    )
    return output, diagnostics


def process_native20_prediction(
    sequence: PredictionPoseSequence,
    reference_xy: Any,
    reference_visibility: Any,
    temporal_config: TemporalProcessingConfig = TemporalProcessingConfig(),
    anatomy_config: AnatomyLBFGSConfig = AnatomyLBFGSConfig(),
) -> tuple[PredictionPoseSequence, ProcessingDiagnostics]:
    """Execute the locked anchor-native20 -> interpolate81 -> clean -> anatomy order."""

    # Anchoring before interpolation is essential: frames 1..4 must transition
    # from the observed reference, not from an arbitrary predicted frame zero.
    source = sequence.validated(expected_frames=NATIVE_STAGE1_FRAMES)
    bridge_reference_xy, bridge_reference_visible = _reference_arrays(
        reference_xy, reference_visibility
    )
    if source.joint_scope == BODY20_SCOPE:
        bridge_reference_xy = bridge_reference_xy.copy()
        bridge_reference_visible = bridge_reference_visible.copy()
        bridge_reference_xy[NUM_BODY_JOINTS:] = float("nan")
        bridge_reference_visible[NUM_BODY_JOINTS:] = False

    native = expand_prediction_to_whole62(source)
    native = _anchor_reference(
        native, bridge_reference_xy, bridge_reference_visible
    )
    output = resample_native20_to81(native)
    output, rejected, interpolated = clean_temporal_pose(output, temporal_config)
    output = _anchor_reference(
        output, bridge_reference_xy, bridge_reference_visible
    )
    output, anatomy = refine_anatomy_lbfgs(
        output,
        bridge_reference_xy,
        bridge_reference_visible,
        temporal_config=temporal_config,
        config=anatomy_config,
    )
    output = _anchor_reference(
        output, bridge_reference_xy, bridge_reference_visible
    )
    return output, ProcessingDiagnostics(rejected, interpolated, anatomy)
