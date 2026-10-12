"""DWPose extraction interfaces with preserved scores and configurable ORT."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .records import CanonicalPoseFrame, TeacherPoseSequence
from .schema import (
    LEFT_HAND_OFFSET,
    NATIVE_STAGE1_FRAMES,
    NUM_BODY_JOINTS,
    NUM_HAND_JOINTS,
    NUM_JOINTS,
    RIGHT_HAND_OFFSET,
    SCHEMA_ID,
)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for DWPose extraction") from exc
    return np


def _onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ONNX Runtime is required for DWPose extraction") from exc
    return ort


def resolve_ort_providers(
    requested: str | Sequence[str] | None = "auto",
    available: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve an explicit, validated ONNX Runtime provider order.

    ``auto`` prefers CUDA with a CPU fallback. Explicit CUDA requests fail when
    CUDAExecutionProvider is unavailable rather than silently changing device.
    """

    if available is None:
        available = tuple(_onnxruntime().get_available_providers())
    available_set = set(available)
    if not available_set:
        raise RuntimeError("ONNX Runtime reports no execution providers")

    if requested is None or requested == "auto":
        if "CUDAExecutionProvider" in available_set:
            order = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available_set:
                order.append("CPUExecutionProvider")
            return tuple(order)
        if "CPUExecutionProvider" in available_set:
            return ("CPUExecutionProvider",)
        return (next(iter(available)),)

    if isinstance(requested, str):
        aliases = {
            "cuda": "CUDAExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }
        providers = [aliases.get(requested.lower(), requested)]
    else:
        providers = list(requested)
    if not providers:
        raise ValueError("At least one ONNX Runtime provider is required")

    unique: list[str] = []
    for provider in providers:
        if provider not in available_set:
            raise RuntimeError(
                f"Requested ONNX Runtime provider {provider!r} is unavailable; "
                f"available providers: {sorted(available_set)}"
            )
        if provider not in unique:
            unique.append(provider)
    return tuple(unique)


def create_ort_session(
    model_path: str | Path,
    providers: str | Sequence[str] | None = "auto",
    provider_options: Sequence[dict[str, Any]] | None = None,
    session_options: Any | None = None,
) -> Any:
    ort = _onnxruntime()
    selected = resolve_ort_providers(providers, ort.get_available_providers())
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    # Explicit CUDA still requires CUDAExecutionProvider first. ORT may still
    # place a few tiny nodes (Slice/Concat) on CPU because it estimates that
    # path is cheaper than a GPU launch; forbidding CPU entirely then rejects
    # otherwise-valid DWPose sessions. Keep CUDA mandatory and first, and only
    # allow CPU as a secondary EP for those preferred nodes.
    require_cuda_first = selected == ("CUDAExecutionProvider",)
    if require_cuda_first and "CPUExecutionProvider" in ort.get_available_providers():
        selected = ("CUDAExecutionProvider", "CPUExecutionProvider")
    kwargs: dict[str, Any] = {"providers": list(selected)}
    if provider_options is not None:
        if len(provider_options) != len(selected):
            raise ValueError("provider_options length must match resolved providers")
        kwargs["provider_options"] = list(provider_options)
    if session_options is not None:
        kwargs["sess_options"] = session_options
    session = ort.InferenceSession(str(path), **kwargs)
    if require_cuda_first and (
        "CUDAExecutionProvider" not in session.get_providers()
        or session.get_providers()[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            f"ONNX model {path} did not initialize with CUDAExecutionProvider first"
        )
    return session


def insert_neck_and_reorder_openpose(
    keypoints: Any,
    scores: Any,
    pose_threshold: float,
) -> tuple[Any, Any]:
    """Convert COCO-WholeBody 133 output to upstream reordered 134 output."""

    np = _numpy()
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if points.ndim != 3 or points.shape[1:] != (133, 2):
        raise ValueError(f"Expected keypoints [N,133,2], got {points.shape}")
    if confidence.shape != points.shape[:2]:
        raise ValueError(f"Expected scores {points.shape[:2]}, got {confidence.shape}")

    neck_xy = np.mean(points[:, [5, 6], :], axis=1)
    neck_score = (
        (confidence[:, 5] > float(pose_threshold))
        & (confidence[:, 6] > float(pose_threshold))
    ).astype(np.float32)
    points = np.insert(points, 17, neck_xy, axis=1)
    confidence = np.insert(confidence, 17, neck_score, axis=1)

    mmpose_indices = np.asarray(
        [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3], dtype=np.int64
    )
    openpose_indices = np.asarray(
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17],
        dtype=np.int64,
    )
    points[:, openpose_indices] = points[:, mmpose_indices]
    confidence[:, openpose_indices] = confidence[:, mmpose_indices]
    return np.ascontiguousarray(points), np.ascontiguousarray(confidence)


def canonicalize_reordered_wholebody(
    keypoints: Any,
    scores: Any,
    image_width: int,
    image_height: int,
    pose_threshold: float = 0.3,
    person_index: int = 0,
    clip_confidence: bool = False,
) -> CanonicalPoseFrame:
    """Reduce one reordered 134-point DWPose person to the 62-point schema."""

    np = _numpy()
    points = np.asarray(keypoints, dtype=np.float32)
    confidence = np.asarray(scores, dtype=np.float32)
    if points.ndim == 2:
        points = points[np.newaxis, ...]
    if confidence.ndim == 1:
        confidence = confidence[np.newaxis, ...]
    if points.ndim != 3 or points.shape[1:] != (134, 2):
        raise ValueError(f"Expected reordered keypoints [N,134,2], got {points.shape}")
    if confidence.shape != points.shape[:2]:
        raise ValueError(f"Expected scores {points.shape[:2]}, got {confidence.shape}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if not 0 <= person_index < points.shape[0]:
        raise IndexError(f"person_index {person_index} outside detected range {points.shape[0]}")

    person_xy = points[person_index]
    person_scores = confidence[person_index]
    if not np.isfinite(person_scores).all():
        raise ValueError("Raw DWPose confidence contains NaN or Inf")
    if (person_scores < 0.0).any() or (person_scores > 1.0).any():
        if not clip_confidence:
            raise ValueError("Raw DWPose confidence must lie in [0,1]")
        person_scores = np.clip(person_scores, 0.0, 1.0)
    xy = np.full((NUM_JOINTS, 2), np.nan, dtype=np.float32)
    raw_score = np.zeros((NUM_JOINTS,), dtype=np.float32)

    xy[:18] = person_xy[:18]
    raw_score[:18] = person_scores[:18]

    # Upstream requires both toe points and averages them. Heel points 20/23
    # are intentionally omitted from the no-face renderer.
    foot_specs = ((18, (18, 19)), (19, (21, 22)))
    for target, source in foot_specs:
        source_points = person_xy[list(source)]
        if np.isfinite(source_points).all():
            xy[target] = source_points.mean(axis=0)
        raw_score[target] = float(np.min(person_scores[list(source)]))

    left_hand = person_xy[92:113]
    right_hand = person_xy[113:134]
    xy[LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_HAND_JOINTS] = left_hand
    xy[RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_HAND_JOINTS] = right_hand
    raw_score[LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_HAND_JOINTS] = person_scores[
        92:113
    ]
    raw_score[RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_HAND_JOINTS] = person_scores[
        113:134
    ]

    xy[:, 0] /= float(image_width)
    xy[:, 1] /= float(image_height)
    detector_valid = np.isfinite(xy).all(axis=1)
    xy[~detector_valid] = np.nan
    visible = (raw_score > float(pose_threshold)) & detector_valid
    return CanonicalPoseFrame(
        xy_norm=xy,
        raw_score=raw_score,
        detector_valid=detector_valid,
        render_visible=visible,
        schema_id=SCHEMA_ID,
    ).validated()


@contextmanager
def _temporary_import_root(root: Path | None) -> Iterator[None]:
    if root is None:
        yield
        return
    value = str(root.resolve())
    inserted = value not in sys.path
    if inserted:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:  # pragma: no cover - defensive
                pass


@dataclass(frozen=True)
class OrtModels:
    detector: Path
    pose: Path


class OrtWholebodyExtractor:
    """Minimal configurable-session equivalent of upstream ``Wholebody``.

    This class only performs frame/sequence extraction. It does not enumerate or
    mutate a dataset.
    """

    def __init__(
        self,
        models: OrtModels,
        providers: str | Sequence[str] | None = "auto",
        det_threshold: float = 0.3,
        pose_threshold: float = 0.3,
        external_root: str | Path | None = None,
        provider_options: Sequence[dict[str, Any]] | None = None,
        clip_confidence: bool = False,
    ) -> None:
        self.det_threshold = float(det_threshold)
        self.pose_threshold = float(pose_threshold)
        self.clip_confidence = bool(clip_confidence)
        self.external_root = Path(external_root).resolve() if external_root else None
        selected = resolve_ort_providers(providers)
        self.providers = selected
        self.det_session = create_ort_session(
            models.detector, selected, provider_options=provider_options
        )
        self.pose_session = create_ort_session(
            models.pose, selected, provider_options=provider_options
        )
        with _temporary_import_root(self.external_root):
            try:
                self._inference_detector = importlib.import_module(
                    "dwpose.onnxdet"
                ).inference_detector
                self._inference_pose = importlib.import_module("dwpose.onnxpose").inference_pose
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "Could not import upstream dwpose.onnxdet/onnxpose; pass external_root"
                ) from exc

    def infer_reordered(self, image_bgr: Any) -> tuple[Any, Any]:
        np = _numpy()
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image [H,W,3], got {image.shape}")
        boxes = self._inference_detector(
            self.det_session, image, det_thresh=self.det_threshold
        )
        keypoints, scores = self._inference_pose(self.pose_session, boxes, image)
        return insert_neck_and_reorder_openpose(keypoints, scores, self.pose_threshold)

    def extract_frame(self, image_bgr: Any, person_index: int = 0) -> CanonicalPoseFrame:
        points, scores = self.infer_reordered(image_bgr)
        height, width = image_bgr.shape[:2]
        return canonicalize_reordered_wholebody(
            points,
            scores,
            image_width=width,
            image_height=height,
            pose_threshold=self.pose_threshold,
            person_index=person_index,
            clip_confidence=self.clip_confidence,
        )

    def extract_frames(
        self,
        images_bgr: Iterable[Any],
        person_index: int = 0,
    ) -> tuple[CanonicalPoseFrame, ...]:
        """Extract caller-supplied frames without enumerating a dataset/video."""

        frames = tuple(
            self.extract_frame(image, person_index=person_index) for image in images_bgr
        )
        if not frames:
            raise ValueError("Cannot extract an empty pose sequence")
        return frames


def build_teacher_sequence(
    raw_frames: Sequence[CanonicalPoseFrame],
    aligned_xy_norm: Any,
    render_visible: Any,
    reference: CanonicalPoseFrame,
    source_frame: Any,
) -> TeacherPoseSequence:
    """Construct the exact teacher artifact after caller-owned alignment.

    This boundary intentionally does not perform alignment: the caller supplies
    deterministic post-alignment coordinates and their final render mask.  That
    keeps extraction reusable while allowing the dataset runner to own sampling,
    alignment failures, and quarantine policy.
    """

    np = _numpy()
    frames = tuple(frame.validated() for frame in raw_frames)
    reference_item = reference.validated()
    if len(frames) != NATIVE_STAGE1_FRAMES:
        raise ValueError(
            f"Teacher artifacts require exactly {NATIVE_STAGE1_FRAMES} frames, "
            f"got {len(frames)}"
        )

    ref_xy = reference_item.xy_norm.copy()
    ref_xy[~reference_item.render_visible] = np.nan
    return TeacherPoseSequence(
        raw_xy_norm=np.stack([frame.xy_norm for frame in frames]),
        aligned_xy_norm=aligned_xy_norm,
        raw_score=np.stack([frame.raw_score for frame in frames]),
        detector_valid=np.stack([frame.detector_valid for frame in frames]),
        render_visible=render_visible,
        ref_xy_norm=ref_xy,
        ref_score=reference_item.raw_score,
        ref_visible=reference_item.render_visible,
        source_frame=source_frame,
    ).validated()
