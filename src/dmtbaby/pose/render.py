"""Exact upstream rendering adapter and strict UniAnimate input helpers."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator

from .records import CanonicalPoseFrame, PredictionPoseSequence
from .schema import (
    CANONICAL_RENDER_HEIGHT,
    CANONICAL_RENDER_WIDTH,
    LEFT_HAND_OFFSET,
    NUM_BODY_JOINTS,
    NUM_HAND_JOINTS,
    NUM_JOINTS,
    RIGHT_HAND_OFFSET,
    UNIANIMATE_FRAMES,
    UNIANIMATE_HEIGHT,
    UNIANIMATE_WIDTH,
    WHOLE62_SCOPE,
)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NumPy is required for pose rendering") from exc
    return np


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required for exact DWPose JPEG rendering") from exc
    return cv2


def _pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required for UniAnimate pose loading") from exc
    return Image


@contextmanager
def _temporary_sys_path(root: Path) -> Iterator[None]:
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


@lru_cache(maxsize=4)
def load_upstream_draw_pose(unianimate_root: str | Path) -> Callable[..., Any]:
    """Load ``run_align_pose.draw_pose`` without constructing DWPose models."""

    root = Path(unianimate_root).resolve()
    source = root / "run_align_pose.py"
    if not source.is_file():
        raise FileNotFoundError(f"UniAnimate run_align_pose.py not found: {source}")
    module_name = f"_dmtbaby_run_align_pose_{abs(hash(str(root)))}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.draw_pose
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module from {source}")
    module = importlib.util.module_from_spec(spec)
    with _temporary_sys_path(root):
        spec.loader.exec_module(module)
    sys.modules[module_name] = module
    draw_pose = getattr(module, "draw_pose", None)
    if not callable(draw_pose):
        raise RuntimeError(f"{source} does not expose callable draw_pose")
    return draw_pose


def canonical_to_upstream_pose(xy_norm: Any, visible: Any) -> dict[str, Any]:
    """Build the exact dictionary consumed by upstream ``draw_pose``."""

    np = _numpy()
    xy = np.asarray(xy_norm, dtype=np.float32)
    visibility = np.asarray(visible, dtype=np.bool_)
    if xy.shape != (NUM_JOINTS, 2):
        raise ValueError(f"xy_norm must have shape [{NUM_JOINTS},2]")
    if visibility.shape != (NUM_JOINTS,):
        raise ValueError(f"visible must have shape ({NUM_JOINTS},)")
    if np.isinf(xy).any():
        raise ValueError("xy_norm contains Inf")
    finite = np.isfinite(xy)
    if (finite.any(axis=-1) & ~finite.all(axis=-1)).any():
        raise ValueError("xy_norm contains a half-defined point")
    if not finite[visibility].all():
        raise ValueError("visible coordinates must be finite")

    body = xy[:NUM_BODY_JOINTS].copy()
    body[~visibility[:NUM_BODY_JOINTS]] = -1.0
    subset = np.full((1, NUM_BODY_JOINTS), -1.0, dtype=np.float32)
    indices = np.arange(NUM_BODY_JOINTS, dtype=np.float32)
    subset[0, visibility[:NUM_BODY_JOINTS]] = indices[visibility[:NUM_BODY_JOINTS]]

    hands = np.stack(
        [
            xy[LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_HAND_JOINTS],
            xy[RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_HAND_JOINTS],
        ],
        axis=0,
    ).copy()
    hand_visibility = np.stack(
        [
            visibility[LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + NUM_HAND_JOINTS],
            visibility[RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + NUM_HAND_JOINTS],
        ],
        axis=0,
    )
    hands[~hand_visibility] = -1.0
    faces = np.full((1, 68, 2), -1.0, dtype=np.float32)
    return {
        "bodies": {"candidate": body, "subset": subset},
        "hands": hands,
        "faces": faces,
    }


class UpstreamDrawPoseAdapter:
    """Render canonical points through the checked-out upstream implementation."""

    def __init__(
        self,
        unianimate_root: str | Path | None = None,
        draw_pose: Callable[..., Any] | None = None,
    ) -> None:
        if draw_pose is None and unianimate_root is None:
            raise ValueError("Provide unianimate_root or draw_pose")
        self._draw_pose = draw_pose or load_upstream_draw_pose(Path(unianimate_root))

    def render_bgr(self, xy_norm: Any, visible: Any) -> Any:
        np = _numpy()
        pose = canonical_to_upstream_pose(xy_norm, visible)
        without_face, _ = self._draw_pose(
            pose, H=CANONICAL_RENDER_HEIGHT, W=CANONICAL_RENDER_WIDTH
        )
        image = np.asarray(without_face)
        expected = (CANONICAL_RENDER_HEIGHT, CANONICAL_RENDER_WIDTH, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise RuntimeError(
                f"Upstream draw_pose returned {image.shape}/{image.dtype}; expected {expected}/uint8"
            )
        return np.ascontiguousarray(image)


def bgr_to_rgb(image_bgr: Any) -> Any:
    np = _numpy()
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected BGR image [H,W,3], got {image.shape}")
    return np.ascontiguousarray(image[..., ::-1])


def encode_bgr_jpeg(image_bgr: Any, quality: int = 95) -> bytes:
    cv2 = _cv2()
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode DWPose JPEG")
    return encoded.tobytes()


def decode_jpeg_rgb(data: bytes) -> Any:
    np = _numpy()
    Image = _pillow_image()
    with Image.open(BytesIO(data)) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _write_bytes(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_pose_directory(
    output_directory: str | Path,
    sequence: PredictionPoseSequence,
    reference: CanonicalPoseFrame,
    renderer: UpstreamDrawPoseAdapter,
    visibility_threshold: float = 0.5,
    jpeg_quality: int = 95,
) -> Path:
    """Atomically create a strict 81-frame UniAnimate pose directory."""

    np = _numpy()
    item = sequence.validated(
        expected_frames=UNIANIMATE_FRAMES, expected_scope=WHOLE62_SCOPE
    )
    ref = reference.validated()
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing pose directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        visible = (
            item.visibility_prob >= float(visibility_threshold)
        ) & np.isfinite(item.xy_norm).all(axis=-1)
        for frame in range(UNIANIMATE_FRAMES):
            bgr = renderer.render_bgr(item.xy_norm[frame], visible[frame])
            _write_bytes(temporary / f"{frame:04d}.jpg", encode_bgr_jpeg(bgr, jpeg_quality))
        reference_bgr = renderer.render_bgr(ref.xy_norm, ref.render_visible)
        _write_bytes(temporary / "pose.jpg", encode_bgr_jpeg(reference_bgr, jpeg_quality))
        # Do not add a sidecar here: the upstream example attempts to open
        # every directory entry except ``pose.jpg`` as a driving-pose image.
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


@dataclass(frozen=True)
class StrictPoseDirectory:
    frames_rgb: tuple[Any, ...]
    reference_rgb: Any


def load_strict_pose_directory(
    directory: str | Path,
    expected_frames: int = UNIANIMATE_FRAMES,
    expected_width: int = CANONICAL_RENDER_WIDTH,
    expected_height: int = CANONICAL_RENDER_HEIGHT,
) -> StrictPoseDirectory:
    """Load exactly numbered pose JPEGs; never truncate or pad silently."""

    np = _numpy()
    Image = _pillow_image()
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {root}")
    expected = {f"{index:04d}.jpg" for index in range(expected_frames)} | {"pose.jpg"}
    actual = {path.name for path in root.iterdir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Invalid strict pose directory; missing={missing}, extra={extra}")

    def load(path: Path) -> Any:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        expected_shape = (expected_height, expected_width, 3)
        if rgb.shape != expected_shape:
            raise ValueError(f"Pose image {path} has shape {rgb.shape}; expected {expected_shape}")
        return np.ascontiguousarray(rgb)

    frames = tuple(load(root / f"{index:04d}.jpg") for index in range(expected_frames))
    reference = load(root / "pose.jpg")
    return StrictPoseDirectory(frames, reference)


def _resize_rgb_bilinear(image_rgb: Any, width: int, height: int) -> Any:
    np = _numpy()
    Image = _pillow_image()
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    resized = image.resize((int(width), int(height)), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def stage2_numpy_tensors(
    pose_directory: StrictPoseDirectory,
    width: int = UNIANIMATE_WIDTH,
    height: int = UNIANIMATE_HEIGHT,
) -> tuple[Any, Any]:
    """Return motion ``[3,81,H,W]`` and reference ``[H,W,3]`` uint8 arrays."""

    np = _numpy()
    if len(pose_directory.frames_rgb) != UNIANIMATE_FRAMES:
        raise ValueError(f"Expected {UNIANIMATE_FRAMES} frames")
    frames = np.stack(
        [_resize_rgb_bilinear(frame, width, height) for frame in pose_directory.frames_rgb],
        axis=0,
    )
    motion = np.ascontiguousarray(frames.transpose(3, 0, 1, 2))
    reference = _resize_rgb_bilinear(pose_directory.reference_rgb, width, height)
    expected_motion = (3, UNIANIMATE_FRAMES, height, width)
    if motion.shape != expected_motion or motion.dtype != np.uint8:
        raise RuntimeError(f"Unexpected motion tensor {motion.shape}/{motion.dtype}")
    return motion, reference


def stage2_torch_tensors(
    pose_directory: StrictPoseDirectory,
    width: int = UNIANIMATE_WIDTH,
    height: int = UNIANIMATE_HEIGHT,
) -> tuple[Any, Any]:
    """Lazy PyTorch wrapper preserving the upstream uint8 layout and range."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for Stage-2 tensor conversion") from exc
    motion, reference = stage2_numpy_tensors(pose_directory, width=width, height=height)
    return torch.from_numpy(motion), torch.from_numpy(reference)
