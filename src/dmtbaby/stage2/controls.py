"""Validation for rendered DWPose controls passed to UniAnimate.

The main cascade contract is intentionally stricter than the historical
``pose_dir`` input: it is always an ordered sequence of exactly 81 RGB JPGs
named ``0000.jpg`` through ``0080.jpg`` plus a separate reference-pose image.
No truncation, padding, or last-frame repetition is permitted at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CASCADE_FRAME_COUNT = 81
CASCADE_FRAME_NAMES = tuple(f"{index:04d}.jpg" for index in range(CASCADE_FRAME_COUNT))


class ControlSequenceError(ValueError):
    """Raised when a rendered control sequence violates the cascade schema."""


@dataclass(frozen=True)
class ControlSequence:
    """Validated paths for one UniAnimate control sequence."""

    directory: Path
    reference_pose: Path
    frames: tuple[Path, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != CASCADE_FRAME_COUNT:
            raise ControlSequenceError(
                f"A cascade ControlSequence must contain exactly {CASCADE_FRAME_COUNT} frames; "
                f"got {len(self.frames)}."
            )


def _preview(names: Iterable[str], limit: int = 6) -> str:
    values = list(names)
    if not values:
        return "none"
    shown = ", ".join(values[:limit])
    return shown if len(values) <= limit else f"{shown}, ... ({len(values)} total)"


def validate_control_sequence(
    control_dir: str | Path,
    reference_pose: str | Path | None = None,
) -> ControlSequence:
    """Validate and return the exact 81-frame cascade control artifact.

    Only ``0000.jpg`` ... ``0080.jpg`` are accepted as temporal controls.
    ``reference_pose`` defaults to ``control_dir/pose.jpg`` for compatibility
    with UniAnimate's upstream examples, but is modeled as a separate artifact.
    """

    directory = Path(control_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Control directory does not exist: {directory}")

    ref_path = (
        Path(reference_pose).expanduser().resolve()
        if reference_pose is not None
        else (directory / "pose.jpg").resolve()
    )
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference-pose image does not exist: {ref_path}")

    expected = set(CASCADE_FRAME_NAMES)
    # ``pose.jpg`` is the conventional separate reference artifact and may
    # coexist with an explicitly selected external reference pose.
    allowed_entries = expected | {"pose.jpg"}
    entries = {path.name for path in directory.iterdir()}
    forbidden_entries = sorted(entries - allowed_entries)
    actual = entries & expected
    missing = sorted(expected - actual)
    if missing or forbidden_entries or len(actual) != CASCADE_FRAME_COUNT:
        details = [
            f"Cascade controls must be exactly {CASCADE_FRAME_COUNT} frames named "
            "0000.jpg through 0080.jpg.",
            f"Found {len(actual)} temporal JPG/JPEG files in {directory}.",
        ]
        if missing:
            details.append(f"Missing: {_preview(missing)}.")
        if forbidden_entries:
            details.append(f"Unexpected: {_preview(forbidden_entries)}.")
        raise ControlSequenceError(" ".join(details))

    frames = tuple(directory / name for name in CASCADE_FRAME_NAMES)
    return ControlSequence(directory=directory, reference_pose=ref_path, frames=frames)


def legacy_pose_frame_paths(pose_dir: str | Path, target_frames: int) -> tuple[Path, ...]:
    """Return the historical cap/repeat sequence used only by ``--pose_dir``.

    This function deliberately lives beside the strict schema so callers must
    make the legacy behavior explicit instead of silently padding cascade data.
    """

    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    directory = Path(pose_dir).expanduser().resolve()
    frame_paths = sorted(
        path
        for path in directory.glob("*.jpg")
        if path.name != "pose.jpg"
    )
    if not frame_paths:
        raise FileNotFoundError(f"No DWPose frame JPGs found in {directory}")
    selected = frame_paths[:target_frames]
    selected.extend([selected[-1]] * (target_frames - len(selected)))
    return tuple(selected)
