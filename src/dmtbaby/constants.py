"""Cross-stage constants which are part of the public artifact contract."""

from __future__ import annotations

SCHEMA_VERSION = "dwpose_woface_62_v1"
ARTIFACT_FORMAT_VERSION = 1
NATIVE_FRAMES = 20
UNIANIMATE_FRAMES = 81
REFERENCE_FRAME_INDEX = 0
RENDER_HEIGHT = 768
RENDER_WIDTH = 512
UNIANIMATE_HEIGHT = 832
UNIANIMATE_WIDTH = 480
LEGACY_81_INDICES = (
    0, 4, 8, 13, 17, 21, 25, 29, 34, 38,
    42, 46, 51, 55, 59, 63, 67, 72, 76, 80,
)
DEFAULT_SEED = 42
DWPOSE_DET_THRESHOLD = 0.3
DWPOSE_POSE_THRESHOLD = 0.3

VAL_PATIENTS = ("pz15", "pz16", "pz18", "pz25", "pz68")
TEST_PATIENTS = ("pz11", "pz17", "pz26", "pz28", "pz73")
