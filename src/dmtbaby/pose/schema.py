"""Versioned semantic schema and topology for UniAnimate's no-face DWPose map."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_ID = "dwpose_woface_62_v1"
SCHEMA_VERSION = 1
BODY20_SCOPE = "body20"
WHOLE62_SCOPE = "whole62"

NATIVE_STAGE1_FRAMES = 20
UNIANIMATE_FRAMES = 81
CANONICAL_RENDER_WIDTH = 512
CANONICAL_RENDER_HEIGHT = 768
UNIANIMATE_WIDTH = 480
UNIANIMATE_HEIGHT = 832

BODY_JOINT_NAMES = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    # The upstream debug-name list reverses these two labels. The renderer's
    # ankle connections establish the canonical laterality used here.
    "left_foot_center",
    "right_foot_center",
)

HAND_JOINT_NAMES = (
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
)

JOINT_NAMES = (
    *BODY_JOINT_NAMES,
    *(f"left_hand_{name}" for name in HAND_JOINT_NAMES),
    *(f"right_hand_{name}" for name in HAND_JOINT_NAMES),
)

NUM_BODY_JOINTS = len(BODY_JOINT_NAMES)
NUM_HAND_JOINTS = len(HAND_JOINT_NAMES)
NUM_JOINTS = len(JOINT_NAMES)
LEFT_HAND_OFFSET = NUM_BODY_JOINTS
RIGHT_HAND_OFFSET = NUM_BODY_JOINTS + NUM_HAND_JOINTS

# Exact zero-based translation of dwpose.util.draw_body_and_foot.limbSeq.
BODY_RENDER_EDGES = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (13, 18),
    (10, 19),
)

# Exact zero-based translation of dwpose.util.draw_handpose.edges.
HAND_RENDER_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

LEFT_HAND_RENDER_EDGES = tuple(
    (LEFT_HAND_OFFSET + start, LEFT_HAND_OFFSET + end)
    for start, end in HAND_RENDER_EDGES
)
RIGHT_HAND_RENDER_EDGES = tuple(
    (RIGHT_HAND_OFFSET + start, RIGHT_HAND_OFFSET + end)
    for start, end in HAND_RENDER_EDGES
)
ANATOMICAL_EDGES = BODY_RENDER_EDGES + LEFT_HAND_RENDER_EDGES + RIGHT_HAND_RENDER_EDGES

# The body wrist and corresponding OpenPose-hand wrist describe the same
# anatomical point but are independently detected upstream.
HAND_WRIST_ATTACHMENTS = (
    (7, LEFT_HAND_OFFSET),
    (4, RIGHT_HAND_OFFSET),
)

# Exact scalar order used by OpenCV in the upstream renderer (BGR on disk).
BODY_BGR_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
    (170, 255, 255),
    (255, 255, 0),
)


@dataclass(frozen=True)
class PoseSchema:
    schema_id: str
    version: int
    joint_names: tuple[str, ...]
    body_joint_count: int
    hand_joint_count: int
    native_frames: int
    target_frames: int

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)


DWPOSE_WOFACE_62 = PoseSchema(
    schema_id=SCHEMA_ID,
    version=SCHEMA_VERSION,
    joint_names=JOINT_NAMES,
    body_joint_count=NUM_BODY_JOINTS,
    hand_joint_count=NUM_HAND_JOINTS,
    native_frames=NATIVE_STAGE1_FRAMES,
    target_frames=UNIANIMATE_FRAMES,
)


def require_schema(schema_id: str) -> PoseSchema:
    if schema_id != SCHEMA_ID:
        raise ValueError(f"Unsupported pose schema {schema_id!r}; expected {SCHEMA_ID!r}")
    return DWPOSE_WOFACE_62
