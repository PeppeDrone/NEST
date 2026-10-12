"""Stable joint ordering and exact no-face DWPose graph topology."""

from __future__ import annotations

from functools import lru_cache


BODY_JOINT_NAMES: tuple[str, ...] = (
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
    "left_foot_center",
    "right_foot_center",
)

HAND_POINT_NAMES: tuple[str, ...] = (
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

LEFT_HAND_JOINT_NAMES = tuple(f"left_hand_{name}" for name in HAND_POINT_NAMES)
RIGHT_HAND_JOINT_NAMES = tuple(f"right_hand_{name}" for name in HAND_POINT_NAMES)
FULL_JOINT_NAMES = BODY_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES

# Zero-based form of dwpose.util.draw_body_and_foot's exact limbSeq.
BODY_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (13, 18),  # left ankle -> left foot center
    (10, 19),  # right ankle -> right foot center
)

HAND_LOCAL_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def joint_names(joint_count: int) -> tuple[str, ...]:
    if joint_count == 20:
        return BODY_JOINT_NAMES
    if joint_count == 62:
        return FULL_JOINT_NAMES
    raise ValueError("joint_count must be 20 or 62")


def topology_edges(joint_count: int) -> tuple[tuple[int, int], ...]:
    if joint_count == 20:
        return BODY_EDGES
    if joint_count != 62:
        raise ValueError("joint_count must be 20 or 62")
    left = tuple((a + 20, b + 20) for a, b in HAND_LOCAL_EDGES)
    right = tuple((a + 41, b + 41) for a, b in HAND_LOCAL_EDGES)
    return BODY_EDGES + left + right


def loss_groups(joint_count: int) -> tuple[tuple[int, ...], ...]:
    if joint_count == 20:
        return (tuple(range(20)),)
    if joint_count == 62:
        return (tuple(range(20)), tuple(range(20, 62)))
    raise ValueError("joint_count must be 20 or 62")


LIMB_KEYS: tuple[str, ...] = ("left_arm", "right_arm", "left_leg", "right_leg")

BODY_LIMB_JOINT_INDICES: dict[str, tuple[int, ...]] = {
    "right_arm": (2, 3, 4),
    "left_arm": (5, 6, 7),
    "right_leg": (8, 9, 10, 19),
    "left_leg": (11, 12, 13, 18),
}

LIMB_ANGLE_INDICES: dict[str, tuple[int, int, int]] = {
    "right_arm": (2, 3, 4),
    "left_arm": (5, 6, 7),
    "right_leg": (8, 9, 10),
    "left_leg": (11, 12, 13),
}

LIMB_DISTAL_INDEX: dict[str, int] = {
    "right_arm": 4,
    "left_arm": 7,
    "right_leg": 10,
    "left_leg": 13,
}


def limb_joint_indices(limb: str, joint_count: int) -> tuple[int, ...]:
    """Body (and, for Whole62, ipsilateral hand) joints belonging to one limb."""

    if limb not in BODY_LIMB_JOINT_INDICES:
        raise ValueError(f"unknown limb {limb!r}")
    joint_names(joint_count)
    indices = BODY_LIMB_JOINT_INDICES[limb]
    if joint_count == 20:
        return indices
    if limb == "left_arm":
        return indices + tuple(range(20, 41))
    if limb == "right_arm":
        return indices + tuple(range(41, 62))
    return indices


def _limb_id(index: int) -> int:
    body_groups = (
        {0, 1, 14, 15, 16, 17},
        {2, 3, 4},
        {5, 6, 7},
        {8, 9, 10, 19},
        {11, 12, 13, 18},
    )
    if index < 20:
        for group_id, group in enumerate(body_groups):
            if index in group:
                return group_id
        return 5
    if index < 41:
        local = index - 20
        return 10 + (0 if local == 0 else (local - 1) // 4 + 1)
    local = index - 41
    return 20 + (0 if local == 0 else (local - 1) // 4 + 1)


def _relation_class(left: int, right: int, connected: set[tuple[int, int]], *, contralateral: bool) -> int:
    """Class IDs: 0 other, 1 same limb, 2 graph edge, 3 self, 4 contralateral."""

    if left == right:
        return 3
    if tuple(sorted((left, right))) in connected:
        return 2
    if _limb_id(left) == _limb_id(right):
        return 1
    if contralateral and _is_contralateral(left, right):
        return 4
    return 0


def _side_family(index: int) -> tuple[str, str] | None:
    """Return ``(arm|leg, left|right)`` for a lateral joint, else ``None``."""

    limb = _limb_id(index)
    if limb == 1:
        return ("arm", "right")
    if limb == 2:
        return ("arm", "left")
    if limb == 3:
        return ("leg", "right")
    if limb == 4:
        return ("leg", "left")
    if 10 <= limb <= 15:
        return ("arm", "left")
    if 20 <= limb <= 25:
        return ("arm", "right")
    return None


def _is_contralateral(left: int, right: int) -> bool:
    left_side = _side_family(left)
    right_side = _side_family(right)
    if left_side is None or right_side is None:
        return False
    return left_side[0] == right_side[0] and left_side[1] != right_side[1]


def _attention_bias_table(joint_count: int, *, contralateral: bool) -> tuple[tuple[int, ...], ...]:
    joint_names(joint_count)  # validate
    connected = {tuple(sorted(edge)) for edge in topology_edges(joint_count)}
    rows: list[tuple[int, ...]] = []
    for left in range(joint_count):
        rows.append(
            tuple(
                _relation_class(left, right, connected, contralateral=contralateral)
                for right in range(joint_count)
            )
        )
    return tuple(rows)


@lru_cache(maxsize=2)
def attention_bias_classes(joint_count: int) -> tuple[tuple[int, ...], ...]:
    """Return class IDs: 0 other, 1 same limb, 2 graph edge, 3 self."""

    return _attention_bias_table(joint_count, contralateral=False)


@lru_cache(maxsize=2)
def attention_bias_classes_v2(joint_count: int) -> tuple[tuple[int, ...], ...]:
    """v1 classes plus 4 for contralateral arm and leg pairs."""

    return _attention_bias_table(joint_count, contralateral=True)
