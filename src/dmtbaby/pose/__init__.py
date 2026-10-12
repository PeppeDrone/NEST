"""Canonical pose interfaces for the DMTBaby Stage-1 -> UniAnimate bridge.

The package deliberately imports no heavy optional dependencies at module import
time. NumPy, SciPy, OpenCV, Pillow, PyTorch, and ONNX Runtime are loaded only by
the functions that need them.
"""

from .records import CanonicalPoseFrame, PredictionPoseSequence, TeacherPoseSequence
from .schema import (
    BODY20_SCOPE,
    BODY_JOINT_NAMES,
    BODY_RENDER_EDGES,
    DWPOSE_WOFACE_62,
    HAND_JOINT_NAMES,
    JOINT_NAMES,
    SCHEMA_ID,
    WHOLE62_SCOPE,
)

__all__ = [
    "BODY_JOINT_NAMES",
    "BODY_RENDER_EDGES",
    "BODY20_SCOPE",
    "CanonicalPoseFrame",
    "DWPOSE_WOFACE_62",
    "HAND_JOINT_NAMES",
    "JOINT_NAMES",
    "PredictionPoseSequence",
    "SCHEMA_ID",
    "TeacherPoseSequence",
    "WHOLE62_SCOPE",
]
