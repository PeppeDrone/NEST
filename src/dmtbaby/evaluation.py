"""Coordinate, confidence, temporal, and anatomy metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PoseMetrics:
    nme: float
    pck_005: float
    velocity_rmse: float
    acceleration_rmse: float
    confidence_f1: float
    bone_relative_error: float
    valid_points: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "nme": self.nme,
            "pck_005": self.pck_005,
            "velocity_rmse": self.velocity_rmse,
            "acceleration_rmse": self.acceleration_rmse,
            "confidence_f1": self.confidence_f1,
            "bone_relative_error": self.bone_relative_error,
            "valid_points": self.valid_points,
        }


def pose_metrics(
    predicted_xy: np.ndarray,
    target_xy: np.ndarray,
    predicted_visible: np.ndarray,
    target_visible: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> PoseMetrics:
    pred = np.asarray(predicted_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    pred_vis = np.asarray(predicted_visible, dtype=bool)
    target_vis = np.asarray(target_visible, dtype=bool)
    if pred.shape != target.shape or pred.shape[-1] != 2:
        raise ValueError(f"Coordinate shape mismatch: {pred.shape} versus {target.shape}")
    if pred_vis.shape != pred.shape[:-1] or target_vis.shape != pred.shape[:-1]:
        raise ValueError("Visibility shape does not match coordinates")

    joint_mask = pred_vis & target_vis & np.isfinite(pred).all(-1) & np.isfinite(target).all(-1)
    distances = np.linalg.norm(pred - target, axis=-1)
    valid_distances = distances[joint_mask]
    nme = float(valid_distances.mean()) if valid_distances.size else float("nan")
    pck = float((valid_distances <= 0.05).mean()) if valid_distances.size else float("nan")

    velocity_rmse = _difference_rmse(pred, target, joint_mask, order=1)
    acceleration_rmse = _difference_rmse(pred, target, joint_mask, order=2)
    confidence_f1 = binary_f1(pred_vis, target_vis)
    bone_error = bone_relative_error(pred, target, joint_mask, edges)
    return PoseMetrics(nme, pck, velocity_rmse, acceleration_rmse, confidence_f1, bone_error, int(joint_mask.sum()))


def binary_f1(predicted: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(predicted, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 1.0


def bone_relative_error(
    predicted: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    edges: Iterable[tuple[int, int]],
) -> float:
    errors: list[np.ndarray] = []
    for left, right in edges:
        edge_valid = valid[..., left] & valid[..., right]
        if not edge_valid.any():
            continue
        pred_length = np.linalg.norm(predicted[..., left, :] - predicted[..., right, :], axis=-1)
        target_length = np.linalg.norm(target[..., left, :] - target[..., right, :], axis=-1)
        relative = np.abs(pred_length - target_length) / np.maximum(target_length, 1e-6)
        errors.append(relative[edge_valid])
    if not errors:
        return float("nan")
    return float(np.concatenate(errors).mean())


def patient_cluster_bootstrap(
    values: Sequence[float],
    patients: Sequence[str],
    *,
    seed: int = 42,
    draws: int = 2000,
) -> tuple[float, float, float]:
    if len(values) != len(patients) or not values:
        raise ValueError("values and patients must be non-empty and have equal length")
    grouped: dict[str, list[float]] = {}
    for value, patient in zip(values, patients):
        if np.isfinite(value):
            grouped.setdefault(patient, []).append(float(value))
    keys = sorted(grouped)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        samples[draw] = np.mean([value for key in chosen for value in grouped[str(key)]])
    point = float(np.mean([value for group in grouped.values() for value in group]))
    low, high = np.quantile(samples, [0.025, 0.975])
    return point, float(low), float(high)


def _difference_rmse(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, order: int) -> float:
    pred_diff = np.diff(pred, n=order, axis=-3)
    target_diff = np.diff(target, n=order, axis=-3)
    diff_valid = valid.copy()
    for _ in range(order):
        diff_valid = diff_valid[..., 1:, :] & diff_valid[..., :-1, :]
    errors = np.sum((pred_diff - target_diff) ** 2, axis=-1)
    return float(np.sqrt(errors[diff_valid].mean())) if diff_valid.any() else float("nan")
