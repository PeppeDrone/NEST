"""Per-limb kinematic features of 2D trajectories, and distribution distances to the teacher.

All features use the teacher-visible support (never predicted visibility) and starting-pose
body units (normalized xy * image_size / torso scale of frame 0), so generated and teacher
trajectories of the same clip are measured on identical joints and frames.
"""
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

from neomotion.common.topology import LIMB_ANGLE_INDICES, BODY_LIMB_JOINT_INDICES, BODY_EDGES
from neomotion.representation import reference_frame

LIMBS = tuple(LIMB_ANGLE_INDICES)  # right_arm, left_arm, right_leg, left_leg
FEATURES = ("distal_speed", "distal_acceleration", "distal_jerk", "distal_path_length",
            "angle_range_deg", "bone_length_deviation")


def body_units(xy, pose0, visible0, image_size):
    """xy [B,T,20,2] normalized -> starting-pose body units (same factor as trajectory_metrics)."""
    frame = reference_frame(pose0, visible0, image_size)
    factor = image_size[:, None, None] / frame["scale"][:, None, None, None]
    return xy * factor


def _limb_bones(limb):
    joints = set(BODY_LIMB_JOINT_INDICES[limb])
    return [(a, b) for a, b in BODY_EDGES if a in joints and b in joints]


def limb_features(xy_body, valid, timestamps):
    """xy_body [B,T,20,2] (frame 0 included), valid [B,T,20] teacher support, timestamps [B,T].

    Returns dict feature -> array [B, 4 limbs] (NaN when the support is insufficient).
    """
    xy, valid, t = (np.asarray(a, dtype=np.float64) if a is not None else None for a in (xy_body, valid, timestamps))
    valid = valid.astype(bool)
    B = xy.shape[0]
    out = {f: np.full((B, len(LIMBS)), np.nan) for f in FEATURES}
    dt = np.diff(t, axis=1)                                       # [B,T-1]
    for li, limb in enumerate(LIMBS):
        a, m, d = LIMB_ANGLE_INDICES[limb]                         # proximal, middle (elbow/knee), distal
        for i in range(B):
            v = valid[i, :, d]
            p = xy[i, :, d]
            step_ok = v[1:] & v[:-1]
            if step_ok.sum() >= 1:
                vel = np.diff(p, axis=0) / dt[i][:, None]
                speed = np.linalg.norm(vel, axis=-1)
                out["distal_speed"][i, li] = speed[step_ok].mean()
                out["distal_path_length"][i, li] = (speed * dt[i])[step_ok].sum()
                acc_ok = step_ok[1:] & step_ok[:-1]
                if acc_ok.sum() >= 1:
                    acc = np.diff(vel, axis=0) / ((dt[i][1:] + dt[i][:-1]) / 2)[:, None]
                    out["distal_acceleration"][i, li] = np.linalg.norm(acc, axis=-1)[acc_ok].mean()
                    jerk_ok = acc_ok[1:] & acc_ok[:-1]
                    if jerk_ok.sum() >= 1:
                        jerk = np.diff(acc, axis=0) / dt[i][1:-1, None]
                        out["distal_jerk"][i, li] = np.linalg.norm(jerk, axis=-1)[jerk_ok].mean()
            tri = valid[i, :, a] & valid[i, :, m] & valid[i, :, d]
            if tri.sum() >= 2:
                u, w = xy[i, :, a] - xy[i, :, m], xy[i, :, d] - xy[i, :, m]
                nu, nw = np.linalg.norm(u, axis=-1), np.linalg.norm(w, axis=-1)
                ok = tri & (nu > 1e-6) & (nw > 1e-6)
                if ok.sum() >= 2:
                    cos = np.clip((u * w).sum(-1)[ok] / (nu[ok] * nw[ok]), -1, 1)
                    ang = np.degrees(np.arccos(cos))
                    out["angle_range_deg"][i, li] = ang.max() - ang.min()
            devs = []
            for ja, jb in _limb_bones(limb):
                ok = valid[i, :, ja] & valid[i, :, jb]
                if ok[0] and ok[1:].any():
                    length = np.linalg.norm(xy[i, :, ja] - xy[i, :, jb], axis=-1)
                    if length[0] > 1e-6:
                        devs.append(np.abs(length[1:][ok[1:]] / length[0] - 1).mean())
            if devs:
                out["bone_length_deviation"][i, li] = float(np.mean(devs))
    return out


def w1(pred, teacher):
    """1-Wasserstein distance between two 1D samples (NaNs dropped)."""
    pred, teacher = np.asarray(pred, float), np.asarray(teacher, float)
    pred, teacher = pred[np.isfinite(pred)], teacher[np.isfinite(teacher)]
    if len(pred) < 2 or len(teacher) < 2:
        return np.nan
    return float(wasserstein_distance(pred, teacher))


def frechet(pred, teacher):
    """Gaussian Fréchet distance between feature matrices [N,F] (rows with NaN dropped),
    after standardizing every feature by the teacher's mean/std."""
    pred, teacher = np.asarray(pred, float), np.asarray(teacher, float)
    pred, teacher = pred[np.isfinite(pred).all(1)], teacher[np.isfinite(teacher).all(1)]
    if len(pred) < 5 or len(teacher) < 5:
        return np.nan
    mu, sd = teacher.mean(0), teacher.std(0) + 1e-9
    p, q = (pred - mu) / sd, (teacher - mu) / sd
    m1, m2 = p.mean(0), q.mean(0)
    c1, c2 = np.cov(p, rowvar=False), np.cov(q, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    covmean = covmean.real if np.iscomplexobj(covmean) else covmean
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))
