"""Load the unchanged accepted teachers and verify the frozen patient split."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from .common.teacher_data import TeacherPoseDataset, collate_teacher_samples
from .representation import reference_frame, encode_future, fit_residual_scales


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_split(manifest, split_path):
    rows = [json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    frozen = json.loads(Path(split_path).read_text())
    seen, patients = set(), {}
    for row in rows:
        clip, split, patient = row["clip_id"], row["split"], row["patient_id"]
        if clip in seen or split not in ("train", "val", "test") or clip not in frozen[split]:
            raise ValueError(f"Invalid/duplicate frozen split membership: {clip}")
        if patient not in frozen[f"{split}_patients"]:
            raise ValueError(f"Patient {patient} is not assigned to {split}")
        if patient in patients and patients[patient] != split:
            raise ValueError(f"Patient leakage: {patient}")
        patients[patient] = split
        seen.add(clip)
    return {"teacher_manifest_sha256": file_sha256(manifest), "patient_split_sha256": file_sha256(split_path),
            "counts": dict(Counter(r["split"] for r in rows)), "patients": patients}


def load_split(manifest, prompt_source, split):
    """Never reads annotation CSV or substitutes another prompt source."""
    manifest = Path(manifest).resolve()
    dataset = TeacherPoseDataset(manifest, prompt_source=prompt_source, joint_count=20,
                                 split=split, verify_checksums=True)
    samples, sizes, times, patients = [], [], [], []
    for i, record in enumerate(dataset.records):
        sample = dataset[i]
        reference = (manifest.parent / record["reference_image"]).resolve()
        with Image.open(reference) as image:
            sizes.append(image.size)
        video = (manifest.parent / record["source_video"]).resolve()
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot read timestamp metadata: {video}")
            fps = capture.get(cv2.CAP_PROP_FPS)
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source FPS for {record['sample_id']}")
        with np.load(record["pose_path"], allow_pickle=False) as archive:
            frames = archive["source_frame"].astype(np.float32)
        timestamps = (frames - frames[0]) / fps
        if not (np.diff(timestamps) > 0).all():
            raise ValueError(f"Native timestamps must increase: {record['sample_id']}")
        samples.append(sample)
        times.append(timestamps)
        patients.append(record["patient_id"])
    batch = collate_teacher_samples(samples)
    result = {"pose0": batch["pose0_xy_norm"], "score0": batch["pose0_raw_score"],
              "visible0": batch["pose0_render_visible"], "target": batch["target_xy_norm"],
              "valid": batch["target_render_visible"], "image_size": torch.tensor(sizes, dtype=torch.float32),
              "timestamps": torch.tensor(np.stack(times)), "ids": batch["sample_id"],
              "prompts": batch["prompt"], "patients": patients}
    if not torch.isfinite(result["timestamps"]).all():
        raise ValueError("Nonfinite timestamps")
    return result


def tensor_batch(data, indices, device="cpu"):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {k: v[indices].to(device) if isinstance(v, torch.Tensor) else [v[i] for i in indices.tolist()]
            for k, v in data.items()}


def fit_training_normalization(train, floor):
    frame = reference_frame(train["pose0"], train["visible0"], train["image_size"])
    states = encode_future(train["target"], train["valid"], frame)
    scales = fit_residual_scales(states, train["valid"], train["visible0"], floor)
    # Teacher-derived activity bins are identical in both prompt tracks.
    displacement = torch.linalg.vector_norm(states, dim=-1)
    mask = train["valid"] & train["visible0"][:, None]
    activity = (displacement * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
    threshold = float(activity.median())
    return scales, activity, threshold


def balanced_weights(patients, activity, threshold):
    """Equal patient mass, then equal mass across observed activity bins per patient."""
    bins = (activity > threshold).long().tolist()
    counts = Counter(zip(patients, bins))
    classes = {p: len({b for q, b in counts if q == p}) for p in set(patients)}
    weights = torch.tensor([1. / (classes[p] * counts[p, b]) for p, b in zip(patients, bins)], dtype=torch.float64)
    return weights / weights.sum()


def patient_subset(data, count):
    """Deterministic round-robin across patients, used only for explicit smoke checks."""
    if count is None or count >= len(data["ids"]):
        return data
    groups = {p: [i for i, q in enumerate(data["patients"]) if p == q] for p in sorted(set(data["patients"]))}
    indices = []
    offset = 0
    while len(indices) < count:
        for group in groups.values():
            if offset < len(group):
                indices.append(group[offset])
                if len(indices) == count:
                    break
        offset += 1
    return tensor_batch(data, indices)
