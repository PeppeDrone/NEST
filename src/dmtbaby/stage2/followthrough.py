"""Re-extract DWPose from frozen Stage-2 videos and compare to the control images.

UniAnimate sees bilinearly resized DWPose RGB controls. Follow-through is the
body-20 pose error between (i) DWPose run on those resized control frames and
(ii) DWPose run on the generated ``output.mp4``. Both sequences share the video
resolution, so aspect-ratio stretch of 512x768 controls onto 832x480 (or any
other request size) is included on both sides.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..constants import DWPOSE_DET_THRESHOLD, DWPOSE_POSE_THRESHOLD, UNIANIMATE_FRAMES
from ..evaluation import patient_cluster_bootstrap, pose_metrics
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_npz
from ..pose.schema import BODY_RENDER_EDGES, NUM_BODY_JOINTS
from .controls import validate_control_sequence


GENERATED_CONDITIONS = ("manual20_generated", "automatic20_generated")
ORACLE_CONDITIONS = ("manual20_oracle", "automatic20_oracle")


class FrameExtractor(Protocol):
    def extract_frame(self, image_bgr: Any, person_index: int = 0) -> Any:
        """Return a canonical pose frame for one BGR image."""


def _resolve(root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() in {"", "None"}:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            clip_id = str(value.get("clip_id") or "").strip()
            if not clip_id:
                raise ValueError(f"Missing clip_id at {path}:{line_number}")
            row = dict(value)
            row["clip_id"] = clip_id
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def decode_stage2_video(
    path: str | Path,
    *,
    expected_frames: int = UNIANIMATE_FRAMES,
) -> list[np.ndarray]:
    """Decode every frame of a Stage-2 ``output.mp4`` as BGR."""

    import cv2

    video_path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open Stage-2 video {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if expected_frames is not None and len(frames) != expected_frames:
        raise ValueError(
            f"{video_path} has {len(frames)} decoded frames; expected {expected_frames}"
        )
    if not frames:
        raise ValueError(f"{video_path} decoded zero frames")
    return frames


def load_resized_control_frames(
    control_dir: str | Path,
    *,
    width: int,
    height: int,
    reference_pose: str | Path | None = None,
) -> list[np.ndarray]:
    """Load 81 control JPEGs and bilinear-resize them to the video size.

    Matches UniAnimate's PIL RGB resize, then converts to BGR for DWPose.
    """

    from PIL import Image

    controls = validate_control_sequence(control_dir, reference_pose)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    frames: list[np.ndarray] = []
    for path in controls.frames:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((int(width), int(height)), resampling)
        frames.append(np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]))
    return frames


def body20_from_frames(
    extractor: FrameExtractor,
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract body-20 xy and render-visibility from BGR frames."""

    xy = np.full((len(frames), NUM_BODY_JOINTS, 2), np.nan, dtype=np.float32)
    visible = np.zeros((len(frames), NUM_BODY_JOINTS), dtype=np.bool_)
    for index, image in enumerate(frames):
        frame = extractor.extract_frame(image)
        xy[index] = np.asarray(frame.xy_norm[:NUM_BODY_JOINTS], dtype=np.float32)
        visible[index] = np.asarray(frame.render_visible[:NUM_BODY_JOINTS], dtype=np.bool_)
    return xy, visible


def followthrough_metrics(
    video_xy: np.ndarray,
    video_visible: np.ndarray,
    control_xy: np.ndarray,
    control_visible: np.ndarray,
) -> dict[str, float | int]:
    """Body-20 pose metrics of video-extracted pose versus control-extracted pose."""

    metrics = pose_metrics(
        video_xy,
        control_xy,
        video_visible,
        control_visible,
        BODY_RENDER_EDGES,
    )
    packed: dict[str, float | int | None] = {}
    for key, value in metrics.as_dict().items():
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            packed[key] = int(value)
            continue
        number = float(value)
        packed[key] = number if np.isfinite(number) else None
    packed["generated_pose_error"] = packed["nme"]
    packed["n_frames"] = int(video_xy.shape[0])
    return packed


def pose_array_path(pose_root: Path, clip_id: str, condition: str) -> Path:
    safe = str(clip_id).replace("/", "_")
    return Path(pose_root) / safe / f"{condition}.npz"


def save_pose_arrays(
    path: str | Path,
    *,
    video_xy: np.ndarray,
    video_visible: np.ndarray,
    control_xy: np.ndarray,
    control_visible: np.ndarray,
) -> Path:
    destination = Path(path)
    atomic_write_npz(
        destination,
        {
            "video_xy": np.asarray(video_xy, dtype=np.float32),
            "video_visible": np.asarray(video_visible, dtype=np.uint8),
            "control_xy": np.asarray(control_xy, dtype=np.float32),
            "control_visible": np.asarray(control_visible, dtype=np.uint8),
        },
    )
    return destination


def load_pose_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {
            "video_xy": np.asarray(archive["video_xy"], dtype=np.float32),
            "video_visible": np.asarray(archive["video_visible"], dtype=bool),
            "control_xy": np.asarray(archive["control_xy"], dtype=np.float32),
            "control_visible": np.asarray(archive["control_visible"], dtype=bool),
        }


def attach_generated_vs_oracle(
    rows: Sequence[Mapping[str, Any]],
    pose_root: str | Path,
) -> list[dict[str, Any]]:
    """Score generated-video poses against the clip's oracle control pose."""

    root = Path(pose_root)
    updated = [dict(row) for row in rows]
    by_clip: dict[str, dict[str, dict[str, Any]]] = {}
    for row in updated:
        by_clip.setdefault(str(row["clip_id"]), {})[str(row.get("condition", ""))] = row

    for clip_id, by_condition in by_clip.items():
        oracle_arrays = None
        oracle_source = None
        for condition in ORACLE_CONDITIONS:
            row = by_condition.get(condition)
            if row is None or row.get("state") != "ok":
                continue
            path = pose_array_path(root, clip_id, condition)
            if not path.is_file():
                continue
            oracle_arrays = load_pose_arrays(path)
            oracle_source = condition
            break
        if oracle_arrays is None:
            continue
        for condition in GENERATED_CONDITIONS:
            row = by_condition.get(condition)
            if row is None or row.get("state") != "ok":
                continue
            path = pose_array_path(root, clip_id, condition)
            if not path.is_file():
                continue
            generated = load_pose_arrays(path)
            metrics = followthrough_metrics(
                generated["video_xy"],
                generated["video_visible"],
                oracle_arrays["control_xy"],
                oracle_arrays["control_visible"],
            )
            row["nme_vs_oracle"] = metrics["nme"]
            row["pck_005_vs_oracle"] = metrics["pck_005"]
            row["bone_relative_error_vs_oracle"] = metrics["bone_relative_error"]
            row["valid_points_vs_oracle"] = metrics["valid_points"]
            row["oracle_control_source"] = oracle_source
    return updated


def summarize_generated_vs_oracle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired manual vs automatic generated-video NME against the oracle control."""

    by_clip: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("state") != "ok" or row.get("nme_vs_oracle") is None:
            continue
        condition = str(row.get("condition", ""))
        if condition not in GENERATED_CONDITIONS:
            continue
        by_clip.setdefault(str(row["clip_id"]), {})[condition] = row

    pairs: list[dict[str, Any]] = []
    for clip_id in sorted(by_clip):
        manual = by_clip[clip_id].get("manual20_generated")
        automatic = by_clip[clip_id].get("automatic20_generated")
        if manual is None or automatic is None:
            continue
        manual_nme = float(manual["nme_vs_oracle"])
        automatic_nme = float(automatic["nme_vs_oracle"])
        if not np.isfinite(manual_nme) or not np.isfinite(automatic_nme):
            continue
        delta = manual_nme - automatic_nme
        pairs.append(
            {
                "clip_id": clip_id,
                "split": manual.get("split") or automatic.get("split"),
                "patient_id": manual.get("patient_id") or automatic.get("patient_id"),
                "nme_manual": manual_nme,
                "nme_automatic": automatic_nme,
                "pck_manual": manual.get("pck_005_vs_oracle"),
                "pck_automatic": automatic.get("pck_005_vs_oracle"),
                "delta_nme_manual_minus_automatic": delta,
                "closer": "manual" if delta < 0 else "automatic" if delta > 0 else "tie",
            }
        )

    deltas = [row["delta_nme_manual_minus_automatic"] for row in pairs]
    wilcoxon: dict[str, Any] | None = None
    if len(deltas) >= 6:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        result = scipy_wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        wilcoxon = {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
    return {
        "n_paired": len(pairs),
        "n_manual_closer": sum(row["closer"] == "manual" for row in pairs),
        "n_automatic_closer": sum(row["closer"] == "automatic" for row in pairs),
        "n_tie": sum(row["closer"] == "tie" for row in pairs),
        "median_nme_manual": float(np.median([row["nme_manual"] for row in pairs])) if pairs else None,
        "median_nme_automatic": (
            float(np.median([row["nme_automatic"] for row in pairs])) if pairs else None
        ),
        "median_delta_nme": float(np.median(deltas)) if deltas else None,
        "mean_delta_nme": float(np.mean(deltas)) if deltas else None,
        "wilcoxon_two_sided": wilcoxon,
        "clips": pairs,
    }


def _patient_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("patient_id") or "").strip()
    if value:
        return value
    clip_id = str(row["clip_id"])
    return clip_id.split("_", 1)[0]


def _aggregate_condition(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    values: list[float] = []
    patients: list[str] = []
    for row in rows:
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))
        patients.append(str(row["patient_id"]))
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    mean, low, high = patient_cluster_bootstrap(values, patients, seed=seed, draws=draws)
    return {
        "n": len(values),
        "mean": mean,
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [low, high],
    }


def select_followthrough_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    clip_ids: Sequence[str] = (),
    limit: int | None = None,
    require_video: bool = True,
) -> list[dict[str, Any]]:
    """Keep manifest rows that have a video and a control directory."""

    selected = set(clip_ids)
    kept: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row["clip_id"])
        if selected and clip_id not in selected:
            continue
        video = _resolve(root, row.get("generated_video") or row.get("output"))
        control_dir = _resolve(root, row.get("control_dir"))
        if video is None or control_dir is None:
            continue
        if require_video and (not video.is_file() or video.stat().st_size <= 0):
            continue
        packed = dict(row)
        packed["generated_video"] = str(video)
        packed["control_dir"] = str(control_dir)
        packed["reference_pose"] = str(
            _resolve(root, row.get("reference_pose")) or (control_dir / "pose.jpg")
        )
        packed["patient_id"] = _patient_id(row)
        packed["condition"] = str(row.get("condition") or "").strip() or "unspecified"
        kept.append(packed)
        if limit is not None and len(kept) >= int(limit):
            break
    return kept


def measure_row(
    row: Mapping[str, Any],
    extractor: FrameExtractor,
    *,
    pose_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract poses from one video/control pair and return metric fields."""

    video_frames = decode_stage2_video(row["generated_video"])
    height, width = video_frames[0].shape[:2]
    control_frames = load_resized_control_frames(
        row["control_dir"],
        width=width,
        height=height,
        reference_pose=row.get("reference_pose"),
    )
    if len(control_frames) != len(video_frames):
        raise ValueError(
            f"{row['clip_id']}: control frames {len(control_frames)} vs video {len(video_frames)}"
        )
    video_xy, video_visible = body20_from_frames(extractor, video_frames)
    control_xy, control_visible = body20_from_frames(extractor, control_frames)
    metrics = followthrough_metrics(video_xy, video_visible, control_xy, control_visible)
    out = dict(row)
    out.update(metrics)
    out["video_width"] = int(width)
    out["video_height"] = int(height)
    out["state"] = "ok"
    out["error"] = None
    if pose_path is not None:
        saved = save_pose_arrays(
            pose_path,
            video_xy=video_xy,
            video_visible=video_visible,
            control_xy=control_xy,
            control_visible=control_visible,
        )
        out["pose_npz"] = str(saved)
    return out


def make_dwpose_extractor(
    *,
    root: Path,
    provider: str = "cuda",
    allow_cpu_smoke: bool = False,
    det_threshold: float = DWPOSE_DET_THRESHOLD,
    pose_threshold: float = DWPOSE_POSE_THRESHOLD,
    detector_model: str | Path | None = None,
    pose_model: str | Path | None = None,
) -> FrameExtractor:
    """Build the production DWPose extractor. Do not call this from unit tests."""

    from ..pose.extraction import OrtModels, OrtWholebodyExtractor

    requested = "cpu" if str(provider).lower().startswith("cpu") else "cuda"
    if requested == "cpu" and not allow_cpu_smoke:
        raise ValueError("CPU ONNX extraction is smoke-test only; pass allow_cpu_smoke explicitly")
    detector = _resolve(
        root,
        detector_model
        or "external/UniAnimate-DiT/models/UniAnimate-DiT/yolox_l.onnx",
    )
    pose = _resolve(
        root,
        pose_model
        or "external/UniAnimate-DiT/models/UniAnimate-DiT/dw-ll_ucoco_384.onnx",
    )
    if detector is None or pose is None or not detector.is_file() or not pose.is_file():
        raise FileNotFoundError("DWPose ONNX models are missing")
    return OrtWholebodyExtractor(
        OrtModels(detector=detector, pose=pose),
        providers=requested,
        det_threshold=float(det_threshold),
        pose_threshold=float(pose_threshold),
        external_root=root / "external/UniAnimate-DiT",
        clip_confidence=True,
    )


def run_stage2_followthrough(
    *,
    root: str | Path,
    manifest: str | Path,
    output: str | Path,
    clip_ids: Sequence[str] = (),
    limit: int | None = None,
    dry_run: bool = False,
    extractor: FrameExtractor | None = None,
    provider: str = "cuda",
    allow_cpu_smoke: bool = False,
    seed: int = 42,
    bootstrap_draws: int = 2000,
) -> dict[str, Any]:
    """Measure Stage-2 pose follow-through for every available video in ``manifest``."""

    project = Path(root).expanduser().resolve()
    manifest_path = _resolve(project, manifest)
    output_path = _resolve(project, output)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"follow-through manifest does not exist: {manifest}")
    if output_path is None:
        raise ValueError("output path is required")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")

    selected = select_followthrough_rows(
        _read_manifest(manifest_path),
        root=project,
        clip_ids=clip_ids,
        limit=limit,
        require_video=not dry_run,
    )
    pose_root = output_path.with_name(f"{output_path.stem}_poses")
    report: dict[str, Any] = {
        "format": "dmtbaby.stage2_followthrough.v1",
        "manifest": str(manifest_path),
        "n_selected": len(selected),
        "dry_run": bool(dry_run),
        "pose_root": str(pose_root),
        "rows": [],
        "conditions": {},
        "generated_vs_oracle": {},
    }
    if dry_run:
        report["rows"] = [
            {
                "clip_id": row["clip_id"],
                "condition": row["condition"],
                "generated_video": row["generated_video"],
                "control_dir": row["control_dir"],
                "state": "planned",
            }
            for row in selected
        ]
        atomic_write_json(output_path, report)
        return report

    if extractor is None:
        extractor = make_dwpose_extractor(
            root=project,
            provider=provider,
            allow_cpu_smoke=allow_cpu_smoke,
        )

    measured: list[dict[str, Any]] = []
    for row in selected:
        pose_path = pose_array_path(pose_root, str(row["clip_id"]), str(row["condition"]))
        try:
            measured.append(measure_row(row, extractor, pose_path=pose_path))
        except Exception as exc:
            failed = dict(row)
            failed.update(
                {
                    "state": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_pose_error": None,
                    "nme": None,
                    "pck_005": None,
                    "pose_npz": str(pose_path),
                }
            )
            measured.append(failed)

    measured = attach_generated_vs_oracle(measured, pose_root)
    ok_rows = [row for row in measured if row.get("state") == "ok"]
    conditions = sorted({str(row["condition"]) for row in ok_rows})
    report["n_ok"] = len(ok_rows)
    report["n_error"] = sum(1 for row in measured if row.get("state") == "error")
    report["rows"] = measured
    report["generated_vs_oracle"] = summarize_generated_vs_oracle(measured)
    report["conditions"] = {
        condition: {
            metric: _aggregate_condition(
                [row for row in ok_rows if row["condition"] == condition],
                metric=metric,
                seed=seed,
                draws=bootstrap_draws,
            )
            for metric in (
                "generated_pose_error",
                "nme",
                "pck_005",
                "bone_relative_error",
                "nme_vs_oracle",
                "pck_005_vs_oracle",
            )
        }
        for condition in conditions
    }
    atomic_write_json(output_path, report)
    atomic_write_jsonl(output_path.with_suffix(".jsonl"), measured)
    return report
