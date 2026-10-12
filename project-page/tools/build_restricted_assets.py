"""Build the demonstration media derived from the infant recordings (cleared for publication by the authors).

    python tools/build_restricted_assets.py \
        --stage1 /path/to/stage1_paper  --stage2-run /path/to/stage2/run  --figures /path/to/manuscript/figures \
        --figure-code /path/to/manuscript/figure_code  --stage2-clip pz11_clip0047

Outputs (static/):
  images/fig4_single_infant.svg                       Fig. 4 of the paper
  images/stage2_sample.webp                           reference still + generated frames every 0.5 s (face
                                                      blurred) and leg displacement, drawn with the paper's
                                                      Fig. 10 code for the --stage2-clip sample
  videos/commands/{manual,automatic}_<command>.mp4   skeleton animations, 13 commands x 2 caption sources,
                                                      rule-selected test infant, first sampling seed
  videos/stage2_sample.mp4                            generated video (face blurred) beside its control skeleton
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "static"
COMMANDS = ["still"] + [f"{l}_{g}" for l in ("right_arm", "left_arm", "right_leg", "left_leg")
                        for g in ("low_slow", "high_slow", "high_fast")]
DISTAL = {"right_arm": 4, "left_arm": 7, "right_leg": 10, "left_leg": 13}
HEAD = [0, 14, 15, 16, 17]
LEFT = {5, 6, 7, 11, 12, 13, 15, 17, 18}
RIGHT = {2, 3, 4, 8, 9, 10, 14, 16, 19}
COLORS = {"v7_manual": ("#004a75", "#56B4E9"), "v7_automatic": ("#8a4b00", "#E8A64F")}
FPS = 24


def writer(path, fps, size):
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=None,
                                    output_params=["-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    w.send(None)
    return w


def interp_frames(xy, vis, times, fps):
    """Linear interpolation between native frames for smooth display (positions only where visible at both ends)."""
    t_out = np.arange(0, times[-1] + 1e-9, 1.0 / fps)
    k = np.clip(np.searchsorted(times, t_out, side="right") - 1, 0, len(times) - 2)
    a = ((t_out - times[k]) / (times[k + 1] - times[k]))[:, None, None]
    out = xy[k] * (1 - a) + xy[k + 1] * a
    v = vis[k] & vis[k + 1]
    return out, v, t_out


def command_videos(stage1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(stage1 / "analysis/artifacts/hero/trajectories.npz")
    meta = json.loads((stage1 / "analysis/artifacts/hero/choice.json").read_text())
    t = np.load(stage1 / "analysis/artifacts/test_partA/teacher/teacher.npz")
    i = list(t["ids"]).index(meta["clip"])
    px = t["image_size"][i].astype(float); times = t["timestamps"][i].astype(float)
    edges = t["edges"]
    allpts = np.concatenate([z[f"{a}__{c}__xy"][0].reshape(-1, 2) * px for a in COLORS for c in COMMANDS])
    lo, hi = np.percentile(allpts, 1, 0), np.percentile(allpts, 99, 0)
    c0, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.2
    (OUT / "videos/commands").mkdir(parents=True, exist_ok=True)
    size = 320
    for arm, (dark, light) in COLORS.items():
        for cmd in COMMANDS:
            xy = z[f"{arm}__{cmd}__xy"][0].astype(float) * px            # first sampling seed, [20, 20, 2]
            vis = z[f"{arm}__{cmd}__vis"][0].astype(bool)
            fx, fv, tt = interp_frames(xy, vis, times, FPS)
            target = None if cmd == "still" else cmd.rsplit("_", 2)[0]
            w = writer(OUT / "videos/commands" / f"{arm.replace('v7_', '')}_{cmd}.mp4", FPS, (size, size))
            fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            for f in range(len(tt)):
                ax.clear(); ax.set_xlim(c0[0] - half, c0[0] + half); ax.set_ylim(c0[1] + half, c0[1] - half)
                ax.set_aspect("equal"); ax.axis("off")
                for a, b in edges:                                      # frame-0 skeleton (grey)
                    if vis[0, a] and vis[0, b]:
                        ax.plot(xy[0, [a, b], 0], xy[0, [a, b], 1], color="#cfcfcf", lw=2.2, solid_capstyle="round")
                if target:                                              # trail of the commanded distal joint
                    j = DISTAL[target]; ok = fv[: f + 1, j]
                    ax.plot(fx[: f + 1, j, 0][ok], fx[: f + 1, j, 1][ok], color=dark, lw=1.4, alpha=0.8)
                for a, b in edges:
                    if fv[f, a] and fv[f, b]:
                        c = dark if (target and (a in LIMB_JOINTS[target] and b in LIMB_JOINTS[target])) else light
                        ax.plot(fx[f, [a, b], 0], fx[f, [a, b], 1], color=c, lw=2.4, solid_capstyle="round")
                ax.scatter(fx[f, fv[f], 0], fx[f, fv[f], 1], s=7, color="#1a1a1a", zorder=3)
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                w.send(np.ascontiguousarray(img))
            w.close(); plt.close(fig)
            print(f"  {arm} {cmd}")
    return meta["clip"]


LIMB_JOINTS = {"right_arm": {2, 3, 4}, "left_arm": {5, 6, 7}, "right_leg": {8, 9, 10, 19}, "left_leg": {11, 12, 13, 18}}


def blur_face(img, head_xy, min_frac=0.095):
    import cv2
    pts = np.asarray([p for p in head_xy if np.isfinite(p).all()])
    if not len(pts):
        return img
    h, w = img.shape[:2]
    c = pts.mean(0)
    span = np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)) if len(pts) > 1 else 0
    r = max(1.5 * span, min_frac * max(h, w))
    small = cv2.resize(img, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
    pix = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), (0, 0), r / 3)
    yy, xx = np.mgrid[:h, :w]
    m = np.clip((r * 1.15 - np.hypot(xx - c[0], yy - c[1])) / (0.15 * r), 0, 1)[..., None]
    return (img * (1 - m) + pix * m).astype(np.uint8)


def stage2_video(stage1, run, clip):
    import cv2
    req = {json.loads(l)["clip_id"]: json.loads(l) for l in (run / "stage2_requests.jsonl").read_text().splitlines() if l.strip()}
    r = req[clip]
    a = np.load(stage1 / "analysis/artifacts/stage2/aligned" / f"{clip}.npz")
    cap = cv2.VideoCapture(r["output"]); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, :, ::-1])
    H, W = frames[0].shape[:2]
    vxy = a["video_xy"].astype(float); vvis = a["video_visible"].astype(bool)
    seen = np.where(vvis[:, HEAD, None], vxy[:, HEAD], np.nan)
    with np.errstate(all="ignore"):
        median_head = np.nanmedian(seen.reshape(-1, 2), axis=0)
    ctrl_dir = Path(r["control_dir"])
    out_w = 2 * W
    w = writer(OUT / "videos/stage2_sample.mp4", 15, (out_w, H))
    for k, img in enumerate(frames):
        img = img.copy()
        for est in (vxy[k][HEAD], a["control_xy"][k][HEAD], a["real_xy"][k][HEAD], median_head[None]):   # privacy
            est = np.asarray(est, float).reshape(-1, 2) * [W, H]
            if np.isfinite(est).all(-1).any():
                img = blur_face(img, est)
        ctrl = cv2.imread(str(ctrl_dir / f"{k:04d}.jpg"))[:, :, ::-1]
        ctrl = cv2.resize(ctrl, (W, H), interpolation=cv2.INTER_AREA)
        w.send(np.ascontiguousarray(np.concatenate([ctrl, img], axis=1)))
    w.close()
    print("  stage2_sample.mp4")


def figures(fig_dir):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(fig_dir / "fig4_single_infant.pdf"), str(OUT / "images" / "fig4_single_infant.svg")], check=True)
    print("  fig4")


def stage2_figure(figure_code, clip, excerpt=None):
    """The paper's Fig. 10 layout (make_figures.draw_stage2_cases) for the page's sample clip."""
    import sys
    import tempfile
    import pandas as pd
    from PIL import Image
    sys.path.insert(0, str(figure_code))
    import make_figures as mf
    folder = mf.D.ART / "stage2"
    row = pd.read_csv(folder / "per_video.csv").set_index("clip_id").loc[clip].copy()
    row["clip_id"] = clip
    mf.FIG10_PROMPT_EXCERPT[clip] = excerpt or str(row.prompt)
    req = {json.loads(l)["clip_id"]: json.loads(l) for l in (mf.STAGE2 / "stage2_requests.jsonl").read_text().splitlines() if l.strip()}
    t, _, _ = mf.D.teacher("A")
    tid = {c: i for i, c in enumerate(t["ids"])}
    with tempfile.TemporaryDirectory() as tmp:
        def save(fig, name):
            for legend in fig.legends:                  # below the x-axis labels (tight bounding box)
                legend.set_bbox_to_anchor((0.5, -0.02), transform=fig.transFigure)
            fig.savefig(Path(tmp) / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
        mf.save = save
        mf.draw_stage2_cases([row], "stage2_sample", folder, req, t, tid, label_ids=False)
        Image.open(Path(tmp) / "stage2_sample.png").convert("RGB").save(OUT / "images" / "stage2_sample.webp", quality=88, method=6)
    print("  stage2_sample.webp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, type=Path, help="folder holding analysis/artifacts (test_partA, hero, stage2)")
    ap.add_argument("--stage2-run", required=True, type=Path, help="stage2 run folder (stage2_requests.jsonl, videos, bridge)")
    ap.add_argument("--figures", required=True, type=Path, help="manuscript/figures")
    ap.add_argument("--figure-code", required=True, type=Path, help="manuscript figure_code (make_figures.py)")
    ap.add_argument("--stage2-clip", required=True, help="clip id of the Stage-2 sample shown on the page")
    ap.add_argument("--prompt-excerpt", help="verbatim excerpt of the prompt shown above the frames (default: full prompt)")
    ap.add_argument("--only", choices=["figures", "commands", "stage2"], help="build one group only")
    args = ap.parse_args()
    if args.only in (None, "figures"):
        figures(args.figures)
    if args.only in (None, "commands"):
        command_videos(args.stage1)
    if args.only in (None, "stage2"):
        stage2_figure(args.figure_code, args.stage2_clip, args.prompt_excerpt)
        stage2_video(args.stage1, args.stage2_run, args.stage2_clip)


if __name__ == "__main__":
    main()
