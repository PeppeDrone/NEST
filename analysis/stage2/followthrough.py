"""Stage-2 follow-through: does the generated video follow its Stage-1 trajectory?

    $PY analysis/stage2/followthrough.py extract [--provider cuda]   # DWPose on videos + controls (GPU, after inference)
    $PY analysis/stage2/followthrough.py analyze                     # statistics (CPU)

Inputs (produced by stage2/build_inputs.py and stage2/run_inference.sh in stage2/run/):
  stage2_requests.jsonl, videos/NN_<clip>/output.mp4, bridge/controls/<clip>/ (81 DWPose control images),
  predictions/<clip>.npz (the 20-frame Stage-1 trajectory), shortlist from the bridge-readiness screen.
Extraction runs the production DWPose extractor (dmtbaby.stage2.followthrough) on every video frame; the
control pose is the bridge's exact 81-frame coordinates (bridge/conditioned/<clip>.npz), from which the
control images were drawn.
Measures (body-20, normalized image coordinates, joints visible in both):
  nme, pck_005, bone_relative_error   video pose vs control pose (production metrics)
  disp_error      mean |displacement-from-frame-0 of video - of control| (px of the source image):
                  motion error with any static offset removed
  amp_ratio       mean distal displacement in the video / in the control (motion transferred)
  sel_video/ctrl  labelled-limb / other-limb distal path ratio (captions with moving limbs)
  mismatch        disp_error of each video against the controls of the other test videos; the paired
                  difference matched - mismatched is tested with a Wilcoxon signed-rank test over videos
                  and an infant-level bootstrap (2,000 draws)
  stage-2 cost    error of the video pose vs the real infant's trajectory compared with the error of the
                  control (generated trajectory) vs the real infant: what Stage 2 adds to Stage 1
Output: analysis/artifacts/stage2/{followthrough.json, followthrough_poses/, per_video.csv, summary.json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]                 # repository root
PROJECT = ROOT                                             # external/UniAnimate-DiT (DWPose models) lives here
INF = ROOT / "stage2" / "run"
OUT = ROOT / "analysis/artifacts/stage2"
sys.path.insert(0, str(ROOT / "src"))

DISTAL = {"right_arm": 4, "left_arm": 7, "right_leg": 10, "left_leg": 13}
LIMB_ORDER = ("left_arm", "right_arm", "left_leg", "right_leg")      # conditioning order
CONDITION = "manual_generated"
N_BOOT = 2000


def requests():
    rows = [json.loads(l) for l in (INF / "stage2_requests.jsonl").read_text().splitlines() if l.strip()]
    pred = {json.loads(l)["clip_id"]: json.loads(l) for l in (INF / "prediction_manifest.jsonl").read_text().splitlines() if l.strip()}
    for r in rows:
        r.update({k: pred[r["clip_id"]][k] for k in ("caption_type", "gallery_rank", "stage1_model", "sampling_seed")})
    return rows


def cmd_extract(args):
    """DWPose on every video frame (production extractor); control = the bridge's exact 81-frame coordinates.

    The rendered control images are skeletons on a black background, on which the DWPose person detector
    finds no person, so the control pose is read from bridge/conditioned/<clip>.npz instead (the same
    coordinates the control images were drawn from). Normalized coordinates are comparable because the
    reference still and the control images are both resized to the video frame by UniAnimate.
    """
    from dmtbaby.evaluation import pose_metrics
    from dmtbaby.pose.schema import BODY_RENDER_EDGES
    from dmtbaby.stage2.followthrough import body20_from_frames, decode_stage2_video, make_dwpose_extractor
    OUT.mkdir(parents=True, exist_ok=True)
    clips = pd.read_csv(ROOT / "analysis/artifacts/test_partA/teacher/clips.csv").set_index("clip_id")
    extractor = make_dwpose_extractor(root=PROJECT, provider=args.provider, allow_cpu_smoke=args.provider == "cpu")
    rows = []
    for r in requests()[: args.limit or None]:
        c = r["clip_id"]
        if not Path(r["output"]).is_file():
            print(f"skip (no video yet): {c}"); continue
        pose_npz = OUT / "followthrough_poses" / f"{c}.npz"
        if pose_npz.exists() and not args.force:
            p = np.load(pose_npz); vxy, vvis = p["video_xy"], p["video_visible"].astype(bool)
        else:
            frames = decode_stage2_video(r["output"])
            vxy, vvis = body20_from_frames(extractor, frames)
        cond = np.load(INF / "bridge/conditioned" / f"{c}.npz")
        cxy, cvis = cond["xy_norm"][:, :20].astype(np.float32), cond["render_visible"][:, :20].astype(bool)
        pose_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pose_npz, video_xy=vxy.astype(np.float32), video_visible=vvis, control_xy=cxy, control_visible=cvis)
        m = pose_metrics(vxy, cxy, vvis, cvis, BODY_RENDER_EDGES).as_dict()
        rows.append({"clip_id": c, "patient_id": clips.loc[c, "patient_id"], "state": "ok", "pose_npz": str(pose_npz),
                     "generated_video": r["output"], **{k: (float(v) if v is not None else None) for k, v in m.items()}})
        print(f"{c}: nme={m['nme']:.4f} pck={m['pck_005']:.3f} video-visible={vvis.mean():.2f}", flush=True)
    (OUT / "followthrough.json").write_text(json.dumps({"format": "nest.stage2_followthrough.v1",
                                                        "control_source": "bridge/conditioned (exact control coordinates)",
                                                        "rows": rows}, indent=2))


def labelled_limbs(prompt):
    from neomotion.conditioning import parse_prompt
    codes = parse_prompt(prompt)[0]
    return [l for l, c in zip(LIMB_ORDER, codes) if int(c[0]) == 1 and int(c[1]) == 1]


VIDEO_PX = np.array([480.0, 832.0])          # video width, height: unit of all reported pixel distances


def align_to_video(src_xy, src_vis, vid_xy, vid_vis, frames=3):
    """Per-axis scale and offset mapping a trajectory into the video frame, fitted only on the first frames.

    UniAnimate re-places and rescales the driving skeleton onto the infant of the reference still, so a
    trajectory and the video agree in motion, not in absolute placement. The fit uses frames 0..frames-1
    (where the video should reproduce the reference pose) and is then held fixed, so all later motion
    error is retained.
    """
    out = np.full_like(src_xy, np.nan, dtype=np.float64)
    ok = (src_vis[:frames] & vid_vis[:frames] & np.isfinite(src_xy[:frames]).all(-1) & np.isfinite(vid_xy[:frames]).all(-1))
    if ok.sum() < 4:
        return out
    for k in range(2):
        a, b = np.polyfit(src_xy[:frames][ok][:, k], vid_xy[:frames][ok][:, k], 1)
        out[..., k] = a * src_xy[..., k] + b
    return out


def smooth(xy, vis, window=5):
    """Centered moving median over frames (per joint and axis, visible frames only): removes the
    frame-to-frame jitter of pose estimation on video (about 0.3 s at the bridge's frame rate)."""
    xy = np.asarray(xy, float).copy(); xy[~vis] = np.nan
    out = np.full_like(xy, np.nan); h = window // 2
    for f in range(xy.shape[0]):
        seg = xy[max(0, f - h): f + h + 1]
        with np.errstate(all="ignore"):
            out[f] = np.nanmedian(seg, axis=0) if np.isfinite(seg).any() else np.nan
    out[~vis] = np.nan
    return out


def limb_corr(dv, dc, limbs):
    """Mean Pearson correlation over limbs between distal displacement magnitudes over time (scale-free)."""
    rs = []
    for l in limbs:
        a, b = np.linalg.norm(dv[:, DISTAL[l]], axis=-1), np.linalg.norm(dc[:, DISTAL[l]], axis=-1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 10 and a[ok].std() > 0 and b[ok].std() > 0:
            rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def disp(xy, vis):
    """Displacement from frame 0 in video pixels; NaN where the joint is not visible in that frame and frame 0."""
    d = (np.asarray(xy, float) - np.asarray(xy, float)[:1]) * VIDEO_PX
    ok = vis & vis[:1] & np.isfinite(d).all(-1)
    d[~ok] = np.nan
    return d


def disp_error(a, b):
    e = np.linalg.norm(a - b, axis=-1)
    return float(np.nanmean(e)) if np.isfinite(e).any() else np.nan


def distal_amplitude(d, limbs=None):
    js = [DISTAL[l] for l in (limbs or DISTAL)]
    m = np.linalg.norm(d[:, js], axis=-1)
    return float(np.nanmean(m)) if np.isfinite(m).any() else np.nan


def path_length(xy, vis, j):
    p = np.asarray(xy, float)[:, j] * VIDEO_PX
    ok = vis[:, j] & np.isfinite(p).all(-1)
    if ok.sum() < 2:
        return np.nan
    return float(np.linalg.norm(np.diff(p[ok], axis=0), axis=-1).sum())


def teacher_81(t, i):
    """Real teacher trajectory (20 native frames) resampled to the bridge's 81-frame grid."""
    xy20 = np.concatenate([t["pose0"][i][None], t["target"][i]])
    vis20 = np.concatenate([t["visible0"][i][None], t["valid"][i]]).astype(bool)
    src, dst = np.linspace(0, 1, 20), np.linspace(0, 1, 81)
    xy = np.full((81, 20, 2), np.nan, np.float32); vis = np.zeros((81, 20), bool)
    for j in range(20):
        ok = vis20[:, j]
        if ok.sum() >= 2:
            for k in range(2):
                xy[:, j, k] = np.interp(dst, src[ok], xy20[ok, j, k])
            vis[:, j] = (dst >= src[ok].min()) & (dst <= src[ok].max())
    return xy, vis


def boot_mean(df, col, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[col].dropna().values for _, g in df.groupby("patient_id")]
    vals = []
    for _ in range(n):
        pick = [groups[k] for k in rng.integers(0, len(groups), len(groups))]
        vals.append(np.mean(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in pick if len(g)])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cmd_analyze(args):
    rep = json.loads((OUT / "followthrough.json").read_text())
    ok = [r for r in rep["rows"] if r.get("state") == "ok"]
    meta = {r["clip_id"]: r for r in requests()}
    t = np.load(ROOT / "analysis/artifacts/test_partA/teacher/teacher.npz")
    tindex = {c: i for i, c in enumerate(t["ids"])}
    raw = {}
    for r in ok:
        z = np.load(r["pose_npz"])
        raw[r["clip_id"]] = {"vxy": z["video_xy"].astype(float), "vvis": z["video_visible"].astype(bool),
                             "cxy": z["control_xy"].astype(float), "cvis": z["control_visible"].astype(bool)}
    (OUT / "aligned").mkdir(exist_ok=True)
    rows = []
    for r in ok:
        c = r["clip_id"]; q = raw[c]; m = meta[c]
        txy, tvis = teacher_81(t, tindex[c])
        ctrl = align_to_video(q["cxy"], q["cvis"], q["vxy"], q["vvis"])
        real = align_to_video(txy, tvis, q["vxy"], q["vvis"])
        cvis = q["cvis"] & np.isfinite(ctrl).all(-1); rvis = tvis & np.isfinite(real).all(-1)
        np.savez_compressed(OUT / "aligned" / f"{c}.npz", video_xy=q["vxy"], video_visible=q["vvis"], control_xy=ctrl,
                            control_visible=cvis, real_xy=real, real_visible=rvis)
        vs, cs, rs_ = smooth(q["vxy"], q["vvis"]), smooth(ctrl, cvis), smooth(real, rvis)
        dv, dc, dr = disp(vs, q["vvis"]), disp(cs, cvis), disp(rs_, rvis)
        frame0 = float(np.nanmean(np.linalg.norm((q["vxy"][0] - ctrl[0]) * VIDEO_PX, axis=-1)[q["vvis"][0] & cvis[0]]))
        lim = labelled_limbs(m["prompt"]); others = [l for l in DISTAL if l not in lim]
        sel = {}
        for key, xy, vis in (("video", q["vxy"], q["vvis"]), ("ctrl", ctrl, cvis)):
            if lim and others:
                a = np.nanmean([path_length(xy, vis, DISTAL[l]) for l in lim])
                b = np.nanmean([path_length(xy, vis, DISTAL[l]) for l in others])
                sel[key] = a / b if np.isfinite(b) and b > 0 else np.nan
            else:
                sel[key] = np.nan
        mismatch = []
        for o in raw:
            if o == c:
                continue
            oc = align_to_video(raw[o]["cxy"], raw[o]["cvis"], q["vxy"], q["vvis"])
            ov = raw[o]["cvis"] & np.isfinite(oc).all(-1)
            mismatch.append(disp_error(dv, disp(smooth(oc, ov), ov)))
        rows.append({"clip_id": c, "patient_id": r["patient_id"], "caption_type": m["caption_type"], "prompt": m["prompt"],
                     "gallery_rank": m["gallery_rank"], "frame0_offset": frame0, "video_visible": float(q["vvis"].mean()),
                     "corr_labelled": limb_corr(dv, dc, lim) if lim else np.nan,
                     "corr_labelled_real": limb_corr(dr, dc, lim) if lim else np.nan,
                     "disp_error": disp_error(dv, dc), "disp_error_mismatch": float(np.nanmean(mismatch)),
                     "amp_video": distal_amplitude(dv), "amp_ctrl": distal_amplitude(dc),
                     "sel_video": sel["video"], "sel_ctrl": sel["ctrl"], "labelled": ",".join(lim),
                     "err_video_vs_real": disp_error(dv, dr), "err_ctrl_vs_real": disp_error(dc, dr)})
    df = pd.DataFrame(rows)
    df["amp_ratio"] = df.amp_video / df.amp_ctrl
    df["mismatch_gap"] = df.disp_error_mismatch - df.disp_error
    df["stage2_cost"] = df.err_video_vs_real - df.err_ctrl_vs_real
    df.to_csv(OUT / "per_video.csv", index=False)

    def ms(x):
        x = pd.Series(x).dropna()
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(x.median()), "n": int(len(x))}
    moving = df[df.caption_type != "still/none"]
    summ = {"n_videos": len(df), "n_patients": int(df.patient_id.nunique()),
            "metrics": {k: ms(df[k]) for k in ("frame0_offset", "disp_error", "disp_error_mismatch",
                                               "amp_video", "amp_ctrl", "err_video_vs_real", "err_ctrl_vs_real")},
            "by_caption_type": {ct: {k: ms(g[k]) for k in ("disp_error", "disp_error_mismatch", "amp_video", "amp_ctrl", "amp_ratio", "corr_labelled")}
                                for ct, g in df.groupby("caption_type")},
            "moving": {k: ms(moving[k]) for k in ("disp_error", "disp_error_mismatch", "amp_ratio", "corr_labelled",
                                                  "corr_labelled_real", "sel_video", "sel_ctrl")},
            "noise_floor_still_amp_video": ms(df[df.caption_type == "still/none"].amp_video)}
    tests = {}
    for name, d, a, b in (("matched_vs_mismatched_all", df, "disp_error", "disp_error_mismatch"),
                          ("matched_vs_mismatched_moving", moving, "disp_error", "disp_error_mismatch"),
                          ("video_vs_control_error_to_real", df, "err_video_vs_real", "err_ctrl_vs_real"),
                          ("sel_video_vs_ctrl", moving.dropna(subset=["sel_video", "sel_ctrl"]), "sel_video", "sel_ctrl")):
        x, y = d[a].values, d[b].values
        okm = np.isfinite(x) & np.isfinite(y)
        w = wilcoxon(x[okm], y[okm]) if okm.sum() >= 5 else None
        diff = pd.DataFrame({"patient_id": d.patient_id.values[okm], "diff": (x - y)[okm]})
        tests[name] = {"n": int(okm.sum()), "mean_a": float(np.mean(x[okm])), "sd_a": float(np.std(x[okm], ddof=1)),
                       "mean_b": float(np.mean(y[okm])), "sd_b": float(np.std(y[okm], ddof=1)),
                       "mean_diff": float(diff["diff"].mean()), "ci95_diff": boot_mean(diff, "diff"),
                       "wilcoxon_p": float(w.pvalue) if w else None,
                       "n_a_lower": int((x[okm] < y[okm]).sum())}
    summ["tests"] = tests
    log_amp = pd.DataFrame({"patient_id": moving.patient_id, "x": np.log(moving.amp_ratio)}).dropna()
    summ["log_amp_ratio"] = {"mean": float(log_amp.x.mean()), "ci95": boot_mean(log_amp, "x")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--provider", default="cuda"); e.add_argument("--limit", type=int)
    e.add_argument("--force", action="store_true")
    sub.add_parser("analyze")
    args = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
