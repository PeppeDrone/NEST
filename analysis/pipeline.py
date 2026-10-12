"""Stage-1 NEST analysis pipeline (validation sweeps, command tests, baselines, kinematics, strips).

    source scripts/env.sh
    $PY analysis/pipeline.py analysis/configs/<config>.yaml [--only MODEL_ID_SUBSTR ...] [--force]

Every model is evaluated stage by stage into
    <output_dir>/<model_id>/<stage>/   with a done.json cache key
so re-running a config recomputes only what changed (settings, checkpoint bytes, code version).
Model ids: "<run>@<step>" for EMA snapshots, "baseline_static", "baseline_retrieval_<source>".
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]                     # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from neomotion.common.clip_text import FrozenCLIPTextConditioner          # noqa: E402
from neomotion.common.topology import BODY_EDGES, BODY_JOINT_NAMES, LIMB_ANGLE_INDICES  # noqa: E402
from neomotion import data as D                                            # noqa: E402
from neomotion.conditioning import parse_prompt, encode_prompt_table, prompt_batch      # noqa: E402
from neomotion.evaluation import trajectory_metrics, patient_summary       # noqa: E402
from neomotion.inference import load_generator, generate_batch, per_clip_noise          # noqa: E402
from neomotion.motion import control_class                                 # noqa: E402
from neomotion.retrieval import build_library, retrieve_batch              # noqa: E402
from neomotion.runner import counterfactual_metrics, summarize_commands    # noqa: E402
import kinematics as K                                                     # noqa: E402

PIPELINE_VERSION = 2      # bump when generation or metric semantics change (invalidates caches)
CAPTION_TYPES = ["still/none", "1 limb", "2 limbs", "3-4 limbs"]
DEFAULT_MANIFEST = str(ROOT / "data" / "dwpose_woface_62_v1" / "teacher_training_manifest.jsonl")


# ============================================================================ helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path, pattern="*.py"):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob(pattern)):
        h.update(str(p.relative_to(path)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def key_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def caption_type(prompt):
    return CAPTION_TYPES[control_class(parse_prompt(prompt)[0])[0]]


def command_set():
    commands = {"still": "the newborn is lying still"}
    for limb in LIMB_ANGLE_INDICES:
        for amplitude, speed in (("low", "slow"), ("high", "slow"), ("high", "fast")):
            commands[f"{limb}_{amplitude}_{speed}"] = (f"the newborn moves its {limb.replace('_', ' ')} "
                                                       f"with {amplitude} amplitude and {speed} speed")
    return commands


def deterministic_backends():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def to_np_str(values):
    return np.array([str(v) for v in values], dtype=str)


# ============================================================================ model resolution
def run_prior_steps(run_dir):
    return int(json.loads((run_dir / "resolved_config.json").read_text())["prior_steps"])


def resolve_models(cfg):
    """Expand config 'models' entries into concrete specs {id, run, step, checkpoint, weights}."""
    specs = []
    selection = None
    for entry in cfg.get("models", []):
        runs = sorted(p.name for p in (ROOT / "runs").iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, entry["runs"]))
        for run in runs:
            run_dir = ROOT / "runs" / run
            snaps = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "ema_snapshots").glob("step_*.pt"))
            want = entry.get("steps", "post_prior")
            if want == "post_prior":
                steps = [s for s in snaps if s > run_prior_steps(run_dir)]
            elif want == "selected":
                if selection is None:
                    selection = json.loads((ROOT / cfg["selection_file"]).read_text())
                if run not in selection:
                    raise SystemExit(f"{run} has no selected step in {cfg['selection_file']}")
                steps = [selection[run]["step"]]
            elif isinstance(want, list):
                steps = [s for s in want if s in snaps]
            else:
                raise SystemExit(f"Unknown steps selector {want!r}")
            for step in steps:
                specs.append({"id": f"{run}@{step:06d}", "run": run, "step": step, "kind": "model",
                              "checkpoint": run_dir / "ema_snapshots" / f"step_{step:06d}.pt", "weights": "ema"})
    for b in cfg.get("baselines", []):
        specs.append({"id": f"baseline_{b}", "kind": "baseline", "baseline": b})
    return specs


# ============================================================================ context
class Context:
    def __init__(self, cfg, force):
        self.cfg, self.force = cfg, force
        self.out = (ROOT / cfg["output_dir"]).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = Path(cfg.get("manifest", DEFAULT_MANIFEST)).resolve()
        self.split_file = (ROOT / cfg.get("split_file", "data_refs/patient_split_v1.json")).resolve()
        self.device = cfg.get("device", "cuda")
        self.code = {"pipeline_version": PIPELINE_VERSION, "neomotion_sha256": sha256_tree(ROOT / "src" / "neomotion"),
                     "kinematics_sha256": sha256_file(ROOT / "analysis" / "kinematics.py")}
        self.records = {r["clip_id"]: r for r in map(json.loads, self.manifest.open()) if r.get("status") == "ok"}
        self._splits, self._table, self._library = {}, None, {}

    # ---- data
    def split(self, source, which=None):
        which = which or self.cfg["split"]
        if (source, which) not in self._splits:
            self._splits[source, which] = D.load_split(self.manifest, source, which)
        return self._splits[source, which]

    def table(self):
        """One CLIP/control table for every prompt this config can need (both sources + commands)."""
        if self._table is None:
            prompts = set(command_set().values())
            for source in ("manual", "automatic"):
                prompts |= set(self.split(source)["prompts"])
            encoder = FrozenCLIPTextConditioner(local_files_only=True)
            # batch_size=1: each prompt's CLIP embedding is independent of which other prompts are
            # encoded with it (batched GPU matmuls differ in the last bits, which can flip near-tied
            # strict command-test comparisons).
            self._table = encode_prompt_table(sorted(prompts), encoder, self.device, batch_size=1)
            del encoder
            torch.cuda.empty_cache()
        return self._table

    def library(self, source):
        if source not in self._library:
            donors = D.load_split(self.manifest, source, "train")
            self._library[source] = build_library(donors)
        return self._library[source]


# ============================================================================ generators
class ModelGenerator:
    def __init__(self, ctx, spec):
        self.ctx, self.spec = ctx, spec
        self.model, self.scales, self.mcfg, payload = load_generator(spec["checkpoint"], ctx.device, spec["weights"])
        self.source = payload["provenance"]["prompt_source"]
        self.identity = {"checkpoint": str(spec["checkpoint"]), "checkpoint_sha256": sha256_file(spec["checkpoint"]),
                         "step": int(payload["step"]), "weights": payload["loaded_weights"], "source": self.source}

    def __call__(self, data, idx, prompts, seed, cfg_scale, steps):
        sub = D.tensor_batch(data, list(idx)); sub["prompts"] = list(prompts)
        preds, bs = [], self.mcfg.batch_size
        for start in range(0, len(idx), bs):
            batch = D.tensor_batch(sub, list(range(start, min(start + bs, len(idx)))), self.ctx.device)
            text = prompt_batch(self.ctx.table(), batch["prompts"], self.ctx.device)
            noise = per_clip_noise(batch["ids"], seed, self.ctx.device)
            pred = generate_batch(self.model, self.scales, batch, text, noise, self.mcfg.diffusion_steps, steps, cfg_scale)
            preds.append({"xy_norm": pred["xy_norm"].float().cpu(), "render_visible": pred["render_visible"].cpu()})
        return {k: torch.cat([p[k] for p in preds]) for k in preds[0]}

    def close(self):
        del self.model
        torch.cuda.empty_cache()


class BaselineGenerator:
    """'static' copies frame 0; 'retrieval_<source>' transfers a label-compatible training clip."""
    def __init__(self, ctx, spec):
        self.ctx, self.kind = ctx, spec["baseline"]
        self.source = self.kind.split("_", 1)[1] if self.kind.startswith("retrieval_") else "manual"
        self.identity = {"baseline": self.kind, "source": self.source}

    def __call__(self, data, idx, prompts, seed, cfg_scale, steps):
        sub = D.tensor_batch(data, list(idx)); sub["prompts"] = list(prompts)
        if self.kind == "static":
            return {"xy_norm": sub["pose0"][:, None].expand(-1, 20, -1, -1).clone(),
                    "render_visible": sub["visible0"][:, None].expand(-1, 20, -1).clone()}
        pred, _ = retrieve_batch(self.ctx.library(self.source), sub, seed)
        return {"xy_norm": pred["xy_norm"].float(), "render_visible": pred["render_visible"]}

    def close(self):
        pass


# ============================================================================ stages
def _cached(out, key, force):
    done = out / "done.json"
    return (not force) and done.exists() and json.loads(done.read_text()).get("key") == key


def _done(out, key, info):
    (out / "done.json").write_text(json.dumps({"key": key, "finished": dt.datetime.now().isoformat(timespec="seconds"),
                                               "pipeline_sha256": sha256_file(__file__), **info}, indent=2, default=str))


def stage_validate(ctx, spec, gen, s):
    out = ctx.out / spec["id"] / "validate"; out.mkdir(parents=True, exist_ok=True)
    key = key_of({"stage": "validate", "s": s, "split": ctx.cfg["split"], "gen": gen.identity, **ctx.code})
    if _cached(out, key, ctx.force):
        return "cached"
    data = ctx.split(gen.source)
    idx = list(range(len(data["ids"])))
    full = D.tensor_batch(data, idx)
    rows = []
    for seed in s["seeds"]:
        pred = gen(data, idx, data["prompts"], seed, s["cfg_scale"], s["inference_steps"])
        rows += [dict(r, seed=seed) for r in trajectory_metrics(pred, full)]
        np.savez_compressed(out / f"trajectories_seed{seed}.npz", ids=to_np_str(data["ids"]),
                            prompts=to_np_str(data["prompts"]), xy_norm=pred["xy_norm"].numpy().astype(np.float32),
                            render_visible=pred["render_visible"].numpy())
    df = pd.DataFrame(rows)
    df["prompt"] = df["clip_id"].map(dict(zip(data["ids"], data["prompts"])))
    df["caption_type_manual"] = df["clip_id"].map(lambda c: caption_type(ctx.records[c]["manual_prompt"]))
    df.to_csv(out / "per_clip.csv", index=False)
    summary = patient_summary([{k: v for k, v in r.items()} for r in rows])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _done(out, key, {"gen": gen.identity, "settings": s})
    return f"{len(df)} rows"


def stage_commands(ctx, spec, gen, s):
    out = ctx.out / spec["id"] / "commands"; (out / "trajectories").mkdir(parents=True, exist_ok=True)
    commands = command_set()
    key = key_of({"stage": "commands", "s": s, "commands": commands, "split": ctx.cfg["split"], "gen": gen.identity, **ctx.code})
    if _cached(out, key, ctx.force):
        return "cached"
    data = D.patient_subset(ctx.split(gen.source), s["max_clips"])
    idx = list(range(len(data["ids"])))
    rows = []
    for command, prompt in commands.items():
        full = D.tensor_batch(dict(data, prompts=[prompt] * len(idx)), idx)
        for seed in s["seeds"]:
            pred = gen(data, idx, [prompt] * len(idx), seed, s["cfg_scale"], s["inference_steps"])
            rows += [dict(r, command=command, prompt=prompt, seed=seed) for r in counterfactual_metrics(pred, full)]
            if s.get("save_trajectories", True):
                np.savez_compressed(out / "trajectories" / f"{command}_seed{seed}.npz", ids=to_np_str(data["ids"]),
                                    xy_norm=pred["xy_norm"].numpy().astype(np.float32),
                                    render_visible=pred["render_visible"].numpy())
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    summary = summarize_commands(rows)
    flat = [{"limb": limb, "test": test, "fraction": v["patient_mean_fraction"], "n_pairs": v["n_evaluable_clip_seed_pairs"],
             **{f"patient_{p}": f for p, f in v["by_patient"].items()}}
            for limb, tests in summary.items() for test, v in tests.items()]
    pd.DataFrame(flat).to_csv(out / "summary.csv", index=False)
    _done(out, key, {"gen": gen.identity, "settings": s, "clips": data["ids"]})
    return f"{len(rows)} rows"


def strip_clips(ctx, s):
    clips = pd.read_csv(ctx.out / "teacher" / "clips.csv")
    if s.get("require_motion_caption", True):
        clips = clips[clips["caption_type_manual"] != "still/none"]
    chosen = clips.sort_values("teacher_amplitude", ascending=False).groupby("patient_id", sort=True).head(s["per_patient"])
    return chosen.sort_values(["patient_id", "teacher_amplitude"], ascending=[True, False])


def stage_strips(ctx, spec, gen, s):
    out = ctx.out / spec["id"] / "strips"; out.mkdir(parents=True, exist_ok=True)
    chosen = strip_clips(ctx, s)
    key = key_of({"stage": "strips", "s": s, "clips": chosen["clip_id"].tolist(), "split": ctx.cfg["split"],
                  "gen": gen.identity, **ctx.code})
    if _cached(out, key, ctx.force):
        return "cached"
    data = ctx.split(gen.source)
    index = {c: i for i, c in enumerate(data["ids"])}
    idx = [index[c] for c in chosen["clip_id"]]
    prompts = [data["prompts"][i] for i in idx]
    full = D.tensor_batch(data, idx)
    arrays, rows = {}, []
    for g in s["cfg_scales"]:
        pred = gen(data, idx, prompts, s["seed"], g, s["inference_steps"])
        arrays[f"xy_cfg{g}"] = pred["xy_norm"].numpy().astype(np.float32)
        arrays[f"vis_cfg{g}"] = pred["render_visible"].numpy()
        rows += [dict(r, cfg_scale=g, prompt=p) for r, p in zip(trajectory_metrics(pred, full), prompts)]
    np.savez_compressed(out / "trajectories.npz", ids=to_np_str(chosen["clip_id"]), prompts=to_np_str(prompts),
                        cfg_scales=np.array(s["cfg_scales"], dtype=float), **arrays)
    df = pd.DataFrame(rows); df["amplitude_ratio"] = df["pred_amplitude"] / df["teacher_amplitude"]
    df.to_csv(out / "per_clip.csv", index=False)
    _done(out, key, {"gen": gen.identity, "settings": s})
    return f"{len(idx)} clips"


def teacher_kinematics(ctx):
    path = ctx.out / "teacher" / "kinematics.csv"
    if path.exists():
        return pd.read_csv(path)
    t = np.load(ctx.out / "teacher" / "teacher.npz")
    xy = np.concatenate([t["pose0"][:, None], t["target"]], 1)
    valid = np.concatenate([t["visible0"][:, None], t["valid"]], 1)
    xb = K.body_units(torch.as_tensor(xy), torch.as_tensor(t["pose0"]), torch.as_tensor(t["visible0"]),
                      torch.as_tensor(t["image_size"])).numpy()
    feats = K.limb_features(xb, valid, t["timestamps"])
    df = _feature_frame(feats, list(t["ids"]))
    df.to_csv(path, index=False)
    return df


def _feature_frame(feats, ids):
    rows = []
    for i, clip in enumerate(ids):
        for li, limb in enumerate(K.LIMBS):
            rows.append({"clip_id": clip, "limb": limb, **{f: feats[f][i, li] for f in K.FEATURES}})
    return pd.DataFrame(rows)


def stage_kinematics(ctx, spec, gen, s):
    """CPU: per-limb features of saved validate trajectories; W1/Fréchet to teacher per caption type."""
    src = ctx.out / spec["id"] / "validate"
    out = ctx.out / spec["id"] / "kinematics"; out.mkdir(parents=True, exist_ok=True)
    vdone = json.loads((src / "done.json").read_text())["key"]
    key = key_of({"stage": "kinematics", "validate": vdone, **ctx.code})
    if _cached(out, key, ctx.force):
        return "cached"
    t = np.load(ctx.out / "teacher" / "teacher.npz")
    tindex = {c: i for i, c in enumerate(t["ids"])}
    teacher = teacher_kinematics(ctx)
    clips = pd.read_csv(ctx.out / "teacher" / "clips.csv").set_index("clip_id")
    frames = []
    for f in sorted(src.glob("trajectories_seed*.npz")):
        z = np.load(f)
        ti = [tindex[c] for c in z["ids"]]
        valid = np.concatenate([t["visible0"][ti][:, None], t["valid"][ti]], 1)       # teacher support
        xb = K.body_units(torch.as_tensor(z["xy_norm"]), torch.as_tensor(t["pose0"][ti]),
                          torch.as_tensor(t["visible0"][ti]), torch.as_tensor(t["image_size"][ti])).numpy()
        df = _feature_frame(K.limb_features(xb, valid, t["timestamps"][ti]), list(z["ids"]))
        df["seed"] = int(f.stem.split("seed")[1])
        frames.append(df)
    pred = pd.concat(frames, ignore_index=True)
    pred["caption_type_manual"] = pred["clip_id"].map(clips["caption_type_manual"])
    pred.to_csv(out / "per_clip_limb.csv", index=False)
    tk = teacher.copy(); tk["caption_type_manual"] = tk["clip_id"].map(clips["caption_type_manual"])
    rows = []
    for ct in CAPTION_TYPES + ["all"]:
        p = pred if ct == "all" else pred[pred.caption_type_manual == ct]
        q = tk if ct == "all" else tk[tk.caption_type_manual == ct]
        row = {"caption_type": ct, "n_pred": len(p), "n_teacher": len(q),
               "frechet": K.frechet(p[list(K.FEATURES)].values, q[list(K.FEATURES)].values)}
        for feat in K.FEATURES:
            row[f"w1_{feat}"] = K.w1(p[feat], q[feat])
            row[f"pred_median_{feat}"] = float(np.nanmedian(p[feat])) if len(p) else np.nan
            row[f"teacher_median_{feat}"] = float(np.nanmedian(q[feat])) if len(q) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    _done(out, key, {"gen": gen.identity if gen else None})
    return "ok"


# ============================================================================ teacher + tables
def export_teacher(ctx):
    out = ctx.out / "teacher"; out.mkdir(exist_ok=True)
    key = key_of({"split": ctx.cfg["split"], "manifest": sha256_file(ctx.manifest), **ctx.code})
    if _cached(out, key, False):
        return
    data = ctx.split("manual")
    ids = data["ids"]
    static = {"xy_norm": data["pose0"][:, None].expand(-1, 20, -1, -1), "render_visible": data["visible0"][:, None].expand(-1, 20, -1)}
    rows = trajectory_metrics(static, data)
    pd.DataFrame({
        "clip_id": ids, "patient_id": data["patients"],
        "manual_prompt": [ctx.records[c]["manual_prompt"] for c in ids],
        "automatic_prompt": [ctx.records[c]["automatic_prompt"] for c in ids],
        "caption_type_manual": [caption_type(ctx.records[c]["manual_prompt"]) for c in ids],
        "caption_type_automatic": [caption_type(ctx.records[c]["automatic_prompt"]) for c in ids],
        "teacher_amplitude": [r["teacher_amplitude"] for r in rows]}).to_csv(out / "clips.csv", index=False)
    np.savez_compressed(out / "teacher.npz", ids=to_np_str(ids), patients=to_np_str(data["patients"]),
                        pose0=data["pose0"].numpy(), visible0=data["visible0"].numpy(), target=data["target"].numpy(),
                        valid=data["valid"].numpy(), timestamps=data["timestamps"].numpy(), image_size=data["image_size"].numpy(),
                        joint_names=to_np_str(BODY_JOINT_NAMES), edges=np.array(BODY_EDGES))
    (out / "kinematics.csv").unlink(missing_ok=True)
    _done(out, key, {"n_clips": len(ids)})


def collect_tables(ctx, specs):
    """Long-format CSVs over all models of this config (inputs for figures and the selection rule)."""
    out = ctx.out / "tables"; out.mkdir(exist_ok=True)
    val, by_type, cmd, kin = [], [], [], []
    for spec in specs:
        base = {"model_id": spec["id"], "run": spec.get("run", spec["id"]), "step": spec.get("step", -1)}
        v = ctx.out / spec["id"] / "validate"
        if (v / "per_clip.csv").exists():
            df = pd.read_csv(v / "per_clip.csv")
            pm = json.loads((v / "summary.json").read_text())["patient_mean"]
            val.append({**base, **{k: pm[k] for k in pm if isinstance(pm[k], (int, float))},
                        "amplitude_ratio": pm["pred_amplitude"] / pm["teacher_amplitude"],
                        "corr_pred_teacher_amplitude": df["pred_amplitude"].corr(df["teacher_amplitude"])})
            for ct in CAPTION_TYPES:
                g = df[df.caption_type_manual == ct]
                by_type.append({**base, "caption_type": ct, "n": len(g),
                                "amplitude_ratio": g.pred_amplitude.mean() / g.teacher_amplitude.mean(),
                                "body_scale_error": g.body_scale_error.mean()})
        c = ctx.out / spec["id"] / "commands" / "summary.csv"
        if c.exists():
            cmd += [{**base, **r} for r in pd.read_csv(c)[["limb", "test", "fraction", "n_pairs"]].to_dict("records")]
        k = ctx.out / spec["id"] / "kinematics" / "summary.csv"
        if k.exists():
            kin += [{**base, **r} for r in pd.read_csv(k).to_dict("records")]
    for name, rows in (("validate", val), ("by_caption_type", by_type), ("commands", cmd), ("kinematics", kin)):
        if rows:
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(f"tables -> {out}")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--only", nargs="+", help="Evaluate only model ids containing any of these substrings")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    deterministic_backends()
    ctx = Context(cfg, args.force)
    specs = resolve_models(cfg)
    if args.only:
        specs = [s for s in specs if any(o in s["id"] for o in args.only)]
    stages = cfg["stages"]
    started = time.monotonic()
    export_teacher(ctx)
    print(f"{cfg['name']}: {len(specs)} models, stages {list(stages)}", flush=True)
    for n, spec in enumerate(specs, 1):
        gpu_stages = [st for st in ("validate", "commands", "strips") if st in stages]
        gen = (ModelGenerator(ctx, spec) if spec["kind"] == "model" else BaselineGenerator(ctx, spec)) if gpu_stages else None
        msgs = []
        for st in gpu_stages:
            msgs.append(f"{st}: " + {"validate": stage_validate, "commands": stage_commands, "strips": stage_strips}[st](ctx, spec, gen, stages[st]))
        if "kinematics" in stages and (ctx.out / spec["id"] / "validate" / "done.json").exists():
            msgs.append("kinematics: " + stage_kinematics(ctx, spec, gen, stages["kinematics"]))
        if gen is not None:
            gen.close()
        print(f"[{n}/{len(specs)}] {spec['id']}: " + "; ".join(msgs), flush=True)
    collect_tables(ctx, specs)
    prov = ctx.out / "provenance"; prov.mkdir(exist_ok=True)
    try:
        git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = None
    (prov / f"run_{dt.datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps({
        "argv": sys.argv, "config": cfg, "code": ctx.code, "pipeline_sha256": sha256_file(__file__), "git": git,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "models": [{k: str(v) for k, v in s.items()} for s in specs],
        "elapsed_seconds": round(time.monotonic() - started, 1)}, indent=2))
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
