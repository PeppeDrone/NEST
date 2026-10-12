"""Evaluation file orchestration, separate from the numerical metrics."""
from pathlib import Path
import json

import numpy as np
import torch

from .common.clip_text import FrozenCLIPTextConditioner
from .common.topology import LIMB_ANGLE_INDICES
from .conditioning import encode_prompt_table, prompt_batch
from .data import load_split, verify_split, patient_subset, tensor_batch, file_sha256
from .evaluation import evaluate_model, plot_examples, patient_summary, trajectory_metrics
from .inference import load_generator, generate_batch, per_clip_noise
from .representation import reference_frame
from .retrieval import build_library, retrieve_batch
from .training import save_json, append_jsonl, source_hash


@torch.inference_mode()
def counterfactual_metrics(pred, batch):
    frame = reference_frame(batch["pose0"], batch["visible0"], batch["image_size"])
    factor = batch["image_size"][:, None, None]/frame["scale"][:, None, None, None]
    displacement = torch.linalg.vector_norm((pred["xy_norm"][:, 1:]-batch["pose0"][:, None])*factor, dim=-1)
    # Interior speed excludes a first-step offset being rewarded as articulation.
    dt = torch.diff(batch["timestamps"][:, 1:], dim=1)[:, :, None]
    speed = torch.linalg.vector_norm(torch.diff(pred["xy_norm"][:, 1:], dim=1)*factor, dim=-1)/dt
    rows = []
    for i, clip in enumerate(batch["ids"]):
        values = {"clip_id": clip, "patient_id": batch["patients"][i]}
        for limb, indices in LIMB_ANGLE_INDICES.items():
            known = [j for j in indices if bool(batch["visible0"][i, j])]
            values[f"{limb}_amplitude"] = float(displacement[i, :, known].mean()) if known else None
            values[f"{limb}_speed"] = float(speed[i, :, known].mean()) if known else None
        values["outside_fraction"] = float(((pred["xy_norm"][i, 1:] < 0) | (pred["xy_norm"][i, 1:] > 1)).any(-1).float().mean())
        rows.append(values)
    return rows


def summarize_commands(rows):
    lookup = {(r["clip_id"], r["seed"], r["command"]): r for r in rows}
    output = {}
    for limb in LIMB_ANGLE_INDICES:
        tests = {"amplitude_order": [], "speed_order": [], "moves_more_than_still": [], "specificity": []}
        for r in rows:
            if r["command"] != "still":
                continue
            key = (r["clip_id"], r["seed"])
            low = lookup[key + (f"{limb}_low_slow",)]
            high = lookup[key + (f"{limb}_high_slow",)]
            fast = lookup[key + (f"{limb}_high_fast",)]
            amp, speed = f"{limb}_amplitude", f"{limb}_speed"
            if r[amp] is None:
                continue
            tests["amplitude_order"].append((r["patient_id"], high[amp] > low[amp]))
            tests["speed_order"].append((r["patient_id"], fast[speed] > high[speed]))
            tests["moves_more_than_still"].append((r["patient_id"], fast[speed] > r[speed]))
            gains = [(fast[f"{other}_speed"]+.01)/(r[f"{other}_speed"]+.01)
                     for other in LIMB_ANGLE_INDICES if other != limb and r[f"{other}_speed"] is not None]
            gain = (fast[speed]+.01)/(r[speed]+.01)
            if gains:
                tests["specificity"].append((r["patient_id"], gain > max(gains) and gain > 1))
        output[limb] = {}
        for name, pairs in tests.items():
            patients = sorted({p for p, v in pairs})
            fractions = {p: float(np.mean([v for q,v in pairs if p == q])) for p in patients}
            output[limb][name] = {"n_evaluable_clip_seed_pairs": len(pairs), "by_patient": fractions,
                                "patient_mean_fraction": float(np.mean(list(fractions.values()))) if fractions else None}
    return output


def evaluate_checkpoint(checkpoint, manifest, split_path, output, split="val", weights="selected", device="cuda",
                        max_clips=None, counterfactual=False, local_files_only=False, steps=None, cfg_scale=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new evaluation directory: {output}")
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be positive")
    data_identity = verify_split(manifest, split_path)
    model, scales, cfg, payload = load_generator(checkpoint, device, weights)
    for key in ("teacher_manifest_sha256", "patient_split_sha256"):
        if data_identity[key] != payload["provenance"][key]:
            raise ValueError(f"Evaluation dataset changed: {key}")
    source = payload["provenance"]["prompt_source"]
    data = patient_subset(load_split(manifest, source, split), max_clips)
    commands = {"still": "the newborn is lying still"}
    for limb in LIMB_ANGLE_INDICES:
        for amplitude, speed in (("low", "slow"), ("high", "slow"), ("high", "fast")):
            commands[f"{limb}_{amplitude}_{speed}"] = f"the newborn moves its {limb.replace('_',' ')} with {amplitude} amplitude and {speed} speed"
    encoder = FrozenCLIPTextConditioner(local_files_only=local_files_only)
    table = encode_prompt_table(data["prompts"] + (list(commands.values()) if counterfactual else []), encoder, device)
    del encoder
    output.mkdir(parents=True, exist_ok=True)
    inference_steps = cfg.inference_steps if steps is None else steps
    guidance = cfg.cfg_scale if cfg_scale is None else cfg_scale
    summary, rows, examples = evaluate_model(model, scales, data, table, cfg.validation_seeds, cfg.batch_size,
                                           cfg.diffusion_steps, inference_steps, guidance)
    for row in rows:
        append_jsonl(output / "per_clip.jsonl", row)
    np.savez_compressed(output / "examples.npz", **{k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                        for k,v in examples.items()})
    fig = plot_examples(examples)
    fig.savefig(output / "examples.png", dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    # Explicit static baseline on the same metrics/masks; no model visibility filtering.
    static = {"xy_norm": data["pose0"][:,None].expand(-1,20,-1,-1),
              "render_visible": data["visible0"][:,None].expand(-1,20,-1)}
    static_rows = trajectory_metrics(static, data)
    for row in static_rows:
        append_jsonl(output / "static_per_clip.jsonl", row)
    result = {"format": "dmtbaby.stage1.v7.evaluation", "split": split, "clips": len(data["ids"]),
              "weights": payload["loaded_weights"], "prompt_source": source,
              "checkpoint_sha256": file_sha256(checkpoint), "data": data_identity,
              "evaluation_source_sha256": source_hash(), "checkpoint_step": payload["step"],
              "diagnostic_training": bool(payload["provenance"]["smoke_test"] or payload["provenance"]["overfit_clips"]),
              "seeds": list(cfg.validation_seeds), "steps": inference_steps, "cfg_scale": guidance,
              "metrics": summary, "static": patient_summary(static_rows)}
    # Selected-source controls and donors from train only; no annotation lookup.
    donors = load_split(manifest, source, "train")
    allowed = set(payload["provenance"]["training_clip_ids"])
    donors = tensor_batch(donors,[i for i,c in enumerate(donors["ids"]) if c in allowed])
    if set(donors["patients"]) & set(data["patients"]):
        raise ValueError("Retrieval donor patient overlaps evaluation patients")
    library = build_library(donors)
    retrieval_rows = []
    for seed in cfg.validation_seeds:
        pred, metadata = retrieve_batch(library,data,seed)
        np.savez_compressed(output / f"retrieval_seed_{seed}.npz",
            **{k:v.numpy() for k,v in pred.items()}, ids=np.asarray(data["ids"]),
            donor_ids=np.asarray([r["donor_clip_id"] or "" for r in metadata]),
            timestamps=data["timestamps"].numpy(), image_size=data["image_size"].numpy())
        measured = trajectory_metrics(pred,data)
        for row, donor in zip(measured,metadata):
            record = dict(row,seed=seed,**donor)
            append_jsonl(output / "retrieval_per_clip.jsonl",record)
            retrieval_rows.append(record)
    result["retrieval"] = patient_summary(retrieval_rows)
    result["retrieval_note"] = "Training-only selected-prompt codes; strict compatibility; invalid transfers fall back to static copy. Coverage is reported."
    if counterfactual:
        cf_rows = []
        for command, prompt in commands.items():
            modified = dict(data, prompts=[prompt]*len(data["ids"]))
            for seed in cfg.validation_seeds:
                generated = []
                for start in range(0, len(data["ids"]), cfg.batch_size):
                    batch = tensor_batch(modified, range(start, min(start+cfg.batch_size, len(data["ids"]))), device)
                    pred = generate_batch(model, scales, batch, prompt_batch(table, batch["prompts"], device),
                        per_clip_noise(batch["ids"], seed, device), cfg.diffusion_steps, inference_steps, guidance)
                    generated.append({k:v.cpu() for k,v in pred.items()})
                    measured = [dict(row, command=command, prompt=prompt, seed=seed) for row in counterfactual_metrics(pred, batch)]
                    for row in measured:
                        append_jsonl(output / "counterfactual_per_clip.jsonl", row)
                    cf_rows.extend(measured)
                cf_dir = output / "counterfactual_predictions"
                cf_dir.mkdir(exist_ok=True)
                np.savez_compressed(cf_dir / f"{command}_seed_{seed}.npz",
                    **{k:torch.cat([p[k] for p in generated]).numpy() for k in generated[0]},
                    ids=np.asarray(data["ids"]),prompt=np.asarray(prompt),
                    controls=prompt_batch(table,[prompt],"cpu")["controls"].numpy()[0])
            print(f"counterfactual: {command}", flush=True)
        result["counterfactual"] = summarize_commands(cf_rows)
        result["counterfactual_note"] = "Paired attribute diagnostics; no accuracy claim for an unobserved ground-truth counterfactual."
    save_json(output / "summary.json", result)
    return result
