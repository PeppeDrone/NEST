"""Generated-trajectory diagnostics on teacher-defined support; patient aggregation."""
from collections import defaultdict
import math

import numpy as np
import torch

from .common.topology import BODY_EDGES
from .conditioning import prompt_batch
from .data import tensor_batch
from .inference import generate_batch, per_clip_noise
from .representation import reference_frame


def trajectory_metrics(prediction, batch):
    """No predicted-visibility filtering in coordinate/geometry errors."""
    xy = prediction["xy_norm"].float()
    target = torch.cat((batch["pose0"][:, None], batch["target"]), 1)
    valid = torch.cat((batch["visible0"][:, None], batch["valid"]), 1)
    frame = reference_frame(batch["pose0"], batch["visible0"], batch["image_size"])
    pixel_factor = batch["image_size"][:, None, None] / frame["scale"][:, None, None, None]
    pixel_error = torch.linalg.vector_norm((xy-target)*pixel_factor, dim=-1)
    image_error = torch.linalg.vector_norm(xy-target, dim=-1)
    dt = torch.diff(batch["timestamps"], dim=1)[:, :, None]
    pred_velocity = torch.diff(xy, dim=1)*pixel_factor/dt[...,None]
    true_velocity = torch.diff(target, dim=1)*pixel_factor/dt[...,None]
    pv = torch.linalg.vector_norm(pred_velocity, dim=-1)
    tv = torch.linalg.vector_norm(true_velocity, dim=-1)
    acceleration_dt = (dt[:,1:]+dt[:,:-1])/2
    pa = torch.linalg.vector_norm(torch.diff(pred_velocity,dim=1),dim=-1)/acceleration_dt
    ta = torch.linalg.vector_norm(torch.diff(true_velocity,dim=1),dim=-1)/acceleration_dt
    transition_mask = valid[:, 1:] & valid[:, :-1]
    acceleration_mask = transition_mask[:,1:] & transition_mask[:,:-1]
    observed_support = valid[:, 1:] & batch["visible0"][:, None]
    amplitude_p = torch.linalg.vector_norm((xy[:, 1:]-xy[:, :1])*pixel_factor, dim=-1)
    amplitude_t = torch.linalg.vector_norm((target[:, 1:]-target[:, :1])*pixel_factor, dim=-1)
    rows = []
    for i, clip in enumerate(batch["ids"]):
        m = batch["valid"][i]
        if not m.any():
            raise ValueError(f"No valid future teacher joints for {clip}")
        if not torch.isfinite(xy[i]).all():
            raise FloatingPointError(f"Nonfinite generated trajectory: {clip}")
        observed = observed_support[i]
        common = transition_mask[i]
        start = common[0]
        frame_error = float(pixel_error[i, 1:][m].mean())
        a_p = float(amplitude_p[i][observed].mean()) if observed.any() else 0.
        a_t = float(amplitude_t[i][observed].mean()) if observed.any() else 0.
        # A symmetric regularized log ratio punishes both under- and over-motion.
        amplitude_error = abs(math.log((a_p+.02)/(a_t+.02)))
        speed_error = float((pv[i]-tv[i]).abs()[common].mean()) if common.any() else 0.
        first_error = float((pv[i, 0]-tv[i, 0]).abs()[start].mean()) if start.any() else 0.
        outside = float((((xy[i, 1:] < 0) | (xy[i, 1:] > 1)).any(-1))[m].float().mean())
        coord_static = torch.linalg.vector_norm((target[i, 1:]-target[i, :1])*pixel_factor[i], dim=-1)
        lengths = []
        for a, b in BODY_EDGES:
            edge_mask = m[:, a] & m[:, b]
            if edge_mask.any():
                pred_length = torch.linalg.vector_norm((xy[i, 1:, a]-xy[i, 1:, b])*pixel_factor[i, 0, 0], dim=-1)
                true_length = torch.linalg.vector_norm((target[i, 1:, a]-target[i, 1:, b])*pixel_factor[i, 0, 0], dim=-1)
                lengths.append(float((pred_length-true_length).abs()[edge_mask].mean()))
        geometry = float(np.mean(lengths)) if lengths else 0.
        predicted_visible = prediction["render_visible"][i, 1:]
        tp = (predicted_visible & m).sum()
        visibility_f1 = float(2*tp/(predicted_visible.sum()+m.sum()).clamp_min(1))
        score = frame_error + .1*amplitude_error + .1*speed_error + .1*first_error + .1*geometry + outside + .1*(1-visibility_f1)
        rows.append({"clip_id": clip, "patient_id": batch["patients"][i], "selection_score": score,
                     "body_scale_error": frame_error, "image_nme": float(image_error[i, 1:][m].mean()),
                     "amplitude_log_error": amplitude_error, "pred_amplitude": a_p, "teacher_amplitude": a_t,
                     "speed_error": speed_error, "first_step_speed_error": first_error, "projected_geometry_error": geometry,
                     "pred_acceleration": float(pa[i][acceleration_mask[i]].mean()) if acceleration_mask[i].any() else None,
                     "teacher_acceleration": float(ta[i][acceleration_mask[i]].mean()) if acceleration_mask[i].any() else None,
                     "acceleration_error": float((pa[i]-ta[i]).abs()[acceleration_mask[i]].mean()) if acceleration_mask[i].any() else None,
                     "outside_fraction": outside, "visibility_recall": float(prediction["render_visible"][i, 1:][m].float().mean()),
                     "visibility_f1": visibility_f1,
                     "body_scale_error_observed_start": float(pixel_error[i,1:][observed].mean()) if observed.any() else None,
                     "static_body_scale_error": float(coord_static[observed].mean()) if observed.any() else None,
                     "observed_start_fraction": float(batch["visible0"][i].float().mean())})
    return rows


def patient_summary(rows):
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation rows")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row)
    fields = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "seed"})
    def mean_present(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    by_patient = {p: {k: mean_present([r.get(k) for r in rs]) for k in fields}
                  for p, rs in grouped.items()}
    return {"patient_mean": {k: mean_present([r[k] for r in by_patient.values()]) for k in fields},
            "by_patient": by_patient, "n_patients": len(grouped), "n_clip_seed_pairs": len(rows)}


@torch.inference_mode()
def evaluate_model(model, scales, data, text_table, seeds, batch_size, diffusion_steps, inference_steps, cfg_scale, force_unconditional=False):
    device = next(model.parameters()).device
    rows, example_parts, first_predictions = [], [], {}
    # One example per patient before taking a second from any patient.
    example_indices = []
    for patient in dict.fromkeys(data["patients"]):
        example_indices.append(data["patients"].index(patient))
    example_indices = set(example_indices[:4])
    for seed in seeds:
        for start in range(0, len(data["ids"]), batch_size):
            batch = tensor_batch(data, list(range(start, min(start+batch_size, len(data["ids"])))), device)
            text = prompt_batch(text_table, batch["prompts"], device)
            pred = generate_batch(model, scales, batch, text, per_clip_noise(batch["ids"], seed, device),
                                  diffusion_steps, inference_steps, cfg_scale, force_unconditional)
            metrics = trajectory_metrics(pred, batch)
            if seed == seeds[0]:
                first_predictions[start] = pred["xy_norm"].cpu()
                for row in metrics:
                    row["diversity_from_first_seed"] = None
            else:
                frame = reference_frame(batch["pose0"],batch["visible0"],batch["image_size"])
                factor = batch["image_size"][:,None,None]/frame["scale"][:,None,None,None]
                difference = torch.linalg.vector_norm((pred["xy_norm"][:,1:]-first_predictions[start].to(device)[:,1:])*factor,dim=-1)
                for i,row in enumerate(metrics):
                    row["diversity_from_first_seed"] = float(difference[i][batch["valid"][i]].mean())
            rows.extend(dict(r, seed=int(seed)) for r in metrics)
            selected = [i for i in range(len(batch["ids"])) if start+i in example_indices]
            if seed == seeds[0] and selected:
                part = {k: v[selected].detach().cpu() for k, v in pred.items()}
                part.update(ids=[batch["ids"][i] for i in selected],
                    teacher=torch.cat((batch["pose0"][:,None],batch["target"]),1)[selected].cpu(),
                    teacher_visible=torch.cat((batch["visible0"][:,None],batch["valid"]),1)[selected].cpu())
                example_parts.append(part)
    examples = {k:torch.cat([p[k] for p in example_parts]) if isinstance(example_parts[0][k],torch.Tensor)
                else sum([p[k] for p in example_parts],[]) for k in example_parts[0]}
    return patient_summary(rows), rows, examples


def plot_examples(examples, max_clips=4):
    """Return a skeleton figure; caller owns saving. Geometry plotted in pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count = min(max_clips, len(examples["ids"]))
    fig, axes = plt.subplots(count*2, 5, figsize=(12, 3*count), squeeze=False)
    for c in range(count):
        width, height = examples["image_size"][c].tolist()
        for row, (key, mask_key) in enumerate([("teacher", "teacher_visible"), ("xy_norm", "render_visible")]):
            xy = examples[key][c].numpy()*np.array([width, height])
            visible = examples[mask_key][c].numpy()
            for col, t in enumerate([0, 1, 5, 10, 19]):
                ax = axes[c*2+row, col]
                for a,b in BODY_EDGES:
                    if visible[t,a] and visible[t,b]:
                        ax.plot(xy[t,[a,b],0],xy[t,[a,b],1], lw=1.2, color="#303030" if row==0 else "#287f9c")
                ax.set(xlim=(0,width), ylim=(height,0), aspect="equal", xticks=[], yticks=[])
                if col == 0:
                    ax.set_ylabel(examples["ids"][c]+("\nteacher" if row==0 else "\nprediction"), fontsize=8)
                if row == 0:
                    ax.set_title(f"t={float(examples['timestamps'][c,t]):.2f}s")
    fig.tight_layout()
    return fig
