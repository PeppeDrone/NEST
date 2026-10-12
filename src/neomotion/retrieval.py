"""Training-only, prompt-matched segment-transfer baseline; never a learned target."""
import hashlib

import torch

from .common.topology import BODY_EDGES
from .conditioning import parse_prompt
from .representation import reference_frame, encode_future, decode_future


def build_library(training):
    """Caller must supply ONLY training donors from the verified frozen split."""
    frame = reference_frame(training["pose0"],training["visible0"],training["image_size"])
    future = encode_future(training["target"],training["valid"],frame)+frame["pose"][:,None]
    return {"ids":training["ids"], "patients":training["patients"], "frame":frame,
            "q":torch.cat((frame["pose"][:,None],future),1),
            "valid":torch.cat((training["visible0"][:,None],training["valid"]),1),
            "timestamps":training["timestamps"]-training["timestamps"][:,:1],
            "codes":torch.stack([parse_prompt(p)[0] for p in training["prompts"]])}


def transfer_segments(query_frame, donor_q, donor_valid):
    """Preserve query bones at frame zero; transfer donor angle/length changes.

    No interpolation, clipping, or fixed projected-bone-length constraint. A
    missing donor edge is held at its starting orientation and marked invisible.
    Unknown query edges use the donor's canonical edge, not a zero sentinel.
    """
    q0 = query_frame["pose"][0]
    seen = query_frame["visible"][0]
    q = q0[None].expand(20,-1,-1).clone()
    valid = torch.zeros(20,20,dtype=torch.bool)
    root_ok = donor_valid[:,1] & donor_valid[0,1] & seen[1]
    q[:,1] = q0[1]+torch.where(root_ok[:,None],donor_q[:,1]-donor_q[0,1],0)
    valid[:,1] = root_ok
    for a,b in BODY_EDGES:
        vector = donor_q[:,b]-donor_q[:,a]
        length = torch.linalg.vector_norm(vector,dim=-1)
        edge_ok = donor_valid[:,a] & donor_valid[:,b]
        if seen[a] and seen[b]:
            base = q0[b]-q0[a]
            start_ok = donor_valid[0,a] & donor_valid[0,b] & (length[0] > .03)
            edge_ok = edge_ok & start_ok
            angle = torch.atan2(vector[:,1],vector[:,0])-torch.atan2(vector[0,1],vector[0,0])
            c,s = angle.cos(),angle.sin()
            moved = torch.stack((c*base[0]-s*base[1],s*base[0]+c*base[1]),-1)
            moved = moved*(length/length[0].clamp_min(.03))[:,None]
            moved = torch.where(edge_ok[:,None],moved,base)
        else:
            moved = torch.where(edge_ok[:,None],vector,0)
        q[:,b] = q[:,a]+moved
        valid[:,b] = edge_ok & valid[:,a]
    xy = decode_future((q-query_frame["pose"][0])[None],query_frame,torch.ones(2,20,2))[0]
    xy[0] = query_frame["input_pose"][0]
    valid[0] = seen
    return {"xy_norm":xy,"render_visible":valid}


def retrieve_batch(library, query, seed, candidates=16):
    """Choose using prompt + starting pose + sampling times only, then transfer.

    Strict known-code compatibility; unsupported/no safe match falls back to
    static copy and is reported. Neither held-out future nor annotations are read.
    """
    frame = reference_frame(query["pose0"],query["visible0"],query["image_size"])
    outputs, metadata = [], []
    for i,prompt in enumerate(query["prompts"]):
        codes,_ = parse_prompt(prompt)
        known = codes[:,0].bool()
        compatible = ((library["codes"][:,:,0] == 1) | ~known).all(1)
        compatible &= ((library["codes"][:,:,1] == codes[:,1]) | ~known).all(1)
        for grade in (2,3):
            compatible &= ((library["codes"][:,:,grade] == codes[:,grade]) | (codes[:,grade] == 0)).all(1)
        # This baseline supports the native sampled timestamps; no silent time warp.
        times = query["timestamps"][i]-query["timestamps"][i,0]
        compatible &= torch.isclose(library["timestamps"],times[None],atol=1e-5).all(1)
        common = library["frame"]["visible"] & query["visible0"][i]
        compatible &= (common.sum(1)>=4) & common[:,1]
        cost = ((library["frame"]["pose"]-frame["pose"][i]).square().sum(-1)*common).sum(1)/common.sum(1).clamp_min(1)
        cost += .25*(1-common.float().mean(1))
        choices = torch.argsort(cost.masked_fill(~compatible,torch.inf))[:candidates].tolist()
        choices = [d for d in choices if compatible[d]]
        # Sample among the four nearest compatible poses; retain others as safe fallbacks.
        rng_seed = int.from_bytes(hashlib.sha256(f"{seed}:{query['ids'][i]}".encode()).digest()[:8],"big") % (2**63-1)
        count = min(4,len(choices))
        order = torch.randperm(count,generator=torch.Generator().manual_seed(rng_seed)).tolist()
        choices = [choices[n] for n in order]+choices[count:]
        one = {k:v[i:i+1] for k,v in frame.items()}
        chosen, rejected, pred = None, 0, None
        for donor in choices:
            candidate = transfer_segments(one,library["q"][donor],library["valid"][donor])
            future, support = candidate["xy_norm"][1:],candidate["render_visible"][1:]
            outside = ((future<0)|(future>1)).any(-1)
            if not torch.isfinite(future).all() or (outside & support).any() or not support.any():
                rejected += 1
                continue
            chosen,pred = donor,candidate
            break
        if pred is None:
            pred = {"xy_norm":query["pose0"][i,None].expand(20,-1,-1),
                    "render_visible":query["visible0"][i,None].expand(20,-1)}
        outputs.append(pred)
        metadata.append({"donor_clip_id":None if chosen is None else library["ids"][chosen],
            "donor_patient_id":None if chosen is None else library["patients"][chosen],
            "pose_cost":None if chosen is None else float(cost[chosen]),
            "compatible_donors":int(compatible.sum()),"rejected_transfers":rejected,
            "static_fallback":chosen is None,"transferred_joint_fraction":float(pred["render_visible"][1:].float().mean())})
    return {k:torch.stack([p[k] for p in outputs]) for k in outputs[0]},metadata
