"""Optional limb tokens derived from the selected prompt only, never annotation lookup."""
import re
import torch

LIMBS = ("left_arm", "right_arm", "left_leg", "right_leg")


def parse_prompt(prompt: str) -> tuple[torch.Tensor, dict]:
    """[4,4]: known, moving, amplitude(unknown/low/high), speed(unknown/slow/fast).

    Explicit stillness sets known=1,moving=0. Unmentioned limbs remain unknown.
    Unparsed language is retained in CLIP conditioning. Null CFG is a separate
    model token; an unknown parsed limb is NOT an unconditional example.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    text = re.sub(r"\s+", " ", prompt.lower().replace("-", " ")).strip()
    codes = torch.zeros(4, 4, dtype=torch.long)
    movement = r"\b(moves?|moving|movements?|kicks?|kicking|raises?|raising|extends?|extending|bends?|bending|flexes?|flexing|stretches?|stretching|lifts?|lifting|waves?|waving)\b"
    moving_words = re.findall(movement, text)
    negation = r"(?:does|do) not move|doesn't move|don't move|no movement|not moving"
    # Split descriptions when a new action starts; keep lists of body parts together.
    clauses = re.split(r"[;.]|,\s+and\s+(?=its |his |her )|,?\s+(?:and|while|but)\s+(?=(?:the |its |his |her )?(?:newborn |baby )?(?:moves|kicks|raises|bends|remains))", text)
    clauses = [piece for clause in clauses for piece in re.split(
        r",?\s+(?:and|while|but)\s+(?=(?:the |its |his |her )?(?:left|right) (?:arm|leg) (?:is |remains |moves|kicks|does|stays))", clause)]
    for clause in clauses:
        explicit = set()
        for i, limb in enumerate(LIMBS):
            side, part = limb.split("_")
            if re.search(rf"\b{side}\s+{part}\b", clause):
                explicit.add(i)
        for part, indices in [("arm", [0, 1]), ("leg", [2, 3])]:
            if re.search(rf"\b(?:both\s+)?{part}s\b", clause):
                explicit.update(indices)
        if re.search(r"\blimbs\b|whole body|all four", clause):
            explicit.update(range(4))
        inherited_verb = clause.startswith(("its ", "his ", "her ")) and bool(moving_words)
        moving = bool(re.search(movement, clause)) or inherited_verb
        negative = bool(re.search(negation, clause))
        still = negative or (bool(re.search(r"\b(still|motionless|stationary)\b", clause))
                             and not moving and not re.search(r"not (?:still|motionless|stationary)", clause))
        if still:
            # A named still limb never labels unrelated limbs. Global stillness
            # is recognized only when no other body part/action is described.
            if not explicit and not re.search(r"\b(head|hands?|feet|foot|fingers?)\b", clause):
                explicit.update(range(4))
            for i in explicit:
                codes[i] = torch.tensor([1, 0, 0, 0])
            continue
        if not moving or negative:
            continue
        # Resolve grades per clause, rather than transferring arm grades to legs.
        amp = 2 if re.search(r"high amplitude|large amplitude|wide", clause) else 1 if re.search(r"low amplitude|small amplitude|slight", clause) else 0
        speed = 2 if re.search(r"fast|rapid|quick", clause) else 1 if re.search(r"slow|gently", clause) else 0
        for i in explicit:
            codes[i] = torch.tensor([1, 1, amp, speed])
    return codes, {"recognized_limbs": [limb for i, limb in enumerate(LIMBS) if codes[i, 0]],
                   "explicit_still": bool(codes[:, 0].any()) and not bool((codes[:, 1] == 1).any()),
                   "has_structured_information": bool(codes[:, 0].any()),
                   "unsupported_structured_parts": sorted(set(re.findall(r"\b(?:head|hands?|feet|foot|fingers?)\b", text))),
                   "conditioning_source": "selected_prompt_only"}


def encode_prompt_table(prompts, encoder, device, batch_size=128):
    """Frozen CLIP features cached on CPU; include an empty-text CFG row."""
    unique = sorted(set(prompts) | {""})
    hidden, masks = [], []
    for start in range(0, len(unique), batch_size):
        with torch.inference_mode():
            h, m = encoder.encode(unique[start:start+batch_size], device)
        hidden.append(h.detach().float().cpu())
        masks.append(m.detach().bool().cpu())
    return {"prompts": unique, "hidden": torch.cat(hidden), "mask": torch.cat(masks),
            "controls": torch.stack([parse_prompt(p)[0] for p in unique])}


def prompt_batch(table, prompts, device):
    lookup = {p: i for i, p in enumerate(table["prompts"])}
    indices = torch.tensor([lookup[p] for p in prompts])
    return {"text": table["hidden"][indices].to(device), "text_mask": table["mask"][indices].to(device),
            "controls": table["controls"][indices].to(device)}
