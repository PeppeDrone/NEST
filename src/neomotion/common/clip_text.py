"""Frozen, hash-pinned CLIP ViT-L/14 text encoder (extracted from the legacy stage1.service)."""


from __future__ import annotations

import hashlib
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn



CLIP_TEXT_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_TEXT_MODEL_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
CLIP_MODEL_SAFETENSORS_SHA256 = (
    "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"
)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCLIPTextConditioner:
    """Lazy Stable-Diffusion-compatible CLIP text encoder.

    ``transformers`` is imported only when this optional production adapter is
    constructed. Tests and callers can supply any object exposing the same
    ``encode(prompts, device) -> (hidden_states, attention_mask)`` contract.
    """

    def __init__(
        self,
        model_name: str = CLIP_TEXT_MODEL_ID,
        *,
        revision: str = CLIP_TEXT_MODEL_REVISION,
        local_files_only: bool = False,
        torch_dtype: torch.dtype | None = None,
        verify_model_hash: bool = True,
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError as error:
            raise RuntimeError(
                "FrozenCLIPTextConditioner requires the optional transformers package"
            ) from error
        if not revision:
            raise ValueError("the CLIP text tower must use an immutable revision")
        self.model_id = model_name
        self.revision = revision
        self.model_sha256 = None
        if verify_model_hash:
            resolved_model_file = hf_hub_download(
                repo_id=model_name,
                filename="model.safetensors",
                revision=revision,
                local_files_only=local_files_only,
            )
            actual_hash = _file_sha256(resolved_model_file)
            if (
                model_name == CLIP_TEXT_MODEL_ID
                and revision == CLIP_TEXT_MODEL_REVISION
                and actual_hash != CLIP_MODEL_SAFETENSORS_SHA256
            ):
                raise RuntimeError(
                    "pinned CLIP model.safetensors failed SHA-256 verification"
                )
            self.model_sha256 = actual_hash
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        self.text_encoder.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(
        self, prompts: Sequence[str], device: str | torch.device
    ) -> tuple[Tensor, Tensor]:
        device = torch.device(device)
        self.text_encoder.to(device)
        tokens = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        hidden = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return hidden, attention_mask
