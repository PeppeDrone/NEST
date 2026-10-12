"""Small factorized clean-state denoiser, independent of historical model versions."""
import math
import torch
from torch import nn

from .common.topology import attention_bias_classes_v2


def embedding(values, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=values.device) / max(width // 2 - 1, 1))
    phase = values.float()[..., None] * frequencies
    out = torch.cat((phase.cos(), phase.sin()), -1)
    return torch.nn.functional.pad(out, (0, width - out.shape[-1]))


class MotionBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.temporal = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.spatial = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*width, width))
        self.modulation = nn.Linear(width, 8*width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.graph_bias = nn.Parameter(torch.zeros(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, padding, condition, graph):
        b, t, j, w = x.shape
        mods = self.modulation(condition).chunk(8, dim=-1)
        h = self.norms[0](x) * (1 + mods[0][:, None, None]) + mods[1][:, None, None]
        h = h.transpose(1, 2).reshape(b*j, t, w)
        h = self.temporal(h, h, h, need_weights=False)[0]
        x = x + self.dropout(h.reshape(b, j, t, w).transpose(1, 2))
        h = self.norms[1](x) * (1 + mods[2][:, None, None]) + mods[3][:, None, None]
        h = h.reshape(b*t, j, w)
        h = self.spatial(h, h, h, attn_mask=self.graph_bias[graph].to(h.dtype), need_weights=False)[0]
        x = x + self.dropout(h.reshape(b, t, j, w))
        h = self.norms[2](x) * (1 + mods[4][:, None, None]) + mods[5][:, None, None]
        h = self.cross(h.reshape(b, t*j, w), context, context, key_padding_mask=padding, need_weights=False)[0]
        x = x + self.dropout(h.reshape(b, t, j, w))
        h = self.norms[3](x) * (1 + mods[6][:, None, None]) + mods[7][:, None, None]
        return x + self.dropout(self.ff(h))


class ResidualMotionDenoiser(nn.Module):
    """Input/output [B,19,20,2] normalized clean state; visibility logits [B,19,20]."""
    def __init__(self, width=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.width = width
        self.input = nn.Linear(2, width)
        self.pose = nn.Linear(4, width)
        self.text = nn.Linear(768, width)
        self.joints = nn.Embedding(20, width)
        self.limbs = nn.Embedding(4, width)
        self.known = nn.Embedding(2, width)
        self.moving = nn.Embedding(2, width)
        self.amplitude = nn.Embedding(3, width)
        self.speed = nn.Embedding(3, width)
        self.null_condition = nn.Parameter(torch.randn(width)*0.02)
        self.diffusion_time = nn.Sequential(nn.Linear(width, width*4), nn.SiLU(), nn.Linear(width*4, width))
        self.time = nn.Linear(width, width)
        self.blocks = nn.ModuleList(MotionBlock(width, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.clean_head = nn.Linear(width, 2)
        self.visibility_head = nn.Linear(width, 1)
        nn.init.zeros_(self.clean_head.weight)
        nn.init.zeros_(self.clean_head.bias)
        self.register_buffer("graph", torch.tensor(attention_bias_classes_v2(20), dtype=torch.long))

    def forward(self, noisy, timestep, pose, visible, score, timestamps, text, text_mask, controls, dropped):
        if noisy.ndim != 4 or noisy.shape[1:] != (19, 20, 2):
            raise ValueError("NEST diffusion state must have shape [B,19,20,2]")
        b = noisy.shape[0]
        if timestamps.shape != (b, 20) or not torch.isfinite(timestamps).all() or not ((timestamps[:, 1:]-timestamps[:, :-1]) > 0).all():
            raise ValueError("NEST requires 20 finite, strictly increasing timestamps")
        pose_tokens = self.pose(torch.cat((pose, score[..., None], visible[..., None].float()), -1)) + self.joints.weight
        limb_tokens = (self.limbs.weight + self.known(controls[..., 0]) + self.moving(controls[..., 1])
                       + self.amplitude(controls[..., 2]) + self.speed(controls[..., 3]))
        # Drop every text/control memory token; the separate null token stays visible.
        text_padding = ~text_mask.bool() | dropped[:, None]
        control_padding = dropped[:, None].expand(b, 4)
        null = self.null_condition[None, None].expand(b, 1, -1)
        context = torch.cat((self.text(text), limb_tokens, pose_tokens, null), 1)
        padding = torch.cat((text_padding, control_padding, ~visible, ~dropped[:, None]), 1)
        relative_time = timestamps[:, 1:] - timestamps[:, :1]
        h = self.input(noisy) + self.joints.weight[None, None] + self.time(embedding(relative_time, self.width))[:, :, None]
        condition = self.diffusion_time(embedding(timestep, self.width))
        for block in self.blocks:
            h = block(h, context, padding, condition, self.graph)
        h = self.norm(h)
        return self.clean_head(h), self.visibility_head(h).squeeze(-1)
