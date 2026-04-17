"""
CubeTransformer — small transformer for 2x2x2 cube next-move prediction.

Input:  (batch, 144) float32 one-hot cube state
Output: (batch, 18)  logits over 18 moves

Architecture:
    embed:  Linear(144, d_model)          → hook_embed
    blocks: TransformerBlock × n_layers
              LN → Attention → residual   → hook_resid_mid
              LN → MLP      → residual    → hook_resid_post
    head:   LayerNorm → Linear(d_model, 18)

Note: input is a single token so attention is degenerate (softmax weight
is always 1.0 for seq_len=1). The residual stream evolves primarily through
the MLP sublayers. HookPoints follow TransformerLens naming so
run_with_cache() works out of the box for Phase 4 probing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.hook_points import HookedRootModule, HookPoint

from cube import NUM_MOVES, STATE_DIM


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.hook_attn_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        q, k, v = self.qkv(x.unsqueeze(1)).chunk(3, dim=-1)           # (B,1,D) each
        q = q.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)   # (B,H,1,dh)
        k = k.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.softmax(q @ k.transpose(-2, -1) * self.d_head ** -0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, D)
        return self.hook_attn_out(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)
        self.hook_mlp_out = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hook_mlp_out(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult)
        self.hook_resid_pre  = HookPoint()
        self.hook_resid_mid  = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        x = self.hook_resid_mid(x + self.attn(self.ln1(x)))
        x = self.hook_resid_post(x + self.mlp(self.ln2(x)))
        return x


class CubeTransformer(HookedRootModule):
    """
    Small transformer for next-move prediction on 2x2x2 Rubik's cube states.

    Hooks available via run_with_cache():
        hook_embed
        blocks.{i}.hook_resid_pre
        blocks.{i}.hook_resid_mid
        blocks.{i}.hook_resid_post
        blocks.{i}.attn.hook_attn_out
        blocks.{i}.mlp.hook_mlp_out
    """

    def __init__(
        self,
        d_model:  int = 128,
        n_layers: int = 4,
        n_heads:  int = 4,
        mlp_mult: int = 4,
        n_classes: int = NUM_MOVES,
    ):
        super().__init__()
        self.cfg = {
            "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "mlp_mult": mlp_mult,
            "n_classes": n_classes,
        }
        self.embed    = nn.Linear(STATE_DIM, d_model)
        self.hook_embed = HookPoint()
        self.blocks   = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_mult) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, n_classes, bias=False)
        self.setup()  # required by HookedRootModule — registers all HookPoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_embed(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_final(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_model(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> CubeTransformer:
    """Reconstruct a CubeTransformer from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = CubeTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    return model
