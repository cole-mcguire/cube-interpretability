"""
CubeTransformer and CubeTransformerCorner.

CubeTransformer (arch="flat"):
    Input:  (batch, 144) float32 one-hot cube state — single token
    Attention is degenerate (seq_len=1); computation flows through MLP layers.

CubeTransformerCorner (arch="corner"):
    Input:  (batch, 8, 18) — one token per corner cubie (3 stickers × 6 colors)
    Attention operates over 8 meaningful tokens, enabling inter-cubie reasoning.
    Hooks expose per-head 8×8 attention weights for interpretability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens.hook_points import HookedRootModule, HookPoint

from cube import NUM_MOVES, STATE_DIM, CORNER_STICKERS


# ---------------------------------------------------------------------------
# Corner tokenization
# ---------------------------------------------------------------------------

N_CORNERS        = 8
CORNER_TOKEN_DIM = 18   # 3 stickers × 6 color one-hot
CORNER_NAMES     = ["UFR", "UFL", "UBL", "UBR", "DFR", "DFL", "DBL", "DBR"]


def states_to_corners(states: np.ndarray) -> np.ndarray:
    """Convert (N, 144) flat one-hot states to (N, 8, 18) per-corner tokens."""
    stickers = states.reshape(len(states), 24, 6)   # (N, 24, 6)
    out = np.empty((len(states), N_CORNERS, CORNER_TOKEN_DIM), dtype=np.float32)
    for ci, (s0, s1, s2) in enumerate(CORNER_STICKERS):
        out[:, ci, :6]   = stickers[:, s0]
        out[:, ci, 6:12] = stickers[:, s1]
        out[:, ci, 12:]  = stickers[:, s2]
    return out


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
    Small transformer for cube-state classification tasks, usually optimal-distance classification.

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


# ---------------------------------------------------------------------------
# Corner-tokenized model
# ---------------------------------------------------------------------------

class CornerAttention(nn.Module):
    """Multi-head self-attention over 8 corner tokens (B, 8, d_model)."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.hook_attn_weights = HookPoint()   # (B, n_heads, 8, 8) — interpretable!
        self.hook_attn_out     = HookPoint()   # (B, 8, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, S, D)
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,S,dh)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        w = F.softmax(q @ k.transpose(-2, -1) * self.d_head ** -0.5, dim=-1)
        w = self.hook_attn_weights(w)
        out = (w @ v).transpose(1, 2).reshape(B, S, D)
        return self.hook_attn_out(self.out_proj(out))


class CornerBlock(nn.Module):
    """Transformer block operating on (B, 8, d_model) corner-token sequences."""
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CornerAttention(d_model, n_heads)
        self.ln2  = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2  = nn.Linear(mlp_mult * d_model, d_model)
        self.hook_mlp_out    = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, S, D)
        x = x + self.attn(self.ln1(x))
        mlp_out = self.hook_mlp_out(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return self.hook_resid_post(x + mlp_out)


class CubeTransformerCorner(HookedRootModule):
    """
    Corner-tokenized transformer for optimal-distance classification.

    Input:  (batch, 8, 18) — 8 corners, each a concatenation of 3 sticker
            one-hot vectors (3 × 6 = 18 dims). Build with states_to_corners().
    Output: (batch, n_classes)

    With 8 real tokens, attention heads can learn inter-cubie relationships:
    which corners co-misalign, which form solved faces, etc.

    Hooks available via run_with_cache():
        hook_embed                          (B, 8, d_model)
        blocks.{i}.attn.hook_attn_weights   (B, n_heads, 8, 8)
        blocks.{i}.attn.hook_attn_out       (B, 8, d_model)
        blocks.{i}.hook_mlp_out             (B, 8, d_model)
        blocks.{i}.hook_resid_post          (B, 8, d_model)
        hook_pooled                         (B, d_model)
    """

    def __init__(
        self,
        d_model:   int = 128,
        n_layers:  int = 4,
        n_heads:   int = 4,
        mlp_mult:  int = 4,
        n_classes: int = 12,
    ):
        super().__init__()
        self.cfg = {
            "d_model": d_model, "n_layers": n_layers,
            "n_heads": n_heads, "mlp_mult": mlp_mult,
            "n_classes": n_classes, "arch": "corner",
        }
        self.corner_proj = nn.Linear(CORNER_TOKEN_DIM, d_model)
        self.pos_embed   = nn.Embedding(N_CORNERS, d_model)
        self.hook_embed  = HookPoint()
        self.blocks      = nn.ModuleList([
            CornerBlock(d_model, n_heads, mlp_mult) for _ in range(n_layers)
        ])
        self.hook_pooled = HookPoint()
        self.ln_final    = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, n_classes, bias=False)
        self.setup()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, 8, 18)
        pos = torch.arange(N_CORNERS, device=x.device)
        h   = self.corner_proj(x) + self.pos_embed(pos)   # (B, 8, d_model)
        h   = self.hook_embed(h)
        for block in self.blocks:
            h = block(h)
        pooled = self.hook_pooled(h.mean(dim=1))           # (B, d_model)
        return self.head(self.ln_final(pooled))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model loading (dispatches by arch)
# ---------------------------------------------------------------------------

def load_model(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> CubeTransformer | CubeTransformerCorner:
    """Load a CubeTransformer or CubeTransformerCorner from a checkpoint."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg  = ckpt["config"]
    if cfg.get("arch") == "corner":
        model = CubeTransformerCorner(**{k: v for k, v in cfg.items() if k != "arch"})
    else:
        model = CubeTransformer(**cfg)
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model = model.to(device)
    return model
