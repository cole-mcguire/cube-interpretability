"""
Export three static figures for the final report PDF.

Outputs (docs/figures/):
    fig_dla_ablation.pdf   — per-head DLA vs ablation-sensitivity scatter
    fig_attn_heatmap.pdf   — L0H1 attention heatmap at d=1,5,9,11
    fig_neuron_tuning.pdf  — top-10 mlp_0 neuron tuning curves (z-scored)

Usage:
    uv run python -m interp.export_paper_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

from dataset import load_split
from model import load_model, states_to_corners, CORNER_NAMES
from interp.corner_circuit import compute_per_head_dla, compute_ablation_deltas
from interp.corner_analysis import extract_attention
from interp.neuron_profile import get_mlp0_activations, compute_neuron_dla_mlp0

FIGURE_DIR = ROOT / "docs" / "figures"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Figure 1: DLA vs ablation sensitivity scatter
# ---------------------------------------------------------------------------

def save_dla_ablation_scatter(
    dla: np.ndarray,
    deltas: np.ndarray,
    baseline: float,
) -> None:
    n_layers, n_heads = dla.shape
    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Highlight the two interpretively important heads
    HIGHLIGHT = {
        (0, 1): ("#d62728", "L0H1\n(routing)"),   # highest ablation sensitivity
        (3, 1): ("#1f77b4", "L3H1\n(suppression)"), # largest |DLA|
    }

    for l in range(n_layers):
        for h in range(n_heads):
            x = float(dla[l, h])
            y = float(deltas[l, h] * 100)
            lh = (l, h)
            color, note = HIGHLIGHT.get(lh, ("#999999", None))
            size = 60 if lh in HIGHLIGHT else 30
            zorder = 5 if lh in HIGHLIGHT else 3
            ax.scatter(x, y, s=size, color=color, edgecolors="white",
                       linewidth=0.5, zorder=zorder)
            # Always label; adjust offset to avoid overlap
            va = "bottom"
            offset_y = 0.9
            if lh == (3, 1):
                offset_y = -2.5
                va = "top"
            ax.annotate(
                f"L{l}H{h}", (x, y),
                xytext=(x + 0.005, y + offset_y),
                fontsize=7, color=color, va=va,
                arrowprops=None,
            )

    ax.axhline(0, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_xlabel("Mean DLA (contribution to correct-distance logit)")
    ax.set_ylabel("Accuracy drop when ablated (pp)")
    ax.set_title(
        f"Per-head DLA vs. ablation sensitivity  "
        f"(baseline {baseline * 100:.1f}%)"
    )
    fig.tight_layout()

    out = FIGURE_DIR / "fig_dla_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: L0H1 attention heatmap across distances
# ---------------------------------------------------------------------------

def save_attn_heatmap(
    attn_by_dist: np.ndarray,   # (max_dist, n_layers, n_heads, 8, 8)
    head_idx: int = 1,
    layer_idx: int = 0,
) -> None:
    max_dist_in_data = attn_by_dist.shape[0]
    distances = [1, 4, 7, 10]
    # Filter to distances that exist and have data
    valid = [d for d in distances if d < max_dist_in_data and attn_by_dist[d].max() > 0]

    n_panels = len(valid)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.3 * n_panels, 2.5),
                             gridspec_kw={"wspace": 0.4})
    if n_panels == 1:
        axes = [axes]

    panels = [attn_by_dist[d, layer_idx, head_idx] for d in valid]
    vmax = max(p.max() for p in panels)

    for ax, d, attn in zip(axes, valid, panels):
        im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        ax.set_title(f"$d = {d}$", fontsize=9)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels(CORNER_NAMES, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticklabels(CORNER_NAMES, fontsize=5.5)
        ax.tick_params(length=0)

    axes[0].set_ylabel("Query cubie", fontsize=8)
    axes[n_panels // 2].set_xlabel("Key cubie", fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Attention weight", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"L{layer_idx}H{head_idx} mean attention weights by optimal distance",
        fontsize=10, y=1.04,
    )

    out = FIGURE_DIR / "fig_attn_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Neuron tuning curves
# ---------------------------------------------------------------------------

def save_neuron_tuning(
    activations: np.ndarray,
    labels: np.ndarray,
    top_neurons: list[int],
    dla: np.ndarray,
) -> None:
    n_classes = 12
    K = len(top_neurons)
    matrix = np.zeros((K, n_classes))
    for ci in range(n_classes):
        mask = labels == ci
        if mask.sum() > 0:
            matrix[:, ci] = activations[np.ix_(mask, top_neurons)].mean(axis=0)

    mu = matrix.mean(axis=1, keepdims=True)
    sd = matrix.std(axis=1, keepdims=True) + 1e-8
    matrix_z = (matrix - mu) / sd

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    im = ax.imshow(matrix_z, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(range(n_classes))
    ax.set_yticks(range(K))
    ax.set_yticklabels(
        [f"n{n}  ({dla[n]:.3f})" for n in top_neurons],
        fontsize=7.5,
    )
    ax.set_xlabel("Optimal distance")
    ax.set_ylabel("Neuron  (DLA score)")
    ax.set_title(r"Top-10 mlp$_0$ neurons — mean activation by distance (z-scored per neuron)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("z-score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    fig.tight_layout()
    out = FIGURE_DIR / "fig_neuron_tuning.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    flat_ckpt   = ROOT / "checkpoints" / "best.pt"
    corner_ckpt = ROOT / "checkpoints" / "corner" / "best.pt"

    print("Loading val set...")
    data   = load_split(ROOT / "data" / "val.npz")
    states = data["states"].astype(np.float32)
    labels = data["optimal_distance"].astype(int)
    states_corner = states_to_corners(states)
    print(f"  {len(states):,} val examples")

    # ---- Corner model (Figures 1 & 2) ----
    print(f"\nLoading corner model from {corner_ckpt}...")
    corner_model = load_model(corner_ckpt, device=device)
    corner_model.eval()

    print("Figure 1: computing per-head DLA...")
    dla_heads = compute_per_head_dla(corner_model, states_corner, labels, device=device)

    print("Figure 1: computing ablation deltas (16 ablations — takes ~2 min)...")
    baseline, deltas = compute_ablation_deltas(corner_model, states_corner, labels, device=device)
    save_dla_ablation_scatter(dla_heads, deltas, baseline)

    print("Figure 2: extracting attention weights...")
    n_layers = corner_model.cfg["n_layers"]
    n_heads  = corner_model.cfg["n_heads"]
    all_attn = extract_attention(corner_model, states_corner, n_layers, n_heads, device=device)
    max_dist = int(labels.max()) + 1
    attn_by_dist = np.stack([
        all_attn[:, labels == d].mean(axis=1) if (labels == d).any()
        else np.zeros((n_layers, n_heads, 8, 8), dtype=np.float32)
        for d in range(max_dist)
    ])  # (max_dist, n_layers, n_heads, 8, 8)
    save_attn_heatmap(attn_by_dist, head_idx=1, layer_idx=0)

    del corner_model

    # ---- Flat model (Figure 3) ----
    print(f"\nLoading flat model from {flat_ckpt}...")
    flat_model = load_model(flat_ckpt, device=device)
    flat_model.eval()

    print("Figure 3: caching mlp_0 activations...")
    activations = get_mlp0_activations(flat_model, states, device=device)

    print("Figure 3: computing neuron DLA...")
    dla_neurons = compute_neuron_dla_mlp0(flat_model, activations, labels)
    top_neurons = list(np.argsort(dla_neurons)[::-1][:10])
    save_neuron_tuning(activations, labels, top_neurons, dla_neurons)

    print(f"\nDone. Figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
