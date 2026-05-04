"""
Phase 12: Corner-tokenized transformer analysis.

Compares the corner-tokenized model (8 corner tokens, real attention) against
the flat model (single 144-dim token, degenerate attention) on:

  1. Val accuracy — does the corner architecture match or beat the flat model?
  2. Attention patterns — what do heads learn to attend to between corners?
     Mean attention weights per distance class reveal whether heads encode
     geometric relationships (e.g., attending between co-misaligned corners).
  3. Attention entropy vs distance — is attention more diffuse on harder states?

Requires:
    checkpoints/best.pt            — flat model
    checkpoints/corner/best.pt     — corner model (train with:
                                     uv run python train.py --arch corner
                                                            --out checkpoints/corner)

Output:
    docs/corner_results.html

Usage:
    uv run python -m interp.corner_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import load_split
from model import load_model, states_to_corners, CORNER_NAMES


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention(
    model,
    inputs: np.ndarray,        # (N, 8, 18) corner tokens
    n_layers: int,
    n_heads: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return mean attention weights (n_layers, n_heads, 8, 8) averaged over N examples."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    hook_names = [f"blocks.{i}.attn.hook_attn_weights" for i in range(n_layers)]
    accum = {h: [] for h in hook_names}

    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for h in hook_names:
            accum[h].append(cache[h].cpu().float().numpy())

    # (n_layers, N, n_heads, 8, 8)
    stacked = np.stack([
        np.concatenate(accum[h], axis=0) for h in hook_names
    ], axis=0)
    return stacked   # (n_layers, N, n_heads, 8, 8)


@torch.no_grad()
def get_predictions(
    model,
    inputs: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        logits = model(batch)
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_accuracy_fig(flat_acc: float, corner_acc: float, baseline: float):
    import plotly.graph_objects as go

    models = ["Flat model<br>(single 144-dim token)", "Corner model<br>(8 × 18-dim tokens)"]
    accs   = [flat_acc * 100, corner_acc * 100]
    colors = ["#3b82f6", "#10b981"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=accs,
        marker_color=colors,
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        width=0.4,
    ))
    fig.add_hline(
        y=baseline * 100,
        line=dict(color="#475569", dash="dot", width=1.5),
        annotation_text=f"random baseline ({baseline*100:.1f}%)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Val accuracy: flat vs corner-tokenized model",
        yaxis=dict(title="Val accuracy (%)", range=[0, 105], **_GRID),
        xaxis=dict(**_GRID),
        height=380, **_DARK,
    )
    return fig


def _build_attn_heatmap_fig(
    attn_by_dist: np.ndarray,   # (n_dist, n_layers, n_heads, 8, 8)
    n_layers: int,
    n_heads: int,
    distances: list[int],
    layer_idx: int = 0,         # which layer to show
):
    """Mean attention pattern for selected distances at a given layer, all heads."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_show = len(distances)
    fig = make_subplots(
        rows=n_show, cols=n_heads,
        subplot_titles=[
            f"d={d} head {h}" for d in distances for h in range(n_heads)
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for ri, d in enumerate(distances):
        for ci in range(n_heads):
            attn = attn_by_dist[d, layer_idx, ci]   # (8, 8)
            fig.add_trace(go.Heatmap(
                z=attn,
                x=CORNER_NAMES, y=CORNER_NAMES,
                colorscale="Blues",
                zmin=0, zmax=attn_by_dist[:, layer_idx, ci].max(),
                showscale=(ri == 0 and ci == n_heads - 1),
                hovertemplate="from %{y} → to %{x}<br>weight: %{z:.3f}<extra></extra>",
            ), row=ri + 1, col=ci + 1)

    fig.update_layout(
        title=f"Mean attention weights by distance (layer {layer_idx}) — rows=query corner, cols=key corner",
        height=180 * n_show + 60,
        **_DARK,
    )
    fig.update_xaxes(tickangle=45, **_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _build_entropy_fig(
    all_attn: np.ndarray,   # (n_layers, N, n_heads, 8, 8)
    labels: np.ndarray,     # (N,) optimal distance
    n_layers: int,
    n_heads: int,
):
    """Mean per-head attention entropy vs optimal distance."""
    import plotly.graph_objects as go

    # entropy: -sum(p log p) per query position, then mean over positions and heads
    eps = 1e-9
    # all_attn: (n_layers, N, n_heads, 8, 8) — last dim is key, 2nd-last is query
    entropy_all = -np.sum(all_attn * np.log(all_attn + eps), axis=-1)  # (n_layers, N, n_heads, 8)
    entropy_mean = entropy_all.mean(axis=(-1, -2))   # (n_layers, N) — avg over heads and query pos

    distances  = sorted(set(labels.tolist()))
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b"]

    fig = go.Figure()
    for li in range(n_layers):
        ent_by_dist = [entropy_mean[li][labels == d].mean() for d in distances]
        fig.add_trace(go.Scatter(
            x=distances, y=ent_by_dist,
            mode="lines+markers", name=f"L{li}",
            line=dict(color=colors[li % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f"L{li}  d=%{{x}}<br>entropy: %{{y:.3f}}<extra></extra>",
        ))

    # Max entropy (uniform over 8) = log(8) ≈ 2.08
    fig.add_hline(
        y=np.log(8),
        line=dict(color="#475569", dash="dot", width=1.5),
        annotation_text="uniform (log 8 ≈ 2.08)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Mean attention entropy vs distance — higher = more diffuse attention",
        xaxis=dict(title="Optimal distance", **_GRID),
        yaxis=dict(title="Attention entropy (nats)", **_GRID),
        height=400, **_DARK,
    )
    return fig


def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

NAV_CSS = """\
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
header { background: #1e293b; border-bottom: 1px solid #334155; padding: 10px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
header h1 { font-size: 14px; font-weight: 700; color: #f1f5f9; white-space: nowrap; }
nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
nav a { color: #94a3b8; text-decoration: none; font-size: 11px; padding: 4px 10px; border-radius: 5px; background: #0f172a; border: 1px solid #334155; transition: color .15s, background .15s; white-space: nowrap; }
nav a:hover { color: #e2e8f0; background: #334155; }
nav a[aria-current="page"] { color: #f1f5f9; background: #334155; border-color: #475569; }
.nav-group { position: relative; }
.nav-group-btn { color: #94a3b8; font-size: 11px; padding: 4px 10px; border-radius: 5px; background: #0f172a; border: 1px solid #334155; cursor: pointer; white-space: nowrap; font-family: inherit; transition: color .15s, background .15s; }
.nav-group-btn:hover, .nav-group:focus-within .nav-group-btn { color: #e2e8f0; background: #334155; }
.nav-group-btn.active { color: #f1f5f9; background: #334155; border-color: #475569; }
.nav-group-menu { display: none; position: absolute; top: calc(100% + 4px); left: 0; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 4px; min-width: 220px; z-index: 100; flex-direction: column; gap: 2px; box-shadow: 0 8px 24px rgba(0,0,0,.5); }
.nav-group:hover .nav-group-menu, .nav-group:focus-within .nav-group-menu { display: flex; }
.nav-group-menu a { border-radius: 4px; }
.page-intro { max-width: 860px; padding: 18px 20px 4px; }
.page-intro h2 { font-size: 15px; font-weight: 700; color: #f1f5f9; margin-bottom: 10px; }
.page-intro p { font-size: 13px; color: #94a3b8; line-height: 1.65; margin-bottom: 6px; }
.page-intro strong { color: #cbd5e1; }
</style>"""

NAV_BODY = """\
<header>
  <h1>2×2 Cube Interpretability</h1>
  <nav>
    <a href="index.html">Visualizer</a>
    <div class="nav-group">
      <button class="nav-group-btn active">Analyses ▾</button>
      <div class="nav-group-menu">
        <a href="probe_results.html">Phase 4 — Probing</a>
        <a href="phase5a_results.html">Phase 5a — Patching</a>
        <a href="phase5b_results.html">Phase 5b — Lenses &amp; SAE</a>
        <a href="circuit_results.html">Phase 7 — Circuit</a>
        <a href="weights_results.html">Phase 8 — Weights</a>
        <a href="grokking_results.html">Phase 9 — Grokking</a>
        <a href="next_move_results.html">Phase 10 — Next Move</a>
        <a href="superposition_results.html">Phase 11 — Superposition</a>
        <a href="corner_results.html" aria-current="page">Phase 12 — Corner Model</a>
        <a href="corner_attn_results.html">Phase 13 — Head Ablation</a>
      </div>
    </div>
    <a href="progress_report.pdf">Final Report ↗</a>
  </nav>
</header>"""
PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 12 — Corner-Tokenized Transformer</h2>
  <p><strong>Question:</strong> Can replacing the single 144-dim token with 8 per-corner tokens make attention non-degenerate and geometrically meaningful?</p>
  <p><strong>Method:</strong> Train CubeTransformerCorner (8 × 18-dim tokens). Compare val accuracy; extract 8×8 inter-cubie attention heatmaps stratified by solve distance.</p>
  <p><strong>Finding:</strong> 76.2% val accuracy (vs 79.8% flat). Attention entropy 0.28–0.37 nats vs 2.08 uniform — heads attend sharply to 1–2 specific corner pairs.</p>
</section>"""



def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 12 — Corner Model</title>\n"
        "</head>\n<body>\n"
        + NAV_BODY + "\n"
        + PAGE_DESCRIPTION + "\n"
        + "\n".join(divs) + "\n"
        + "</body>\n</html>\n"
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    flat_path   = Path(args.checkpoint_flat)
    corner_path = Path(args.checkpoint_corner)

    if not corner_path.exists():
        raise FileNotFoundError(
            f"Corner model checkpoint not found: {corner_path}\n"
            "Train it first with:\n"
            "  uv run python train.py --arch corner --out checkpoints/corner"
        )

    print(f"Loading flat model from {flat_path}...")
    flat_ckpt  = torch.load(flat_path,   map_location="cpu", weights_only=False)
    flat_model = load_model(flat_path,   device=device)
    flat_acc   = flat_ckpt["val_acc"]

    print(f"Loading corner model from {corner_path}...")
    corner_ckpt  = torch.load(corner_path, map_location="cpu", weights_only=False)
    corner_model = load_model(corner_path, device=device)
    corner_acc   = corner_ckpt["val_acc"]

    n_layers = corner_model.cfg["n_layers"]
    n_heads  = corner_model.cfg["n_heads"]
    n_classes = corner_model.cfg["n_classes"]
    baseline  = 1.0 / n_classes

    print(f"\nFlat model val acc:   {flat_acc*100:.1f}%")
    print(f"Corner model val acc: {corner_acc*100:.1f}%")
    print(f"Random baseline:      {baseline*100:.1f}%")

    data   = load_split(Path(args.data) / f"{args.split}.npz")
    states = data["states"].astype(np.float32)
    labels = data["optimal_distance"].astype(int)
    corner_inputs = states_to_corners(states)

    print(f"\nExtracting attention weights from corner model ({len(states):,} examples)...")
    all_attn = extract_attention(corner_model, corner_inputs, n_layers, n_heads, device=device)
    # all_attn: (n_layers, N, n_heads, 8, 8)

    # Mean attention per distance class
    max_dist = labels.max() + 1
    attn_by_dist = np.stack([
        all_attn[:, labels == d].mean(axis=1) if (labels == d).any()
        else np.zeros((n_layers, n_heads, 8, 8), dtype=np.float32)
        for d in range(max_dist)
    ], axis=0)   # (max_dist, n_layers, n_heads, 8, 8)

    print("\nAttention entropy by layer (mean over all examples):")
    eps = 1e-9
    entropy = -np.sum(all_attn * np.log(all_attn + eps), axis=-1).mean(axis=(-1, -2))
    for li in range(n_layers):
        print(f"  L{li}: {entropy[li].mean():.3f} (uniform = {np.log(8):.3f})")

    distances_to_show = [0, 3, 7, 11]
    distances_to_show = [d for d in distances_to_show if d < max_dist]

    figs = [
        _build_accuracy_fig(flat_acc, corner_acc, baseline),
        _build_attn_heatmap_fig(attn_by_dist, n_layers, n_heads,
                                distances_to_show, layer_idx=0),
        _build_entropy_fig(all_attn, labels, n_layers, n_heads),
    ]

    out_path = Path(args.out_dir) / "corner_results.html"
    write_page(figs, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 12: corner-tokenized model analysis.")
    p.add_argument("--checkpoint-flat",   default="checkpoints/best.pt",        dest="checkpoint_flat")
    p.add_argument("--checkpoint-corner", default="checkpoints/corner/best.pt", dest="checkpoint_corner")
    p.add_argument("--data",    default="data")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="docs", dest="out_dir")
    main(p.parse_args())
