"""
Phase 13: Corner-model attention head ablation and specialization.

For each of the 16 heads (4 layers × 4 heads) in CubeTransformerCorner:
  1. Ablation — replace that head's attention weights with uniform (1/8),
     measure val-accuracy drop.  Identifies which heads are load-bearing.
  2. Mean attention pattern — average 8×8 attention map across the val set,
     revealing which corner pairs each head routes information between.
  3. Distance modulation — for the top-3 most-ablated heads, show how the
     mean attention pattern shifts from d=0 (solved) to d=11 (hardest).

Requires:
    checkpoints/corner/best.pt   — corner model (train with:
                                    uv run python train.py --arch corner
                                                           --out checkpoints/corner)

Output:
    docs/corner_attn_results.html

Usage:
    uv run python -m interp.corner_attn
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
# Accuracy with a single head ablated
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy_ablate_head(
    model,
    inputs: np.ndarray,       # (N, 8, 18)
    labels: np.ndarray,       # (N,)
    layer: int,
    head: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> float:
    """Val accuracy when head `head` in layer `layer` attends uniformly (1/8)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    hook_name = f"blocks.{layer}.attn.hook_attn_weights"

    def _ablate(value, hook):
        # value: (B, n_heads, 8, 8) — replace head h with uniform
        value = value.clone()
        value[:, head, :, :] = 1.0 / 8
        return value

    correct = total = 0
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        tgts  = labels[start : start + batch_size]
        logits = model.run_with_hooks(batch, fwd_hooks=[(hook_name, _ablate)])
        preds  = logits.argmax(-1).cpu().numpy()
        correct += (preds == tgts).sum()
        total   += len(tgts)

    return correct / total


@torch.no_grad()
def baseline_accuracy(
    model,
    inputs: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        preds = model(batch).argmax(-1).cpu().numpy()
        correct += (preds == labels[start : start + batch_size]).sum()
        total   += len(labels[start : start + batch_size])
    return correct / total


# ---------------------------------------------------------------------------
# Mean attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_attention_per_head(
    model,
    inputs: np.ndarray,          # (N, 8, 18)
    n_layers: int,
    n_heads: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Returns (n_layers, n_heads, 8, 8) mean attention across the val set."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    hook_names = [f"blocks.{i}.attn.hook_attn_weights" for i in range(n_layers)]
    accum = {h: [] for h in hook_names}

    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for h in hook_names:
            accum[h].append(cache[h].cpu().float().numpy())   # (B, H, 8, 8)

    # mean over N
    return np.stack([
        np.concatenate(accum[h], axis=0).mean(axis=0)   # (H, 8, 8)
        for h in hook_names
    ], axis=0)   # (n_layers, n_heads, 8, 8)


@torch.no_grad()
def mean_attention_by_distance(
    model,
    inputs: np.ndarray,    # (N, 8, 18)
    labels: np.ndarray,    # (N,) optimal distance
    n_layers: int,
    n_heads: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Returns (max_dist, n_layers, n_heads, 8, 8)."""
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

    all_attn = np.stack([
        np.concatenate(accum[h], axis=0) for h in hook_names
    ], axis=0)   # (n_layers, N, n_heads, 8, 8)
    all_attn = all_attn.transpose(1, 0, 2, 3, 4)   # (N, n_layers, n_heads, 8, 8)

    max_dist = int(labels.max()) + 1
    result = np.zeros((max_dist, n_layers, n_heads, 8, 8), dtype=np.float32)
    for d in range(max_dist):
        mask = labels == d
        if mask.any():
            result[d] = all_attn[mask].mean(axis=0)
    return result


# ---------------------------------------------------------------------------
# Specialization metrics
# ---------------------------------------------------------------------------

def head_specialization(mean_attn: np.ndarray) -> dict:
    """
    Given mean_attn (n_layers, n_heads, 8, 8), compute per-head metrics:
      - sharpness: mean of max attention weight per query position
      - within_layer: mean attention to same-layer corners (U↔U or D↔D)
      - cross_layer: mean attention to opposite-layer corners
    """
    n_layers, n_heads = mean_attn.shape[:2]
    U = list(range(4))   # UFR,UFL,UBL,UBR = indices 0–3
    D = list(range(4, 8))  # DFR,DFL,DBL,DBR = indices 4–7

    sharpness    = np.zeros((n_layers, n_heads))
    within_layer = np.zeros((n_layers, n_heads))
    cross_layer  = np.zeros((n_layers, n_heads))

    for li in range(n_layers):
        for hi in range(n_heads):
            A = mean_attn[li, hi]   # (8, 8) query × key
            sharpness[li, hi] = A.max(axis=-1).mean()

            # query U attending to key U, or query D attending to key D
            wl = (A[np.ix_(U, U)].mean() + A[np.ix_(D, D)].mean()) / 2
            # query U attending to key D, or query D attending to key U
            cl = (A[np.ix_(U, D)].mean() + A[np.ix_(D, U)].mean()) / 2
            within_layer[li, hi] = wl
            cross_layer[li, hi]  = cl

    return {"sharpness": sharpness, "within_layer": within_layer, "cross_layer": cross_layer}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_ablation_fig(
    acc_drop: np.ndarray,   # (n_layers, n_heads)  positive = accuracy dropped
    baseline: float,
    n_layers: int,
    n_heads: int,
):
    import plotly.graph_objects as go

    head_labels = [f"H{h}" for h in range(n_heads)]
    layer_labels = [f"L{l}" for l in range(n_layers)]

    fig = go.Figure(go.Heatmap(
        z=acc_drop * 100,
        x=head_labels,
        y=layer_labels,
        colorscale="Reds",
        text=[[f"{acc_drop[l, h]*100:.1f}pp" for h in range(n_heads)] for l in range(n_layers)],
        texttemplate="%{text}",
        hovertemplate="L%{y}  H%{x}<br>accuracy drop: %{z:.2f} pp<extra></extra>",
        colorbar=dict(title="Acc drop (pp)"),
    ))
    fig.update_layout(
        title=f"Accuracy drop when head is ablated (uniform attention) — baseline {baseline*100:.1f}%",
        xaxis=dict(title="Head", **_GRID),
        yaxis=dict(title="Layer", autorange="reversed", **_GRID),
        height=320, **_DARK,
    )
    return fig


def _build_mean_attn_fig(
    mean_attn: np.ndarray,   # (n_layers, n_heads, 8, 8)
    n_layers: int,
    n_heads: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_layers, cols=n_heads,
        subplot_titles=[f"L{l} H{h}" for l in range(n_layers) for h in range(n_heads)],
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )
    vmax = mean_attn.max()

    for li in range(n_layers):
        for hi in range(n_heads):
            fig.add_trace(go.Heatmap(
                z=mean_attn[li, hi],
                x=CORNER_NAMES, y=CORNER_NAMES,
                colorscale="Blues",
                zmin=0, zmax=vmax,
                showscale=(li == 0 and hi == n_heads - 1),
                hovertemplate="query %{y} → key %{x}<br>weight: %{z:.3f}<extra></extra>",
            ), row=li + 1, col=hi + 1)

    fig.update_layout(
        title="Mean attention pattern per head (averaged over full val set) — rows=query corner, cols=key corner",
        height=200 * n_layers + 60,
        **_DARK,
    )
    fig.update_xaxes(tickangle=45, **_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _build_specialization_fig(
    metrics: dict,
    n_layers: int,
    n_heads: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Sharpness (mean max attn weight)",
            "Within-layer bias (U↔U or D↔D)",
            "Cross-layer bias (U↔D or D↔U)",
        ],
    )

    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    head_labels = [f"H{h}" for h in range(n_heads)]

    for metric_key, col in [("sharpness", 1), ("within_layer", 2), ("cross_layer", 3)]:
        data = metrics[metric_key]   # (n_layers, n_heads)
        for li in range(n_layers):
            fig.add_trace(go.Bar(
                x=head_labels,
                y=data[li],
                name=f"L{li}",
                marker_color=colors[li],
                legendgroup=f"L{li}",
                showlegend=(col == 1),
                hovertemplate=f"L{li}  %{{x}}<br>{metric_key}: %{{y:.3f}}<extra></extra>",
            ), row=1, col=col)

    fig.update_layout(
        title="Head specialization — U-layer: UFR,UFL,UBL,UBR (idx 0–3)  D-layer: DFR,DFL,DBL,DBR (idx 4–7)",
        barmode="group",
        height=400, **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _build_distance_mod_fig(
    attn_by_dist: np.ndarray,   # (max_dist, n_layers, n_heads, 8, 8)
    top_heads: list[tuple[int, int]],   # [(layer, head), ...]
    distances: list[int],
):
    """For the top ablated heads, show attention pattern at selected distances."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_show = len(distances)
    n_heads_show = len(top_heads)

    fig = make_subplots(
        rows=n_heads_show, cols=n_show,
        subplot_titles=[
            f"L{l}H{h}  d={d}" for (l, h) in top_heads for d in distances
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for ri, (li, hi) in enumerate(top_heads):
        # color scale max across all distances for this head
        vmax = max(attn_by_dist[d, li, hi].max() for d in distances if d < len(attn_by_dist))
        for ci, d in enumerate(distances):
            if d >= len(attn_by_dist):
                continue
            attn = attn_by_dist[d, li, hi]
            fig.add_trace(go.Heatmap(
                z=attn,
                x=CORNER_NAMES, y=CORNER_NAMES,
                colorscale="Blues",
                zmin=0, zmax=vmax,
                showscale=(ri == 0 and ci == n_show - 1),
                hovertemplate=f"L{li}H{hi} d={d}<br>%{{y}}→%{{x}}: %{{z:.3f}}<extra></extra>",
            ), row=ri + 1, col=ci + 1)

    fig.update_layout(
        title="Top ablated heads — how attention pattern shifts with solve distance",
        height=200 * n_heads_show + 60,
        **_DARK,
    )
    fig.update_xaxes(tickangle=45, **_GRID)
    fig.update_yaxes(**_GRID)
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
        <a href="corner_results.html">Phase 12 — Corner Model</a>
        <a href="corner_attn_results.html" aria-current="page">Phase 13 — Head Ablation</a>
      </div>
    </div>
    <a href="progress_report.pdf">Final Report ↗</a>
  </nav>
</header>"""
PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 13 — Attention Head Ablation</h2>
  <p><strong>Question:</strong> Which attention heads are load-bearing, and what geometric routing do they implement?</p>
  <p><strong>Method:</strong> Ablate each of the 16 heads (replace with uniform 1/8 attention); measure accuracy drop. Compute sharpness and U/D-layer routing bias per head.</p>
  <p><strong>Finding:</strong> Layer 0 dominates: ablating L0H1 drops accuracy 44.9 pp, L0H0 by 30.7 pp, L0H2 by 25.1 pp. All L1–L3 heads individually dispensable (≤7.7 pp).</p>
</section>"""



def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 13 — Head Ablation</title>\n"
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

    corner_path = Path(args.checkpoint)
    if not corner_path.exists():
        raise FileNotFoundError(
            f"Corner model checkpoint not found: {corner_path}\n"
            "Train it first with:\n"
            "  uv run python train.py --arch corner --out checkpoints/corner"
        )

    print(f"Loading corner model from {corner_path}...")
    model = load_model(corner_path, device=device)
    n_layers = model.cfg["n_layers"]
    n_heads  = model.cfg["n_heads"]

    data    = load_split(Path(args.data) / f"{args.split}.npz")
    states  = data["states"].astype(np.float32)
    labels  = data["optimal_distance"].astype(int)
    inputs  = states_to_corners(states)

    # --- Baseline ---
    print("\nComputing baseline accuracy...")
    base_acc = baseline_accuracy(model, inputs, labels, device=device)
    print(f"Baseline val acc: {base_acc*100:.1f}%")

    # --- Ablation sweep ---
    print(f"\nAblating {n_layers * n_heads} heads (uniform attention)...")
    acc_ablated = np.zeros((n_layers, n_heads))
    for li in range(n_layers):
        for hi in range(n_heads):
            acc = accuracy_ablate_head(model, inputs, labels, li, hi, device=device)
            acc_ablated[li, hi] = acc
            drop = (base_acc - acc) * 100
            print(f"  L{li}H{hi}: {acc*100:.1f}%  (drop {drop:+.1f} pp)")

    acc_drop = base_acc - acc_ablated   # positive = head was helpful

    print("\nTop 3 most ablation-sensitive heads:")
    flat_idx = np.argsort(acc_drop.ravel())[::-1]
    top_heads = []
    for idx in flat_idx[:3]:
        li, hi = divmod(int(idx), n_heads)
        top_heads.append((li, hi))
        print(f"  L{li}H{hi}: drop {acc_drop[li, hi]*100:.1f} pp")

    # --- Mean attention patterns ---
    print("\nExtracting mean attention patterns...")
    mean_attn = mean_attention_per_head(model, inputs, n_layers, n_heads, device=device)

    # --- Specialization metrics ---
    metrics = head_specialization(mean_attn)
    print("\nSharpness (mean max attn weight per head):")
    for li in range(n_layers):
        row = "  ".join(f"H{hi}:{metrics['sharpness'][li,hi]:.3f}" for hi in range(n_heads))
        print(f"  L{li}:  {row}")

    # --- Distance modulation for top heads ---
    distances_show = [0, 3, 7, 11]
    distances_show = [d for d in distances_show if d <= labels.max()]
    print(f"\nExtracting attention by distance ({distances_show}) for top heads...")
    attn_by_dist = mean_attention_by_distance(
        model, inputs, labels, n_layers, n_heads, device=device
    )

    figs = [
        _build_ablation_fig(acc_drop, base_acc, n_layers, n_heads),
        _build_mean_attn_fig(mean_attn, n_layers, n_heads),
        _build_specialization_fig(metrics, n_layers, n_heads),
        _build_distance_mod_fig(attn_by_dist, top_heads, distances_show),
    ]

    out_path = Path(args.out_dir) / "corner_attn_results.html"
    write_page(figs, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 13: corner model head ablation.")
    p.add_argument("--checkpoint", default="checkpoints/corner/best.pt")
    p.add_argument("--data",    default="data")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="docs", dest="out_dir")
    main(p.parse_args())
