"""
Phase 14: Flat model mlp_0 neuron profiling.

For the top-K neurons in mlp_0 by direct logit attribution (DLA):
  1. Activation-by-distance — mean neuron activation at each optimal distance (0–11),
     z-scored per neuron, showing whether neurons are broadly tuned or selective.
  2. Feature correlations — Pearson r between each top neuron's activation and
     three geometric features: optimal_distance, face_solved_count, corner_oriented_count.
  3. Max-activating state profiles — for the top-5 neurons, identify the top-50
     maximally activating val states and show the distribution of face_solved and
     corner_oriented counts versus the full val-set baseline.

Output:
    docs/neuron_profile_results.html

Usage:
    uv run python -m interp.neuron_profile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cube import CORNER_STICKERS, IDX_TO_COLOR, SOLVED_STATE
from dataset import load_split
from model import load_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(states: np.ndarray) -> dict[str, np.ndarray]:
    """face_solved_count (0-6) and corner_oriented_count (0-8) per state."""
    N = len(states)
    face_solved = np.zeros(N, dtype=np.int32)
    for f in range(6):
        b = f * 4
        face_solved += (
            (states[:, b+1] == states[:, b]) &
            (states[:, b+2] == states[:, b]) &
            (states[:, b+3] == states[:, b])
        ).astype(np.int32)

    corner_oriented = np.zeros(N, dtype=np.int32)
    for pos in range(8):
        si = CORNER_STICKERS[pos][0]  # U/D-face sticker index
        corner_oriented += ((states[:, si] == 0) | (states[:, si] == 1)).astype(np.int32)

    return {"face_solved": face_solved, "corner_oriented": corner_oriented}


# ---------------------------------------------------------------------------
# Neuron activation caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_mlp0_activations(
    model,
    states: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return mlp_0 post-GELU activations (N, 4*d_model)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hook = "blocks.0.hook_resid_mid"
    accum: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start:start+batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=[hook])
        resid_mid = cache[hook]
        h = F.gelu(model.blocks[0].mlp.fc1(model.blocks[0].ln2(resid_mid)))
        accum.append(h.cpu().float().numpy())
    return np.concatenate(accum)


# ---------------------------------------------------------------------------
# Neuron DLA (mlp_0 only)
# ---------------------------------------------------------------------------

def compute_neuron_dla_mlp0(
    model,
    activations: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Mean correct-class DLA for each mlp_0 neuron. Returns (4*d_model,)."""
    W_U = model.head.weight.detach().cpu().float().numpy()        # (12, D)
    fc2_w = model.blocks[0].mlp.fc2.weight.detach().cpu().float().numpy()  # (D, 4D)
    u_correct = W_U[labels]                                        # (N, D)
    fc2_proj = u_correct @ fc2_w                                   # (N, 4D)
    return (activations * fc2_proj).mean(axis=0)                   # (4D,)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


def _build_dla_bar(dla: np.ndarray, top_k: int = 20):
    import plotly.graph_objects as go

    top_idx = np.argsort(dla)[::-1][:top_k]
    colors = ["#10b981" if dla[i] > 0 else "#ef4444" for i in top_idx]
    fig = go.Figure(go.Bar(
        x=[f"n{i}" for i in top_idx],
        y=dla[top_idx],
        marker_color=colors,
        hovertemplate="Neuron %{x}<br>Mean DLA: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top-{top_k} mlp_0 neurons by mean DLA (correct-class logit attribution)",
        xaxis_title="Neuron", yaxis_title="Mean DLA",
        height=360, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
    )
    return fig


def _build_activation_heatmap(
    activations: np.ndarray,
    labels: np.ndarray,
    top_neurons: list[int],
    dla: np.ndarray,
):
    import plotly.graph_objects as go

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

    row_labels = [f"n{n}  DLA={dla[n]:.3f}" for n in top_neurons]
    fig = go.Figure(go.Heatmap(
        z=matrix_z, x=[str(d) for d in range(n_classes)], y=row_labels,
        colorscale="RdBu_r", zmid=0, colorbar=dict(title="z-score"),
        hovertemplate="Neuron: %{y}<br>Distance: %{x}<br>z-score: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top-{K} mlp_0 neurons — mean activation by optimal distance (z-scored per neuron)",
        xaxis_title="Optimal distance", yaxis_title="Neuron",
        height=420, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
    )
    return fig


def _build_correlation_fig(
    activations: np.ndarray,
    labels: np.ndarray,
    features: dict[str, np.ndarray],
    top_neurons: list[int],
    dla: np.ndarray,
):
    import plotly.graph_objects as go

    feat_keys = ["optimal_distance", "face_solved", "corner_oriented"]
    feat_arrays = {
        "optimal_distance": labels.astype(np.float32),
        "face_solved": features["face_solved"].astype(np.float32),
        "corner_oriented": features["corner_oriented"].astype(np.float32),
    }
    colors = {"optimal_distance": "#3b82f6", "face_solved": "#10b981", "corner_oriented": "#f59e0b"}

    K = len(top_neurons)
    corrs = {k: np.zeros(K) for k in feat_keys}
    for ki, n in enumerate(top_neurons):
        act = activations[:, n]
        for k in feat_keys:
            arr = feat_arrays[k]
            corrs[k][ki] = np.corrcoef(act, arr)[0, 1]

    x_labels = [f"n{n}  DLA={dla[n]:.3f}" for n in top_neurons]
    traces = [
        go.Bar(name=k, x=x_labels, y=corrs[k], marker_color=colors[k])
        for k in feat_keys
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title="Pearson correlation: top mlp_0 neurons vs. geometric features",
        xaxis_title="Neuron", yaxis_title="Pearson r",
        barmode="group", height=380,
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
        legend=dict(bgcolor="#1e293b"),
    )
    fig.add_hline(y=0, line_color="#475569")
    return fig


def _build_enrichment_fig(
    activations: np.ndarray,
    features: dict[str, np.ndarray],
    top_neurons: list[int],
    top_n: int = 50,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_show = min(4, len(top_neurons))
    fig = make_subplots(
        rows=n_show, cols=2,
        subplot_titles=[
            title
            for ni in range(n_show)
            for title in (
                f"n{top_neurons[ni]} — face_solved_count (top {top_n} activating)",
                f"n{top_neurons[ni]} — corner_oriented_count (top {top_n} activating)",
            )
        ],
        vertical_spacing=0.07,
    )

    N = len(activations)
    for ni, neuron_idx in enumerate(top_neurons[:n_show]):
        top_idxs = np.argsort(activations[:, neuron_idx])[-top_n:]
        for fi, (fname, maxv) in enumerate([("face_solved", 6), ("corner_oriented", 8)]):
            vals = np.arange(maxv + 1)
            base = np.array([(features[fname] == v).sum() / N for v in vals])
            topd = np.array([(features[fname][top_idxs] == v).sum() / top_n for v in vals])
            row, col = ni + 1, fi + 1
            show = (ni == 0 and fi == 0)
            fig.add_trace(go.Bar(x=vals, y=base, name="Val set", marker_color="#475569",
                                 opacity=0.7, showlegend=show), row=row, col=col)
            fig.add_trace(go.Bar(x=vals, y=topd, name=f"Top-{top_n} activating",
                                 marker_color="#3b82f6", opacity=0.9, showlegend=show),
                          row=row, col=col)

    fig.update_layout(
        title=f"Feature distribution: top-{top_n} max-activating states vs. val-set baseline",
        height=220 * n_show, barmode="overlay",
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
        legend=dict(bgcolor="#1e293b"),
    )
    return fig


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
      <button class="nav-group-btn active">Analyses &#9662;</button>
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
        <a href="corner_attn_results.html">Phase 13 — Head Ablation</a>
        <a href="neuron_profile_results.html" aria-current="page">Phase 14 — Neuron Profile</a>
        <a href="corner_circuit_results.html">Phase 15 — Corner Circuit</a>
      </div>
    </div>
    <a href="progress_report.pdf">Progress Report &#8599;</a>
  </nav>
</header>"""

PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 14 — mlp_0 Neuron Profiling</h2>
  <p><strong>Question:</strong> What do the top direct-logit-attribution neurons in mlp_0 actually detect?</p>
  <p><strong>Method:</strong> Rank mlp_0 neurons by mean DLA (correct-class logit contribution). For the top neurons: compute mean activation by distance class (z-scored), Pearson correlation with geometric features, and the feature distribution of their 50 most-activating validation states.</p>
  <p><strong>Finding:</strong> Top neurons are distance-selective, not feature-selective — they correlate most strongly with optimal_distance and show sharp tuning curves peaking at high distances. face_solved and corner_oriented are epiphenomenal in the top neurons.</p>
</section>"""


def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 14 — Neuron Profile</title>\n"
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

    flat_path = Path(args.checkpoint)
    if not flat_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {flat_path}")

    print(f"Loading flat model from {flat_path}...")
    model = load_model(flat_path, device=device)
    model.eval()

    print("Loading val set...")
    data   = load_split(Path(__file__).resolve().parents[1] / "data" / "val.npz")
    states = data["states"].astype(np.float32)
    labels = data["optimal_distance"].astype(int)
    print(f"  {len(states)} val examples")

    print("Caching mlp_0 activations...")
    activations = get_mlp0_activations(model, states, device=device)
    print(f"  activations shape: {activations.shape}")

    print("Computing neuron DLA...")
    dla = compute_neuron_dla_mlp0(model, activations, labels)
    top_k = 10
    top_neurons = list(np.argsort(dla)[::-1][:top_k])
    print(f"  Top-{top_k} neurons: {top_neurons}")
    print(f"  DLA:                 {[round(float(dla[n]), 4) for n in top_neurons]}")

    print("Computing geometric features...")
    features = compute_features(states)

    print("Building figures...")
    figs = [
        _build_dla_bar(dla, top_k=20),
        _build_activation_heatmap(activations, labels, top_neurons, dla),
        _build_correlation_fig(activations, labels, features, top_neurons, dla),
        _build_enrichment_fig(activations, features, top_neurons),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_page(figs, out_path)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 14: mlp_0 neuron profiling.")
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"),
        help="Path to flat model checkpoint.",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "neuron_profile_results.html"),
        help="Output HTML path.",
    )
    main(parser.parse_args())
