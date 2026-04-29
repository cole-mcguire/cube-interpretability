"""
Phase 8: Weight-level analysis for CubeTransformer.

Three analyses:

1. Embedding SVD
   SVD of the 128×144 embedding matrix W_E.  Top-8 right singular vectors,
   each reshaped to (24 stickers × 6 colors), reveal the principal input
   patterns the embedding captures.

2. Direct read-out path
   W_direct = W_U @ W_E  (12 × 144) bypasses all transformer blocks.
   Accuracy of this linear probe vs the full model (78.5%).
   Per-distance rows show what sticker-color patterns each distance class
   reads directly from the input.

3. Top neuron profiles
   For the highest-DLA neurons identified in Phase 7, compute:
     • Input profile:  fc1.weight[n] @ W_E → (24, 6)
       (what sticker-color patterns drive the neuron)
     • Output profile: W_U @ fc2.weight[:, n] → (12,)
       (which distances the neuron promotes / suppresses)

Output:
    docs/weights_results.html

Usage:
    uv run python -m interp.weights
    uv run python -m interp.weights --top-k 6 --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_split
from model import load_model


COLOR_LABELS = ["W", "Y", "O", "R", "G", "B"]
FACE_LABELS = [
    f"{face}{i}"
    for face in ["U", "D", "F", "B", "L", "R"]
    for i in range(4)
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Analysis 1: Embedding SVD
# ---------------------------------------------------------------------------

def embedding_svd(W_E: np.ndarray, top_k: int = 8):
    """Top-k right singular vectors of W_E, each reshaped to (24, 6)."""
    _, S, Vt = np.linalg.svd(W_E, full_matrices=False)
    return Vt[:top_k].reshape(top_k, 24, 6), S[:top_k]


# ---------------------------------------------------------------------------
# Analysis 2: Direct read-out path
# ---------------------------------------------------------------------------

def direct_readout(
    W_E: np.ndarray,
    W_U: np.ndarray,
    states: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return W_direct (n_classes × 144) and its accuracy on (states, labels)."""
    W_direct = W_U @ W_E
    preds = (states @ W_direct.T).argmax(axis=-1)
    acc = float((preds == labels).mean())
    return W_direct, acc


# ---------------------------------------------------------------------------
# Analysis 3: Neuron profiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_neuron_dla(
    model,
    states: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> list[np.ndarray]:
    """Mean per-neuron correct-class DLA for each MLP layer."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    W_U_np = model.head.weight.detach().cpu().float().numpy()
    hooks = [f"blocks.{i}.hook_resid_mid" for i in range(n_layers)]
    accum: dict[str, list[np.ndarray]] = {h: [] for h in hooks}
    all_lbls: list[np.ndarray] = []

    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hooks)
        for h in hooks:
            accum[h].append(cache[h].cpu().float().numpy())
        all_lbls.append(labels[start : start + batch_size])

    all_lbls_np = np.concatenate(all_lbls)
    result: list[np.ndarray] = []

    for i in range(n_layers):
        resid = np.concatenate(accum[hooks[i]])
        mlp   = model.blocks[i].mlp
        x_t   = torch.from_numpy(resid).to(device)
        h     = F.gelu(mlp.fc1(model.blocks[i].ln2(x_t))).cpu().float().numpy()
        fc2_w = mlp.fc2.weight.detach().cpu().float().numpy()
        u_c   = W_U_np[all_lbls_np]
        result.append((h * (u_c @ fc2_w)).mean(axis=0))

    return result


def top_neurons_by_dla(
    neuron_dla: list[np.ndarray], top_k: int
) -> list[tuple[int, int, float]]:
    entries = [
        (layer, n, float(v))
        for layer, dla in enumerate(neuron_dla)
        for n, v in enumerate(dla)
    ]
    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    return entries[:top_k]


def neuron_profiles(
    model,
    neurons: list[tuple[int, int, float]],
    W_E: np.ndarray,
    W_U: np.ndarray,
) -> list[dict]:
    profiles = []
    for layer, n, dla in neurons:
        mlp     = model.blocks[layer].mlp
        fc1_row = mlp.fc1.weight.detach().cpu().float().numpy()[n]   # (d_model,)
        fc2_col = mlp.fc2.weight.detach().cpu().float().numpy()[:, n] # (d_model,)
        profiles.append({
            "label":  f"L{layer}N{n}",
            "layer":  layer,
            "neuron": n,
            "dla":    dla,
            "input":  (fc1_row @ W_E).reshape(24, 6),  # (24, 6)
            "output": W_U @ fc2_col,                    # (n_classes,)
        })
    return profiles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_svd_fig(vecs: np.ndarray, singular_values: np.ndarray):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    top_k = len(vecs)
    cols  = 4
    rows  = (top_k + cols - 1) // cols
    titles = [f"SV {i+1}  σ={singular_values[i]:.1f}" for i in range(top_k)]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.12)

    for i, vec in enumerate(vecs):
        r, c = divmod(i, cols)
        lim  = float(np.abs(vec).max())
        fig.add_trace(go.Heatmap(
            z=vec, x=COLOR_LABELS, y=FACE_LABELS,
            colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
            showscale=(i == 0),
            hovertemplate="sticker=%{y}  color=%{x}<br>value=%{z:.4f}<extra></extra>",
        ), row=r + 1, col=c + 1)

    fig.update_layout(
        title="Embedding SVD — top right singular vectors (24 stickers × 6 colors)",
        height=200 * rows + 80, **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def _build_direct_fig(
    W_direct: np.ndarray,
    direct_acc: float,
    full_acc: float,
    n_classes: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.3, 0.7],
        subplot_titles=["Direct vs full model accuracy",
                        "W_direct rows (distance × sticker-color input weights)"],
    )

    fig.add_trace(go.Bar(
        x=["Direct  W_U@W_E", "Full model"],
        y=[direct_acc * 100, full_acc * 100],
        marker_color=["#f59e0b", "#10b981"],
        text=[f"{v*100:.1f}%" for v in [direct_acc, full_acc]],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text="Accuracy (%)", row=1, col=1)

    lim = float(np.abs(W_direct).max())
    fig.add_trace(go.Heatmap(
        z=W_direct,
        x=list(range(144)),
        y=[f"d={d}" for d in range(n_classes)],
        colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
        colorbar=dict(title="weight", x=1.01),
        hovertemplate="sticker-color dim %{x}<br>distance %{y}<br>weight=%{z:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.update_xaxes(title_text="Input dimension (sticker×6 + color)", row=1, col=2,
                     **_GRID)
    fig.update_yaxes(row=1, col=2, **_GRID)

    fig.update_layout(
        title="Direct read-out path: W_U @ W_E bypasses all transformer blocks",
        height=420, showlegend=False, **_DARK,
    )
    fig.update_xaxes(row=1, col=1, **_GRID)
    fig.update_yaxes(row=1, col=1, **_GRID)
    return fig


def _build_profiles_fig(profiles: list[dict], n_classes: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(profiles)
    titles = []
    for p in profiles:
        titles += [
            f"{p['label']} input profile (DLA {p['dla']:+.3f})",
            f"{p['label']} output profile",
        ]

    fig = make_subplots(
        rows=n, cols=2, column_widths=[0.55, 0.45],
        subplot_titles=titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    layer_colors = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e",
                    "#a78bfa", "#38bdf8", "#fb923c", "#4ade80"]

    for i, p in enumerate(profiles):
        row = i + 1
        col_c = layer_colors[p["layer"] % len(layer_colors)]

        # Left: input heatmap
        lim = float(np.abs(p["input"]).max()) or 1e-6
        fig.add_trace(go.Heatmap(
            z=p["input"], x=COLOR_LABELS, y=FACE_LABELS,
            colorscale="RdBu_r", zmid=0, zmin=-lim, zmax=lim,
            showscale=False,
            hovertemplate="sticker=%{y}  color=%{x}<br>value=%{z:.4f}<extra></extra>",
        ), row=row, col=1)

        # Right: output bar
        fig.add_trace(go.Bar(
            x=list(range(n_classes)), y=p["output"].tolist(),
            marker_color=col_c,
            hovertemplate="distance %{x}<br>logit contribution: %{y:.4f}<extra></extra>",
        ), row=row, col=2)

    fig.update_layout(
        title="Top neuron profiles — input receptive field and output projection",
        height=240 * n + 60, showlegend=False, **_DARK,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


NAV_CSS = """\
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
header { background: #1e293b; border-bottom: 1px solid #334155; padding: 10px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
header h1 { font-size: 14px; font-weight: 700; color: #f1f5f9; white-space: nowrap; }
nav { display: flex; gap: 6px; flex-wrap: wrap; }
nav a { color: #94a3b8; text-decoration: none; font-size: 11px; padding: 4px 10px; border-radius: 5px; background: #0f172a; border: 1px solid #334155; transition: color .15s, background .15s; white-space: nowrap; }
nav a:hover { color: #e2e8f0; background: #334155; }
nav a[aria-current="page"] { color: #f1f5f9; background: #334155; border-color: #475569; }
</style>"""

NAV_BODY = """\
<header>
  <h1>2×2 Cube Interpretability</h1>
  <nav>
    <a href="index.html">Visualizer</a>
    <a href="probe_results.html">Phase 4 — Probing</a>
    <a href="phase5_results.html">Phase 5 — Patching &amp; Lenses</a>
    <a href="circuit_results.html">Phase 7 — Circuit</a>
    <a href="weights_results.html" aria-current="page">Phase 8 — Weights</a>
    <a href="grokking_results.html">Phase 9 — Grokking</a>
    <a href="progress_report.pdf">Progress Report ↗</a>
  </nav>
</header>"""


def write_weights_page(
    svd_fig, direct_fig, profiles_fig, out_path: Path
) -> None:
    div_svd      = _fig_div(svd_fig,      first=True)
    div_direct   = _fig_div(direct_fig,   first=False)
    div_profiles = _fig_div(profiles_fig, first=False)

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 8 — Weight Analysis</title>\n"
        "</head>\n<body>\n"
        + NAV_BODY + "\n"
        + div_svd + "\n"
        + div_direct + "\n"
        + div_profiles + "\n"
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

    model = load_model(args.checkpoint, device=device)
    model.eval()
    n_layers  = model.cfg["n_layers"]
    n_classes = model.cfg["n_classes"]
    print(f"Model: {n_layers} layers, {n_classes} classes, {model.n_params:,} params")

    data_path = Path(args.data_dir) / f"{args.split}.npz"
    data      = load_split(data_path)
    states    = data["states"].astype(np.float32)
    labels    = data["optimal_distance"].astype(int)

    np.random.seed(42)
    keep = []
    for d in range(n_classes):
        idx = np.where(labels == d)[0]
        if len(idx) > args.n_per_class:
            idx = np.random.choice(idx, args.n_per_class, replace=False)
        keep.append(idx)
    keep   = np.concatenate(keep)
    states = states[keep]
    labels = labels[keep]
    print(f"Evaluating on {len(states):,} examples ({args.n_per_class}/class max)\n")

    # Full-model accuracy
    model.eval()
    with torch.no_grad():
        logits  = model(torch.from_numpy(states).to(device)).cpu().numpy()
    full_acc = float((logits.argmax(-1) == labels).mean())
    print(f"Full model accuracy: {full_acc*100:.1f}%")

    W_E = model.embed.weight.detach().cpu().float().numpy()    # (128, 144)
    W_U = model.head.weight.detach().cpu().float().numpy()     # (12, 128)

    # ── Analysis 1: SVD ──────────────────────────────────────────────────────
    print("\nComputing embedding SVD…")
    svd_vecs, svd_S = embedding_svd(W_E, top_k=8)
    print("Top-8 singular values:", ", ".join(f"{s:.2f}" for s in svd_S))

    # ── Analysis 2: Direct read-out ──────────────────────────────────────────
    print("\nComputing direct read-out path…")
    W_direct, direct_acc = direct_readout(W_E, W_U, states, labels)
    print(f"  Direct accuracy:    {direct_acc*100:.1f}%")
    print(f"  Full model accuracy: {full_acc*100:.1f}%")

    # ── Analysis 3: Neuron profiles ──────────────────────────────────────────
    print(f"\nComputing neuron DLA for top-{args.top_k} profiles…")
    neuron_dla = compute_neuron_dla(model, states, labels, n_layers, device=device)
    top_neurons = top_neurons_by_dla(neuron_dla, args.top_k)
    print("Top neurons by |DLA|:")
    for layer, n, dla in top_neurons:
        print(f"  L{layer}N{n:>4d}  DLA={dla:+.4f}")
    profiles = neuron_profiles(model, top_neurons, W_E, W_U)

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.plot:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating plots…")
        svd_fig      = _build_svd_fig(svd_vecs, svd_S)
        direct_fig   = _build_direct_fig(W_direct, direct_acc, full_acc, n_classes)
        profiles_fig = _build_profiles_fig(profiles, n_classes)
        write_weights_page(svd_fig, direct_fig, profiles_fig,
                           out_dir / "weights_results.html")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8: Weight-level analysis")
    p.add_argument("--checkpoint",  default="checkpoints/best.pt")
    p.add_argument("--data-dir",    default="data")
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--n-per-class", type=int, default=500)
    p.add_argument("--top-k",       type=int, default=4,
                   help="number of top neurons to profile (default 4)")
    p.add_argument("--out-dir",     default="docs")
    p.add_argument("--no-plot",     dest="plot", action="store_false")
    p.set_defaults(plot=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
