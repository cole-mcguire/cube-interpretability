"""
Phase 16: Next-move prediction model — accuracy, implicit distance encoding, and move distribution.

Analyses:
  1. Accuracy by distance — how does next-move prediction accuracy decay as optimal
     distance grows? (Harder states may have more valid first moves, so we also track
     top-k accuracy.)
  2. Implicit distance probe — even though the next-move model was trained only on the
     next move label, does its residual stream linearly encode optimal distance?
     Fit ridge regression probes at each layer and compare MAE to the distance model.
  3. Move distribution — which moves does the model predict most often, and does that
     match the empirical distribution of optimal first moves in the val set?

Output:
    docs/next_move_analysis_results.html

Usage:
    uv run python -m interp.next_move_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from dataset import load_split
from model import load_model

ROOT = Path(__file__).resolve().parents[1]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


MOVE_NAMES = ["U", "U'", "U2", "D", "D'", "D2",
              "F", "F'", "F2", "B", "B'", "B2",
              "L", "L'", "L2", "R", "R'", "R2"]


# ---------------------------------------------------------------------------
# Accuracy analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_accuracy_stats(
    model,
    states: np.ndarray,
    move_labels: np.ndarray,
    dist_labels: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_logits, all_preds = [], []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start:start+batch_size]).to(device)
        logits = model(batch).cpu().float().numpy()
        all_logits.append(logits)
        all_preds.append(logits.argmax(-1))

    all_logits = np.concatenate(all_logits)   # (N, 18)
    all_preds  = np.concatenate(all_preds)    # (N,)

    # Top-1 accuracy overall and per distance
    top1_overall = float((all_preds == move_labels).mean())
    max_dist = int(dist_labels.max())
    top1_by_dist = np.zeros(max_dist + 1)
    top3_by_dist = np.zeros(max_dist + 1)
    count_by_dist = np.zeros(max_dist + 1)
    for d in range(max_dist + 1):
        mask = dist_labels == d
        if not mask.any():
            continue
        count_by_dist[d] = mask.sum()
        top1_by_dist[d]  = (all_preds[mask] == move_labels[mask]).mean()
        top3 = np.argsort(all_logits[mask], axis=-1)[:, -3:]
        top3_by_dist[d]  = np.array([move_labels[mask][i] in top3[i] for i in range(mask.sum())]).mean()

    # Predicted move distribution
    pred_counts = np.bincount(all_preds, minlength=18)
    true_counts = np.bincount(move_labels, minlength=18)

    return {
        "top1_overall": top1_overall,
        "top1_by_dist": top1_by_dist,
        "top3_by_dist": top3_by_dist,
        "count_by_dist": count_by_dist,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "max_dist": max_dist,
    }


# ---------------------------------------------------------------------------
# Implicit distance probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_probe_mae(
    model,
    states: np.ndarray,
    dist_labels: np.ndarray,
    n_train: int = 4000,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Fit ridge probes at each residual stream position; return MAE array (n_layers+1,)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    n_layers = model.cfg["n_layers"]
    hook_names = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    cache_accum: dict[str, list] = {h: [] for h in hook_names}
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start:start+batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for h in hook_names:
            cache_accum[h].append(cache[h].cpu().float().numpy())

    probe_mae = []
    for h in hook_names:
        acts = np.concatenate(cache_accum[h])   # (N, d_model)
        reg = Ridge(alpha=1.0)
        reg.fit(acts[:n_train], dist_labels[:n_train])
        pred = reg.predict(acts[n_train:])
        probe_mae.append(mean_absolute_error(dist_labels[n_train:], pred))

    return np.array(probe_mae)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_accuracy_fig(stats: dict):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dists = list(range(stats["max_dist"] + 1))
    valid = [d for d in dists if stats["count_by_dist"][d] > 0]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Top-1 and Top-3 accuracy by optimal distance",
        "Val example count by distance",
    ])

    fig.add_trace(go.Scatter(
        x=valid, y=[stats["top1_by_dist"][d] * 100 for d in valid],
        mode="lines+markers", name="Top-1",
        line=dict(color="#3b82f6", width=2), marker=dict(size=6),
        hovertemplate="d=%{x}  Top-1: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=valid, y=[stats["top3_by_dist"][d] * 100 for d in valid],
        mode="lines+markers", name="Top-3",
        line=dict(color="#10b981", width=2, dash="dash"), marker=dict(size=6),
        hovertemplate="d=%{x}  Top-3: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=valid, y=[int(stats["count_by_dist"][d]) for d in valid],
        name="count", marker_color="#334155",
        hovertemplate="d=%{x}  n=%{y}<extra></extra>",
    ), row=1, col=2)

    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 105], row=1, col=1, **_GRID)
    fig.update_xaxes(title_text="Optimal distance", row=1, col=1, **_GRID)
    fig.update_yaxes(title_text="Count", row=1, col=2, **_GRID)
    fig.update_xaxes(title_text="Optimal distance", row=1, col=2, **_GRID)
    fig.update_layout(
        title=f"Next-move model: accuracy by distance (overall Top-1: {stats['top1_overall']*100:.1f}%)",
        height=400, **_DARK,
    )
    return fig


def _build_probe_fig(probe_mae: np.ndarray, n_layers: int):
    import plotly.graph_objects as go

    layer_labels = ["Embed"] + [f"L{i}" for i in range(n_layers)]
    fig = go.Figure(go.Bar(
        x=layer_labels, y=probe_mae.tolist(),
        marker_color="#8b5cf6",
        text=[f"{v:.3f}" for v in probe_mae],
        textposition="outside",
        hovertemplate="%{x}<br>Probe MAE: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Ridge probe MAE on distance — implicit distance encoding in next-move model<br>"
              "<sup>Lower MAE = residual stream more linearly encodes distance, despite not being trained on it</sup>",
        xaxis=dict(title="Layer", **_GRID),
        yaxis=dict(title="Probe MAE (moves)", **_GRID),
        height=380, **_DARK,
    )
    return fig


def _build_move_dist_fig(stats: dict):
    import plotly.graph_objects as go

    pred_pct = stats["pred_counts"] / stats["pred_counts"].sum() * 100
    true_pct = stats["true_counts"] / stats["true_counts"].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=MOVE_NAMES, y=pred_pct.tolist(), name="Predicted",
        marker_color="#3b82f6", opacity=0.85,
        hovertemplate="%{x}: predicted %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=MOVE_NAMES, y=true_pct.tolist(), name="True (optimal)",
        marker_color="#10b981", opacity=0.85,
        hovertemplate="%{x}: true %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted vs true move distribution in the val set",
        xaxis=dict(title="Move", **_GRID),
        yaxis=dict(title="Frequency (%)", **_GRID),
        barmode="group",
        height=400, **_DARK,
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
        <a href="corner_model_results.html">Phase 12 — Corner Model</a>
        <a href="corner_attn_results.html">Phase 13 — Head Ablation</a>
        <a href="neuron_profile_results.html">Phase 14 — Neuron Profile</a>
        <a href="corner_circuit_results.html">Phase 15 — Corner Circuit</a>
        <a href="next_move_analysis_results.html" aria-current="page">Phase 16 — Next-Move Analysis</a>
        <a href="llm_eval_results.html">Phase 17 — LLM Eval</a>
      </div>
    </div>
    <a href="progress_report.pdf">Final Report &#8599;</a>
  </nav>
</header>"""

PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 16 — Next-Move Prediction: Accuracy and Implicit Distance Encoding</h2>
  <p><strong>Question:</strong> How does the next-move model perform across difficulty levels, and does it implicitly learn to encode distance even though it was only trained on the next-move label?</p>
  <p><strong>Method:</strong> (1) Top-1 and Top-3 accuracy by optimal distance. (2) Ridge probes on residual stream activations at each layer, measuring how well distance can be read off linearly. (3) Predicted vs true move distribution to check for systematic biases.</p>
  <p><strong>Hypothesis:</strong> Distance should be implicitly encoded by at least the final layers, since accurate next-move selection in complex states requires reasoning about how far the cube is from solved.</p>
</section>"""


def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 16 — Next-Move Analysis</title>\n"
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

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading next-move model from {ckpt}...")
    model = load_model(ckpt, device=device)
    model.eval()

    print("Loading val set...")
    data = load_split(Path(__file__).resolve().parents[1] / "data" / "val.npz")
    states      = data["states"].astype(np.float32)
    move_labels = data["next_moves"].astype(int)
    dist_labels = data["optimal_distance"].astype(int)
    print(f"  {len(states)} val examples")

    print("Computing accuracy statistics...")
    stats = compute_accuracy_stats(model, states, move_labels, dist_labels, device=device)
    print(f"  Overall Top-1: {stats['top1_overall']*100:.1f}%")
    for d in range(stats["max_dist"] + 1):
        if stats["count_by_dist"][d] > 0:
            print(f"  d={d}: top1={stats['top1_by_dist'][d]*100:.1f}%  "
                  f"top3={stats['top3_by_dist'][d]*100:.1f}%  "
                  f"n={int(stats['count_by_dist'][d])}")

    print("Computing implicit distance probes...")
    probe_mae = compute_probe_mae(model, states, dist_labels, device=device)
    n_layers = model.cfg["n_layers"]
    layer_labels = ["Embed"] + [f"L{i}" for i in range(n_layers)]
    for label, mae in zip(layer_labels, probe_mae):
        print(f"  {label}: MAE={mae:.4f}")

    print("\nBuilding figures...")
    figs = [
        _build_accuracy_fig(stats),
        _build_probe_fig(probe_mae, n_layers),
        _build_move_dist_fig(stats),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_page(figs, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 16: next-move model analysis.")
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "checkpoints" / "next_move" / "best.pt"),
    )
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "next_move_analysis_results.html"),
    )
    main(parser.parse_args())
