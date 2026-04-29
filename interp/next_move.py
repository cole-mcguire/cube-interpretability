"""
Phase 10: Next-move prediction variant.

Trains a model on next-move prediction (18 classes) and probes its residual
stream for optimal_distance — testing whether planning implicitly encodes
the distance representation that emerges under direct supervision.

Key question: does optimal_distance MAE per layer match the distance-supervised
model, or is that representation only built when you supervise on it directly?

Requires:
    checkpoints/best.pt               — distance-supervised model
    checkpoints/next_move/best.pt     — next-move model (train with:
                                        uv run python train.py --task next_move
                                                               --out checkpoints/next_move)

Output:
    docs/next_move_results.html

Usage:
    uv run python -m interp.next_move
    uv run python -m interp.next_move --checkpoint-nm checkpoints/next_move/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from dataset import load_split
from model import load_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model,
    states: np.ndarray,
    hook_names: list[str],
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    accum: dict[str, list[np.ndarray]] = {k: [] for k in hook_names}
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for name in hook_names:
            accum[name].append(cache[name].cpu().float().numpy())
    return {name: np.concatenate(accum[name], axis=0) for name in hook_names}


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def probe_regression(acts: np.ndarray, labels: np.ndarray, train_frac: float = 0.8) -> float:
    n = len(labels)
    n_tr = int(n * train_frac)
    sc = StandardScaler()
    X_tr = sc.fit_transform(acts[:n_tr])
    X_te = sc.transform(acts[n_tr:])
    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, labels[:n_tr].astype(float))
    return float(np.mean(np.abs(reg.predict(X_te) - labels[n_tr:].astype(float))))


def probe_binary(acts: np.ndarray, labels: np.ndarray, train_frac: float = 0.8) -> float:
    n = len(labels)
    n_tr = int(n * train_frac)
    sc = StandardScaler()
    X_tr = sc.fit_transform(acts[:n_tr])
    X_te = sc.transform(acts[n_tr:])
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(X_tr, labels[:n_tr])
    return float(clf.score(X_te, labels[n_tr:]))


def run_probes(
    model,
    states: np.ndarray,
    data: dict,
    n_layers: int,
    device: torch.device,
) -> dict:
    hook_names = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    acts = extract_activations(model, states, hook_names, device=device)

    opt_dist    = data["optimal_distance"].astype(np.int32)
    face_solved = data["face_solved"]
    corner_ori  = data["corner_oriented"]
    scr_depth   = data["scramble_depth"].astype(np.int32)

    opt_maes, scr_maes, face_means, corner_means = [], [], [], []
    for h in hook_names:
        opt_maes.append(probe_regression(acts[h], opt_dist))
        scr_maes.append(probe_regression(acts[h], scr_depth))
        face_means.append(float(np.mean([probe_binary(acts[h], face_solved[:, fi]) for fi in range(6)])))
        corner_means.append(float(np.mean([probe_binary(acts[h], corner_ori[:, ci]) for ci in range(8)])))

    return {
        "opt_mae":      opt_maes,
        "scr_mae":      scr_maes,
        "face_mean":    face_means,
        "corner_mean":  corner_means,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_training_fig(nm_history: list[dict], dist_history: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Val accuracy over training",
            "Val loss over training",
        ],
    )

    for h, label, color in [
        (nm_history,   "next-move (18 classes)", "#10b981"),
        (dist_history, "distance (12 classes)",  "#3b82f6"),
    ]:
        epochs    = [e["epoch"]    for e in h]
        val_accs  = [e["val_acc"]  for e in h]
        val_loses = [e["val_loss"] for e in h]

        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 for v in val_accs],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2), marker=dict(size=4),
            legendgroup=label,
            hovertemplate=f"{label}  epoch %{{x}}<br>val acc: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=epochs, y=val_loses,
            mode="lines", name=label,
            line=dict(color=color, width=2),
            legendgroup=label, showlegend=False,
            hovertemplate=f"{label}  epoch %{{x}}<br>val loss: %{{y:.4f}}<extra></extra>",
        ), row=1, col=2)

    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1, **_GRID)
    fig.update_xaxes(title_text="Epoch", row=1, col=1, **_GRID)
    fig.update_yaxes(title_text="Loss", row=1, col=2, **_GRID)
    fig.update_xaxes(title_text="Epoch", row=1, col=2, **_GRID)
    fig.update_layout(
        title="Training comparison — note: tasks differ (next-move accuracy ≠ distance accuracy)",
        height=380, **_DARK,
    )
    return fig


def _build_distance_probe_fig(
    nm_results: dict,
    dist_results: dict,
    layer_labels: list[str],
    baseline_mae: float,
):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_hline(
        y=baseline_mae,
        line=dict(color="#475569", dash="dot", width=1.5),
        annotation_text=f"predict-mean baseline ({baseline_mae:.3f})",
        annotation_position="bottom right",
    )

    for results, label, color, dash in [
        (nm_results,   "next-move model",  "#10b981", "solid"),
        (dist_results, "distance model",   "#3b82f6", "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=layer_labels, y=results["opt_mae"],
            mode="lines+markers", name=label,
            line=dict(color=color, dash=dash, width=2.5),
            marker=dict(size=7),
            hovertemplate=f"{label}  %{{x}}<br>MAE: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Optimal-distance probe MAE per layer — key result: does next-move model encode distance?",
        xaxis_title="Layer",
        yaxis_title="Ridge MAE (optimal distance)",
        height=420, **_DARK,
        xaxis=dict(**_GRID),
        yaxis=dict(**_GRID),
    )
    return fig


def _build_structural_probe_fig(
    nm_results: dict,
    dist_results: dict,
    layer_labels: list[str],
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "face_solved (mean acc over 6 faces)",
            "corner_oriented (mean acc over 8 corners)",
            "scramble_depth probe MAE",
        ],
    )

    pairs = [
        ("face_mean",   "Accuracy (%)", True,  1),
        ("corner_mean", "Accuracy (%)", True,  2),
        ("scr_mae",     "MAE",          False, 3),
    ]

    for key, ylab, pct, col in pairs:
        for results, label, color, dash in [
            (nm_results,   "next-move",  "#10b981", "solid"),
            (dist_results, "distance",   "#3b82f6", "dash"),
        ]:
            vals = [v * 100 for v in results[key]] if pct else results[key]
            fig.add_trace(go.Scatter(
                x=layer_labels, y=vals,
                mode="lines+markers", name=label,
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=5),
                legendgroup=label,
                showlegend=(col == 1),
                hovertemplate=f"{label}  %{{x}}<br>{ylab}: %{{y:.3f}}<extra></extra>",
            ), row=1, col=col)
        fig.update_yaxes(title_text=ylab, row=1, col=col, **_GRID)
        fig.update_xaxes(title_text="Layer", row=1, col=col, **_GRID)

    fig.update_layout(
        title="Structural feature probes — both models should encode face/corner features similarly",
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
    <a href="phase5a_results.html">Phase 5a — Patching</a>
    <a href="phase5b_results.html">Phase 5b — Lenses &amp; SAE</a>
    <a href="circuit_results.html">Phase 7 — Circuit</a>
    <a href="weights_results.html">Phase 8 — Weights</a>
    <a href="grokking_results.html">Phase 9 — Grokking</a>
    <a href="next_move_results.html" aria-current="page">Phase 10 — Next Move</a>
    <a href="progress_report.pdf">Progress Report ↗</a>
  </nav>
</header>"""


def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 10 — Next-Move Variant</title>\n"
        "</head>\n<body>\n"
        + NAV_BODY + "\n"
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

    nm_path   = Path(args.checkpoint_nm)
    dist_path = Path(args.checkpoint_dist)

    if not nm_path.exists():
        raise FileNotFoundError(
            f"Next-move checkpoint not found: {nm_path}\n"
            "Train it first with:\n"
            "  uv run python train.py --task next_move --out checkpoints/next_move"
        )
    if not dist_path.exists():
        raise FileNotFoundError(
            f"Distance checkpoint not found: {dist_path}\n"
            "Train it first with:  uv run cube-train"
        )

    print(f"Loading next-move model from {nm_path}...")
    nm_ckpt   = torch.load(nm_path,   map_location="cpu", weights_only=False)
    nm_model  = load_model(nm_path,   device=device)
    nm_history = nm_ckpt["history"]

    print(f"Loading distance model from {dist_path}...")
    dist_ckpt    = torch.load(dist_path, map_location="cpu", weights_only=False)
    dist_model   = load_model(dist_path, device=device)
    dist_history = dist_ckpt["history"]

    n_layers = nm_model.cfg["n_layers"]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    data_path = Path(args.data) / f"{args.split}.npz"
    print(f"Loading {args.split} split from {data_path}...")
    data   = load_split(data_path)
    states = data["states"].astype(np.float32)

    opt_dist = data["optimal_distance"].astype(np.int32)
    baseline_mae = float(np.mean(np.abs(opt_dist.astype(float) - opt_dist.mean())))
    print(f"  {len(states):,} samples  |  distance baseline MAE: {baseline_mae:.3f}")

    print("\nRunning probes on next-move model...")
    nm_results = run_probes(nm_model, states, data, n_layers, device)
    print("Running probes on distance model...")
    dist_results = run_probes(dist_model, states, data, n_layers, device)

    print("\nProbe results (optimal_distance MAE):")
    header = f"  {'Layer':>8}  {'next-move':>10}  {'distance':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for lbl, nm_mae, dist_mae in zip(layer_labels, nm_results["opt_mae"], dist_results["opt_mae"]):
        print(f"  {lbl:>8}  {nm_mae:10.3f}  {dist_mae:10.3f}")
    print(f"\n  Baseline (predict-mean): {baseline_mae:.3f}")

    figs = [
        _build_training_fig(nm_history, dist_history),
        _build_distance_probe_fig(nm_results, dist_results, layer_labels, baseline_mae),
        _build_structural_probe_fig(nm_results, dist_results, layer_labels),
    ]

    out_path = Path(args.out_dir) / "next_move_results.html"
    write_page(figs, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 10: next-move prediction variant analysis.")
    p.add_argument("--checkpoint-nm",   default="checkpoints/next_move/best.pt", dest="checkpoint_nm")
    p.add_argument("--checkpoint-dist", default="checkpoints/best.pt",           dest="checkpoint_dist")
    p.add_argument("--data",    default="data")
    p.add_argument("--split",   default="val", choices=["train", "val", "test"])
    p.add_argument("--out-dir", default="docs", dest="out_dir")
    main(p.parse_args())
