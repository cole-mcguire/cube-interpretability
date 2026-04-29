"""
Phase 9: Training dynamics / grokking analysis for CubeTransformer.

Requires re-training with per-epoch checkpoint saves:
    uv run cube-train --save-every 1

For each saved epoch checkpoint, computes:
  1. Val accuracy overall and per distance class
     (reveals which distances "snap" suddenly vs. improve gradually)
  2. Per-layer ridge probe MAE
     (shows when each layer starts linearly encoding distance)
  3. DLA per component
     (tracks when mlp_0 dominates and how the circuit assembles)

Plots:
  • Overall val accuracy + loss over epochs
  • Per-distance accuracy heatmap (distances × epochs) — the grokking view
  • Per-layer probe MAE over epochs
  • DLA per component over epochs

Output:
    docs/grokking_results.html

Usage:
    uv run python -m interp.grokking
    uv run python -m interp.grokking --checkpoint-dir checkpoints --n-per-class 300
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
from interp.circuit import component_names, hook_name_for, compute_dla


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-checkpoint analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_checkpoint(
    model,
    states: np.ndarray,
    labels: np.ndarray,
    n_layers: int,
    n_classes: int,
    device: torch.device,
    probe_train_n: int = 2000,
    batch_size: int = 512,
) -> dict:
    """
    Single forward pass; returns accuracy, probe MAE, and DLA for one checkpoint.
    """
    model.eval()

    comp_names = component_names(n_layers)
    comp_hooks = [hook_name_for(c) for c in comp_names]
    resid_hooks = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    all_hooks = list(dict.fromkeys(comp_hooks + resid_hooks))

    logits_list: list[np.ndarray] = []
    cache_accum: dict[str, list[np.ndarray]] = {h: [] for h in all_hooks}

    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        logits, cache = model.run_with_cache(batch, names_filter=all_hooks)
        logits_list.append(logits.cpu().float().numpy())
        for h in all_hooks:
            cache_accum[h].append(cache[h].cpu().float().numpy())

    all_logits = np.concatenate(logits_list)
    cache_data  = {h: np.concatenate(cache_accum[h]) for h in all_hooks}

    # 1. Accuracy
    preds    = all_logits.argmax(-1)
    val_acc  = float((preds == labels).mean())
    dist_acc = np.array([
        float((preds[labels == d] == d).mean()) if (labels == d).any() else 0.0
        for d in range(n_classes)
    ])

    # 2. DLA (reuse circuit.py implementation)
    comp_acts = {hook_name_for(c): cache_data[hook_name_for(c)] for c in comp_names}
    dla_per_example = compute_dla(model, comp_acts, labels, n_layers, device)
    dla_means = {c: float(dla_per_example[c].mean()) for c in comp_names}

    # 3. Per-layer ridge probe MAE
    # Fit on first probe_train_n examples, evaluate on the rest.
    probe_hooks = resid_hooks  # ["hook_embed", "blocks.0.hook_resid_post", ...]
    n_train = min(probe_train_n, len(labels) * 2 // 3)
    probe_mae = []
    for h in probe_hooks:
        acts = cache_data[h]
        reg  = Ridge(alpha=1.0)
        reg.fit(acts[:n_train], labels[:n_train])
        pred = reg.predict(acts[n_train:])
        probe_mae.append(mean_absolute_error(labels[n_train:], pred))

    return {
        "val_acc":   val_acc,
        "dist_acc":  dist_acc,
        "dla":       dla_means,
        "probe_mae": np.array(probe_mae),  # (n_layers+1,)  embed + L0…L(n-1)
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")


def _build_overview_fig(epochs: list[int], val_accs: list[float], history: list[dict]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Val accuracy over training", "Train / val loss over training"],
    )

    fig.add_trace(go.Scatter(
        x=epochs, y=[v * 100 for v in val_accs],
        mode="lines+markers", name="val acc",
        line=dict(color="#10b981", width=2),
        marker=dict(size=5),
        hovertemplate="epoch %{x}<br>val acc: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1, **_GRID)
    fig.update_xaxes(title_text="Epoch", row=1, col=1, **_GRID)

    train_losses = [h["train_loss"] for h in history]
    val_losses   = [h["val_loss"]   for h in history]
    hist_epochs  = [h["epoch"]      for h in history]

    fig.add_trace(go.Scatter(
        x=hist_epochs, y=train_losses,
        mode="lines", name="train loss",
        line=dict(color="#3b82f6", width=2),
        hovertemplate="epoch %{x}<br>train loss: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=hist_epochs, y=val_losses,
        mode="lines", name="val loss",
        line=dict(color="#f59e0b", width=2),
        hovertemplate="epoch %{x}<br>val loss: %{y:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=2, **_GRID)
    fig.update_xaxes(title_text="Epoch", row=1, col=2, **_GRID)

    fig.update_layout(
        title="Training overview",
        height=380, **_DARK,
    )
    return fig


def _build_dist_heatmap_fig(epochs: list[int], dist_acc_matrix: np.ndarray, n_classes: int):
    """dist_acc_matrix: (n_epochs, n_classes)"""
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(
        z=(dist_acc_matrix * 100).T,  # (n_classes, n_epochs)
        x=epochs,
        y=[f"d={d}" for d in range(n_classes)],
        colorscale="Viridis",
        zmin=0, zmax=100,
        colorbar=dict(title="accuracy (%)"),
        hovertemplate="epoch %{x}  distance %{y}<br>accuracy: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Per-distance accuracy over training — phase transitions appear as sudden row brightening",
        xaxis_title="Epoch",
        yaxis_title="Optimal distance",
        height=450, **_DARK,
        xaxis=dict(**_GRID),
        yaxis=dict(**_GRID),
    )
    return fig


def _build_probe_fig(epochs: list[int], probe_mae_matrix: np.ndarray, n_layers: int):
    """probe_mae_matrix: (n_epochs, n_layers+1) — embed + L0..L(n-1)"""
    import plotly.graph_objects as go

    layer_names  = ["embed"] + [f"L{i}" for i in range(n_layers)]
    layer_colors = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b", "#f43f5e"]

    fig = go.Figure()
    for i, (name, color) in enumerate(zip(layer_names, layer_colors)):
        fig.add_trace(go.Scatter(
            x=epochs, y=probe_mae_matrix[:, i],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f"{name}  epoch %{{x}}<br>MAE: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Ridge probe MAE per layer over training — lower = layer linearly encodes distance",
        xaxis_title="Epoch",
        yaxis_title="MAE (optimal distance)",
        height=420, **_DARK,
        xaxis=dict(**_GRID),
        yaxis=dict(**_GRID),
    )
    return fig


def _build_dla_fig(epochs: list[int], dla_matrix: dict[str, list[float]], n_layers: int):
    import plotly.graph_objects as go

    comp_names = component_names(n_layers)
    color_map = {
        "embed": "#3b82f6",
        **{f"attn_{i}": "#f59e0b" for i in range(n_layers)},
        **{f"mlp_{i}":  "#10b981" for i in range(n_layers)},
    }
    dash_map = {
        "embed": "solid",
        **{f"attn_{i}": ["solid","dash","dot","dashdot"][i % 4] for i in range(n_layers)},
        **{f"mlp_{i}":  ["solid","dash","dot","dashdot"][i % 4] for i in range(n_layers)},
    }

    fig = go.Figure()
    for comp in comp_names:
        fig.add_trace(go.Scatter(
            x=epochs, y=dla_matrix[comp],
            mode="lines", name=comp,
            line=dict(color=color_map[comp], dash=dash_map[comp], width=2),
            hovertemplate=f"{comp}  epoch %{{x}}<br>DLA: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="DLA per component over training — tracks when mlp_0 dominates the circuit",
        xaxis_title="Epoch",
        yaxis_title="Mean correct-class logit attribution",
        height=440,
        yaxis=dict(zeroline=True, zerolinecolor="#475569", **_GRID),
        xaxis=dict(**_GRID),
        **_DARK,
    )
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
    <a href="phase5a_results.html">Phase 5a — Patching</a>
    <a href="phase5b_results.html">Phase 5b — Lenses &amp; SAE</a>
    <a href="circuit_results.html">Phase 7 — Circuit</a>
    <a href="weights_results.html">Phase 8 — Weights</a>
    <a href="grokking_results.html" aria-current="page">Phase 9 — Grokking</a>
    <a href="next_move_results.html">Phase 10 — Next Move</a>
    <a href="progress_report.pdf">Progress Report ↗</a>
  </nav>
</header>"""


def write_grokking_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 9 — Training Dynamics</title>\n"
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

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_paths = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No epoch_NNN.pt checkpoints found in {ckpt_dir}/.\n"
            "Re-train with:  uv run cube-train --save-every 1"
        )
    print(f"Found {len(ckpt_paths)} epoch checkpoints: "
          f"{ckpt_paths[0].name} … {ckpt_paths[-1].name}")

    # Load val set once; subsample per-class
    data_path = Path(args.data_dir) / f"{args.split}.npz"
    data   = load_split(data_path)
    states = data["states"].astype(np.float32)
    labels = data["optimal_distance"].astype(int)

    np.random.seed(42)
    keep = []
    for d in range(12):
        idx = np.where(labels == d)[0]
        if len(idx) > args.n_per_class:
            idx = np.random.choice(idx, args.n_per_class, replace=False)
        keep.append(idx)
    keep   = np.concatenate(keep)
    np.random.shuffle(keep)
    states = states[keep]
    labels = labels[keep]
    print(f"Evaluating on {len(states):,} examples ({args.n_per_class}/class max)\n")

    # Load one checkpoint to read model config and full training history
    last_ckpt = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=False)
    history   = last_ckpt["history"]
    n_classes = last_ckpt["config"]["n_classes"]

    # Run analysis on every epoch checkpoint
    epochs_list:    list[int]              = []
    val_accs:       list[float]            = []
    dist_acc_rows:  list[np.ndarray]       = []
    probe_mae_rows: list[np.ndarray]       = []
    dla_cols:       dict[str, list[float]] = {}

    for path in ckpt_paths:
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt["epoch"]
        model = load_model(path, device=device)
        model.eval()
        n_layers = model.cfg["n_layers"]

        print(f"  epoch {epoch:3d} …", end=" ", flush=True)
        result = analyze_checkpoint(
            model, states, labels, n_layers, n_classes, device,
            probe_train_n=args.probe_train_n,
        )
        print(f"val={result['val_acc']*100:.1f}%  "
              f"probe_mae(embed)={result['probe_mae'][0]:.3f}")

        epochs_list.append(epoch)
        val_accs.append(result["val_acc"])
        dist_acc_rows.append(result["dist_acc"])
        probe_mae_rows.append(result["probe_mae"])
        for comp, v in result["dla"].items():
            dla_cols.setdefault(comp, []).append(v)

    dist_acc_matrix  = np.stack(dist_acc_rows)   # (n_epochs, n_classes)
    probe_mae_matrix = np.stack(probe_mae_rows)  # (n_epochs, n_layers+1)

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.plot:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating plots…")
        fig_overview = _build_overview_fig(epochs_list, val_accs, history)
        fig_dist     = _build_dist_heatmap_fig(epochs_list, dist_acc_matrix, n_classes)
        fig_probe    = _build_probe_fig(epochs_list, probe_mae_matrix, n_layers)
        fig_dla      = _build_dla_fig(epochs_list, dla_cols, n_layers)
        write_grokking_page(
            [fig_overview, fig_dist, fig_probe, fig_dla],
            out_dir / "grokking_results.html",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 9: Training dynamics / grokking")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--data-dir",       default="data")
    p.add_argument("--split",          default="val", choices=["train", "val", "test"])
    p.add_argument("--n-per-class",    type=int, default=300,
                   help="val examples per distance class (default 300)")
    p.add_argument("--probe-train-n",  type=int, default=2000,
                   help="examples used to fit each ridge probe (default 2000)")
    p.add_argument("--out-dir",        default="docs")
    p.add_argument("--no-plot",        dest="plot", action="store_false")
    p.set_defaults(plot=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
