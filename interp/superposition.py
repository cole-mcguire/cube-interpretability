"""
Phase 11: Superposition analysis — SAE expansion sweep.

Sweeps SAE expansion ratios (d_hidden = k × d_model) to estimate how many
true features the model encodes. Reconstruction quality (R²) and active
feature count should plateau at the true feature count; additional capacity
yields more dead features without improving reconstruction.

Key question: at what expansion does R² stop improving? That plateau gives
an upper bound on the number of distinct features the model represents in
superposition.

Output: docs/superposition_results.html

Usage:
    uv run python -m interp.superposition
    uv run python -m interp.superposition --expansions 1,2,4,8,16 --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import load_split
from model import load_model
from interp.sae import train_sae, analyze_features, extract_activations


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    model,
    train_acts: dict[str, torch.Tensor],
    val_acts: dict[str, torch.Tensor],
    concepts: dict[str, np.ndarray],
    layer_labels: list[str],
    hook_names: list[str],
    expansions: list[int],
    epochs: int,
    l1_coeff: float,
    device: torch.device,
) -> dict[int, dict[str, dict]]:
    """
    Returns results[expansion][layer_label] = {r_squared, l0_mean, dead,
                                                d_hidden, correlations}
    """
    d_model = model.cfg["d_model"]
    results: dict[int, dict[str, dict]] = {}

    for exp in expansions:
        d_hidden = d_model * exp
        print(f"\n{'═'*60}")
        print(f"Expansion {exp}×  (d_hidden={d_hidden})")
        print(f"{'═'*60}")
        results[exp] = {}

        for hook, lbl in zip(hook_names, layer_labels):
            print(f"\n  Layer {lbl} — training SAE ({d_hidden} features, {epochs} epochs)...")
            sae, _ = train_sae(
                train_acts[hook], d_hidden, l1_coeff, device,
                epochs=epochs, lr=1e-3, batch_size=512,
            )
            analysis = analyze_features(sae, val_acts[hook], concepts, device)
            results[exp][lbl] = {
                "r_squared":    analysis["r_squared"],
                "l0_mean":      analysis["l0_mean"],
                "dead":         analysis["dead"],
                "d_hidden":     d_hidden,
                "correlations": analysis["correlations"],
            }
            active = d_hidden - analysis["dead"]
            print(
                f"    R²={analysis['r_squared']:.4f}  "
                f"L0={analysis['l0_mean']:.1f}  "
                f"dead={analysis['dead']}/{d_hidden}  "
                f"active={active}"
            )

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
_GRID = dict(gridcolor="#334155")

LAYER_COLORS = {
    "embed": "#94a3b8",
    "L0":    "#3b82f6",
    "L1":    "#10b981",
    "L2":    "#f59e0b",
    "L3":    "#f43f5e",
}


def _build_r2_fig(results: dict, expansions: list[int], layer_labels: list[str]):
    import plotly.graph_objects as go

    fig = go.Figure()
    for lbl in layer_labels:
        r2s = [results[exp][lbl]["r_squared"] for exp in expansions]
        fig.add_trace(go.Scatter(
            x=expansions, y=r2s,
            mode="lines+markers", name=lbl,
            line=dict(color=LAYER_COLORS.get(lbl, "#e2e8f0"), width=2.5),
            marker=dict(size=8),
            hovertemplate=f"{lbl}  %{{x}}×<br>R²: %{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Reconstruction R² vs expansion ratio — plateau = sufficient dictionary capacity",
        xaxis_title="Expansion ratio (d_hidden / d_model)",
        yaxis_title="Reconstruction R²",
        xaxis=dict(tickvals=expansions, ticktext=[f"{e}×" for e in expansions], **_GRID),
        yaxis=dict(range=[0, 1.05], **_GRID),
        height=420, **_DARK,
    )
    return fig


def _build_active_fig(results: dict, expansions: list[int], layer_labels: list[str]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Active features (non-dead) vs expansion",
            "Dead feature fraction vs expansion",
        ],
    )

    for lbl in layer_labels:
        color = LAYER_COLORS.get(lbl, "#e2e8f0")
        actives     = [results[exp][lbl]["d_hidden"] - results[exp][lbl]["dead"] for exp in expansions]
        dead_fracs  = [results[exp][lbl]["dead"] / results[exp][lbl]["d_hidden"] for exp in expansions]

        fig.add_trace(go.Scatter(
            x=expansions, y=actives,
            mode="lines+markers", name=lbl,
            line=dict(color=color, width=2),
            marker=dict(size=7),
            legendgroup=lbl,
            hovertemplate=f"{lbl}  %{{x}}×<br>active: %{{y}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=expansions, y=[f * 100 for f in dead_fracs],
            mode="lines+markers", name=lbl,
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=7),
            legendgroup=lbl, showlegend=False,
            hovertemplate=f"{lbl}  %{{x}}×<br>dead: %{{y:.1f}}%<extra></extra>",
        ), row=1, col=2)

    fig.update_xaxes(tickvals=expansions, ticktext=[f"{e}×" for e in expansions],
                     title_text="Expansion ratio", **_GRID, row=1, col=1)
    fig.update_xaxes(tickvals=expansions, ticktext=[f"{e}×" for e in expansions],
                     title_text="Expansion ratio", **_GRID, row=1, col=2)
    fig.update_yaxes(title_text="Active features", **_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Dead fraction (%)", range=[0, 105], **_GRID, row=1, col=2)
    fig.update_layout(
        title="Active and dead features vs expansion — active count plateaus at true feature count",
        height=420, **_DARK,
    )
    return fig


def _build_alignment_fig(
    results: dict,
    expansions: list[int],
    layer_labels: list[str],
    key_concepts: list[str],
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_concepts = len(key_concepts)
    fig = make_subplots(
        rows=1, cols=n_concepts,
        subplot_titles=[f"max |r|: {c}" for c in key_concepts],
    )

    for ci, concept in enumerate(key_concepts, start=1):
        for lbl in layer_labels:
            color = LAYER_COLORS.get(lbl, "#e2e8f0")
            max_rs = [
                float(np.abs(results[exp][lbl]["correlations"][concept]).max())
                for exp in expansions
            ]
            fig.add_trace(go.Scatter(
                x=expansions, y=max_rs,
                mode="lines+markers", name=lbl,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup=lbl,
                showlegend=(ci == 1),
                hovertemplate=f"{lbl}  %{{x}}×<br>max|r|: %{{y:.3f}}<extra></extra>",
            ), row=1, col=ci)
        fig.update_xaxes(tickvals=expansions, ticktext=[f"{e}×" for e in expansions],
                         title_text="Expansion ratio", **_GRID, row=1, col=ci)
        fig.update_yaxes(title_text="Max |Pearson r|", range=[0, 1.05],
                         **_GRID, row=1, col=ci)

    fig.update_layout(
        title="Concept alignment vs expansion — does a larger dictionary find stronger features?",
        height=420, **_DARK,
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
    <a href="next_move_results.html">Phase 10 — Next Move</a>
    <a href="superposition_results.html" aria-current="page">Phase 11 — Superposition</a>
    <a href="corner_results.html">Phase 12 — Corner Model</a>
    <a href="progress_report.pdf">Progress Report ↗</a>
  </nav>
</header>"""


def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 11 — Superposition</title>\n"
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

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    expansions = [int(e) for e in args.expansions.split(",")]
    print(f"Expansion sweep: {expansions}  |  epochs per SAE: {args.epochs}")

    model = load_model(ckpt_path, device=device)
    model.eval()
    n_layers = model.cfg["n_layers"]

    hook_names   = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    print("\nLoading data...")
    train_data = load_split(Path(args.data) / "train.npz")
    val_data   = load_split(Path(args.data) / "val.npz")

    # Optionally subsample train to speed up sweep
    train_states = train_data["states"]
    if args.max_train_samples and len(train_states) > args.max_train_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(train_states), args.max_train_samples, replace=False)
        train_states = train_states[idx]
        print(f"  Subsampled train to {len(train_states):,} samples")
    else:
        print(f"  Train samples: {len(train_states):,}")

    FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
    concepts = {
        "optimal_distance":   val_data["optimal_distance"].astype(np.float32),
        "face_solved_mean":   val_data["face_solved"].astype(np.float32),
        "corner_orient_mean": val_data["corner_oriented"].astype(np.float32),
        **{f"face_{fn}": val_data["face_solved"][:, fi].astype(np.float32)
           for fi, fn in enumerate(FACE_NAMES)},
        **{f"corner_{ci}": val_data["corner_oriented"][:, ci].astype(np.float32)
           for ci in range(8)},
    }
    key_concepts = ["optimal_distance", "face_solved_mean", "corner_orient_mean"]

    print("\nExtracting activations...")
    train_acts = extract_activations(model, train_states, hook_names, device=device)
    val_acts   = extract_activations(model, val_data["states"], hook_names, device=device)

    results = run_sweep(
        model, train_acts, val_acts, concepts,
        layer_labels, hook_names, expansions,
        epochs=args.epochs, l1_coeff=args.l1, device=device,
    )

    # Print summary table
    print(f"\n\n── Sweep summary: R² ────────────────────────────────────────")
    header = f"  {'expansion':>10}" + "".join(f"  {lbl:>8}" for lbl in layer_labels)
    print(header)
    for exp in expansions:
        row = f"  {exp:>9}×"
        for lbl in layer_labels:
            row += f"  {results[exp][lbl]['r_squared']:8.4f}"
        print(row)

    print(f"\n── Sweep summary: active features ───────────────────────────")
    print(header)
    for exp in expansions:
        row = f"  {exp:>9}×"
        for lbl in layer_labels:
            r = results[exp][lbl]
            row += f"  {r['d_hidden'] - r['dead']:>8d}"
        print(row)

    figs = [
        _build_r2_fig(results, expansions, layer_labels),
        _build_active_fig(results, expansions, layer_labels),
        _build_alignment_fig(results, expansions, layer_labels, key_concepts),
    ]

    out_path = Path(args.out_dir) / "superposition_results.html"
    write_page(figs, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 11: SAE expansion sweep for superposition analysis.")
    p.add_argument("--checkpoint",        default="checkpoints/best.pt")
    p.add_argument("--data",              default="data")
    p.add_argument("--expansions",        default="1,2,4,8,16",
                   help="Comma-separated expansion ratios (default: 1,2,4,8,16)")
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--l1",                type=float, default=2e-4)
    p.add_argument("--max-train-samples", type=int,   default=100_000,
                   dest="max_train_samples",
                   help="Subsample train set for speed (0 = use all)")
    p.add_argument("--out-dir",           default="docs", dest="out_dir")
    main(p.parse_args())
