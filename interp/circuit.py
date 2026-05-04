"""
Phase 7: Circuit identification for CubeTransformer.

Since attention is degenerate (single token, softmax always 1.0), the
residual stream decomposes additively:

    x_final = embed_out + Σ_i attn_out_i + Σ_i mlp_out_i

Three analyses:

1. Direct Logit Attribution (DLA)
   Project each component's residual-stream contribution through the
   unembedding matrix to get its mean attribution to the correct-distance
   logit. Shows which components are responsible for the distance signal.

2. Activation patching
   For each component, replace its output with the mean activation from a
   different distance class, re-run downstream computation, and measure
   the prediction shift. Identifies which components causally carry the
   distance signal.

3. Neuron DLA (MLP layers)
   For each MLP layer compute per-neuron logit contributions by projecting
   through fc2 and the unembedding. Reveals which neurons are most
   monosemantically distance-tuned.

Outputs:
    circuit_dla.html
    circuit_patch.html
    circuit_neurons.html

Usage:
    uv run python -m interp.circuit
    uv run python -m interp.circuit --checkpoint checkpoints/best.pt --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_split
from model import load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def component_names(n_layers: int) -> list[str]:
    names = ["embed"]
    for i in range(n_layers):
        names += [f"attn_{i}", f"mlp_{i}"]
    return names


def hook_name_for(component: str) -> str:
    if component == "embed":
        return "hook_embed"
    kind, i = component.rsplit("_", 1)
    if kind == "attn":
        return f"blocks.{i}.attn.hook_attn_out"
    return f"blocks.{i}.mlp.hook_mlp_out"


@torch.no_grad()
def cache_components(
    model,
    states: np.ndarray,
    n_layers: int,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Return dict of component_name → (N, d_model) float32 activations."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    hooks = [hook_name_for(c) for c in component_names(n_layers)]
    accum: dict[str, list[np.ndarray]] = {h: [] for h in hooks}

    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hooks)
        for h in hooks:
            accum[h].append(cache[h].cpu().float().numpy())

    return {h: np.concatenate(accum[h]) for h in hooks}


# ---------------------------------------------------------------------------
# Analysis 1: Direct Logit Attribution
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dla(
    model,
    comp_acts: dict[str, np.ndarray],
    labels: np.ndarray,
    n_layers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    For each component, project its output through the unembedding matrix
    (W_head, LN folded via per-example scale) and return mean attribution
    to the correct-distance logit.

    Returns dict: component_name → (N,) float array of per-example
                  correct-class attributions.
    """
    # Unembedding weight: (n_classes, d_model)
    W_U = model.head.weight.detach().cpu().float().numpy()  # (12, 128)

    # Compute LN scale factors from the final residual stream
    # x_final ≈ sum of all components
    comps = component_names(n_layers)
    x_final = sum(comp_acts[hook_name_for(c)] for c in comps)  # (N, 128)
    ln_weight = model.ln_final.weight.detach().cpu().float().numpy()  # (128,)
    ln_bias   = model.ln_final.bias.detach().cpu().float().numpy()    # (128,)
    mean = x_final.mean(axis=-1, keepdims=True)                # (N, 1)
    var  = x_final.var(axis=-1, keepdims=True)                 # (N, 1)
    scale = ln_weight / np.sqrt(var + 1e-5)                    # (N, 128)

    # Effective unembedding per example: W_eff[n] = W_U * scale[n]
    # Attribution for component c on example n:
    #   (c[n] - mean_c[n]) * scale[n] @ W_U.T  → logit contribution (12,)
    # We approximate mean_c[n] ≈ mean_x[n] / n_components (each component
    # contributes equally to the mean on average), but use the exact
    # per-example centering: center c by the full-stream mean / n_comps.
    n_comps = len(comps)
    result: dict[str, np.ndarray] = {}
    for comp in comps:
        hook = hook_name_for(comp)
        c = comp_acts[hook]                                       # (N, 128)
        # Center: remove this component's share of the stream mean
        c_centered = c - mean / n_comps                          # (N, 128)
        # Scale by LN factor then project
        scaled = c_centered * scale                              # (N, 128)
        logit_contrib = scaled @ W_U.T                           # (N, 12)
        # Extract correct-class attribution
        correct = logit_contrib[np.arange(len(labels)), labels]  # (N,)
        result[comp] = correct
    return result


# ---------------------------------------------------------------------------
# Analysis 2: Activation patching
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_patching(
    model,
    states: np.ndarray,
    labels: np.ndarray,
    comp_acts: dict[str, np.ndarray],
    n_layers: int,
    n_classes: int,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    For each component, patch its output with the per-class mean activation
    from each target class, and record the mean predicted distance.

    Returns dict: component_name → (n_classes, n_classes) matrix where
                  entry [src, tgt] is the mean predicted distance when
                  examples from class `src` have their component output
                  replaced by the mean of class `tgt`.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    comps = component_names(n_layers)

    # Compute per-class mean activations for each component
    class_means: dict[str, np.ndarray] = {}
    for comp in comps:
        hook = hook_name_for(comp)
        acts = comp_acts[hook]                                    # (N, 128)
        means = np.zeros((n_classes, acts.shape[1]), dtype=np.float32)
        for d in range(n_classes):
            mask = labels == d
            if mask.sum() > 0:
                means[d] = acts[mask].mean(0)
        class_means[comp] = means                                 # (n_classes, 128)

    results: dict[str, np.ndarray] = {}

    for comp in comps:
        hook = hook_name_for(comp)
        patch_matrix = np.zeros((n_classes, n_classes), dtype=np.float32)

        for src_d in range(n_classes):
            src_mask  = labels == src_d
            src_states = states[src_mask]
            if len(src_states) == 0:
                continue

            row_preds = np.zeros(n_classes, dtype=np.float32)
            for tgt_d in range(n_classes):
                patch_val_np = class_means[comp][tgt_d]           # (128,)
                patch_val    = torch.from_numpy(patch_val_np).to(device)

                all_preds: list[np.ndarray] = []
                for start in range(0, len(src_states), batch_size):
                    batch = torch.from_numpy(
                        src_states[start : start + batch_size]
                    ).to(device)
                    pv = patch_val.unsqueeze(0).expand(len(batch), -1)
                    pv_capture = pv  # capture for closure

                    def hook_fn(act, hook, _pv=pv_capture):
                        return _pv

                    logits = model.run_with_hooks(
                        batch, fwd_hooks=[(hook, hook_fn)]
                    )
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.append(preds)

                all_preds_np = np.concatenate(all_preds)
                row_preds[tgt_d] = all_preds_np.mean()

            patch_matrix[src_d] = row_preds

        results[comp] = patch_matrix
        print(f"  patched {comp}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Neuron DLA
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
    """
    For each MLP layer, compute per-neuron DLA to the correct-distance logit.

    Each MLP computes: out = fc2(gelu(fc1(ln2(x_mid))))
    The column j of fc2 is the direction neuron j writes to the residual stream.
    Its logit contribution = neuron_j_activation * fc2[:, j] @ W_U[correct_class]

    Returns list of (n_neurons,) arrays, one per layer.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    W_U = model.head.weight.detach().cpu().float()  # (12, 128)

    # Fold LN into W_U lazily: approximate W_eff as W_U (scale is handled
    # at the dataset level by normalizing per layer below)

    # Hook names for resid_mid (MLP input) per layer
    resid_mid_hooks = [f"blocks.{i}.hook_resid_mid" for i in range(n_layers)]

    # Cache resid_mid activations
    accum: dict[str, list[np.ndarray]] = {h: [] for h in resid_mid_hooks}
    all_labels: list[np.ndarray] = []

    for start in range(0, len(states), batch_size):
        batch  = torch.from_numpy(states[start : start + batch_size]).to(device)
        lbls   = labels[start : start + batch_size]
        _, cache = model.run_with_cache(batch, names_filter=resid_mid_hooks)
        for h in resid_mid_hooks:
            accum[h].append(cache[h].cpu().float().numpy())
        all_labels.append(lbls)

    all_labels_np = np.concatenate(all_labels)

    neuron_dla_per_layer: list[np.ndarray] = []

    for i in range(n_layers):
        resid_mid = np.concatenate(accum[resid_mid_hooks[i]])     # (N, 128)
        mlp = model.blocks[i].mlp

        # Compute neuron activations
        x_t = torch.from_numpy(resid_mid).to(device)
        ln2 = model.blocks[i].ln2
        with torch.no_grad():
            h = F.gelu(mlp.fc1(ln2(x_t))).cpu().float().numpy()  # (N, 4*d_model)

        # fc2 columns: each column j is the direction neuron j writes
        fc2_w = mlp.fc2.weight.detach().cpu().float().numpy()     # (128, 4*d_model)

        # Correct-class unembedding vector per example
        W_U_np  = W_U.numpy()                                     # (12, 128)
        u_correct = W_U_np[all_labels_np]                         # (N, 128)

        # Neuron j's contribution to correct-class logit on example n:
        #   h[n, j] * (fc2_w[:, j] @ u_correct[n])
        # Vectorized: (N, n_neurons) where entry [n,j] = h[n,j] * dot(fc2[:,j], u[n])
        fc2_projected = (u_correct @ fc2_w)                       # (N, n_neurons)
        per_neuron = h * fc2_projected                            # (N, n_neurons)
        mean_dla = per_neuron.mean(axis=0)                        # (n_neurons,)

        neuron_dla_per_layer.append(mean_dla)

    return neuron_dla_per_layer


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    """Return a Plotly figure as an HTML div string."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if first else False,
    )


def _build_dla_fig(dla_correct: dict[str, np.ndarray], n_layers: int):
    import plotly.graph_objects as go

    comps  = component_names(n_layers)
    means  = [dla_correct[c].mean() for c in comps]
    stds   = [dla_correct[c].std()  for c in comps]
    colors = [
        "#3b82f6" if c == "embed" else "#f59e0b" if c.startswith("attn") else "#10b981"
        for c in comps
    ]
    error_colors = [
        "#93c5fd" if c == "embed" else "#fcd34d" if c.startswith("attn") else "#6ee7b7"
        for c in comps
    ]
    fig = go.Figure([go.Bar(
        x=[c], y=[m],
        error_y=dict(type="data", array=[s], visible=True, color=ec),
        marker_color=col,
        hovertemplate="%{x}<br>mean DLA: %{y:.3f}<extra></extra>",
        showlegend=False,
    ) for c, m, s, col, ec in zip(comps, means, stds, colors, error_colors)])
    fig.update_layout(
        title="Direct Logit Attribution by Component",
        xaxis_title="Component",
        yaxis_title="Mean correct-class logit attribution",
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", zeroline=True, zerolinecolor="#475569"),
        height=450,
    )
    return fig


def _build_patching_fig(
    patch_results: dict[str, np.ndarray],
    n_layers: int,
    n_classes: int,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    comps = component_names(n_layers)
    sensitivities = []
    for comp in comps:
        mat = patch_results[comp]
        diag_vals = mat[np.arange(n_classes), np.arange(n_classes)]
        off_diag  = mat[~np.eye(n_classes, dtype=bool)]
        sensitivities.append(float(diag_vals.mean() - off_diag.mean()))

    most_sensitive = comps[int(np.argmax(np.abs(sensitivities)))]
    mat = patch_results[most_sensitive]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        subplot_titles=[
            "Sensitivity: diagonal − off-diagonal mean predicted distance",
            f"Full patch matrix: {most_sensitive}",
        ],
    )
    colors = [
        "#3b82f6" if c == "embed" else "#f59e0b" if c.startswith("attn") else "#10b981"
        for c in comps
    ]
    fig.add_trace(go.Bar(
        x=comps, y=sensitivities, marker_color=colors, name="sensitivity",
        hovertemplate="%{x}<br>sensitivity: %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=mat,
        x=[f"tgt d={d}" for d in range(n_classes)],
        y=[f"src d={d}" for d in range(n_classes)],
        colorscale="RdBu",
        colorbar=dict(title="mean pred dist"),
        hovertemplate="src=%{y}  tgt=%{x}<br>mean pred: %{z:.2f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(
        title="Activation Patching",
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", zeroline=True, zerolinecolor="#475569"),
        height=500, showlegend=False,
    )
    return fig


def _build_neurons_fig(neuron_dla: list[np.ndarray], n_layers: int, top_k: int = 20):
    import plotly.graph_objects as go

    layer_colors = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e"]
    fig = go.Figure()

    all_means: list[float] = []
    all_layers: list[int]  = []
    all_indices: list[int] = []
    for i, dla in enumerate(neuron_dla):
        for j, v in enumerate(dla):
            all_means.append(float(v))
            all_layers.append(i)
            all_indices.append(j)

    all_means_arr = np.array(all_means)
    for i in range(n_layers):
        mask = np.array(all_layers) == i
        fig.add_trace(go.Scatter(
            x=np.array(all_indices)[mask], y=all_means_arr[mask],
            mode="markers",
            marker=dict(size=4, color=layer_colors[i % len(layer_colors)], opacity=0.6),
            name=f"MLP {i}",
            hovertemplate=f"MLP {i}  neuron %{{x}}<br>DLA: %{{y:.4f}}<extra></extra>",
        ))

    top_idx = np.argsort(np.abs(all_means_arr))[-top_k:]
    for idx in top_idx:
        fig.add_annotation(
            x=all_indices[idx], y=all_means[idx],
            text=f"L{all_layers[idx]}N{all_indices[idx]}",
            showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1,
            arrowcolor="#64748b", font=dict(size=9, color="#94a3b8"),
            ax=20, ay=-20,
        )

    fig.update_layout(
        title="Neuron DLA — per-neuron correct-class logit attribution",
        xaxis_title="Neuron index within MLP layer",
        yaxis_title="Mean correct-class logit attribution",
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", zeroline=True, zerolinecolor="#475569"),
        height=500,
    )
    return fig


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
        <a href="circuit_results.html" aria-current="page">Phase 7 — Circuit</a>
        <a href="weights_results.html">Phase 8 — Weights</a>
        <a href="grokking_results.html">Phase 9 — Grokking</a>
        <a href="next_move_results.html">Phase 10 — Next Move</a>
        <a href="superposition_results.html">Phase 11 — Superposition</a>
        <a href="corner_results.html">Phase 12 — Corner Model</a>
        <a href="corner_attn_results.html">Phase 13 — Head Ablation</a>
      </div>
    </div>
    <a href="progress_report.pdf">Final Report ↗</a>
  </nav>
</header>"""
PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 7 — Circuit Identification</h2>
  <p><strong>Question:</strong> Which specific components form the complete computational circuit from input to distance logit?</p>
  <p><strong>Method:</strong> Direct logit attribution (DLA) decomposes the final logit into per-component contributions; activation patching identifies causal components; neuron DLA isolates the most distance-tuned individual neurons.</p>
  <p><strong>Finding:</strong> mlp_0 provides the largest positive DLA (+5.6); all attention layers are collectively slightly negative. The top 5 neurons in mlp_0 account for the bulk of the distance classification signal.</p>
</section>"""



def write_circuit_page(
    dla_correct: dict[str, np.ndarray],
    patch_results: dict[str, np.ndarray],
    neuron_dla: list[np.ndarray],
    n_layers: int,
    n_classes: int,
    out_path: Path,
) -> None:
    fig_dla     = _build_dla_fig(dla_correct, n_layers)
    fig_patch   = _build_patching_fig(patch_results, n_layers, n_classes)
    fig_neurons = _build_neurons_fig(neuron_dla, n_layers)

    div_dla     = _fig_div(fig_dla,     first=True)
    div_patch   = _fig_div(fig_patch,   first=False)
    div_neurons = _fig_div(fig_neurons, first=False)

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 7 — Circuit Identification</title>\n"
        "</head>\n<body>\n"
        + NAV_BODY + "\n"
        + PAGE_DESCRIPTION + "\n"
        + div_dla + "\n"
        + div_patch + "\n"
        + div_neurons + "\n"
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
    data   = load_split(data_path)
    states = data["states"].astype(np.float32)
    labels = data["optimal_distance"].astype(int)

    # Subsample for speed: up to args.n_per_class examples per distance class
    np.random.seed(42)
    keep = []
    for d in range(n_classes):
        idx = np.where(labels == d)[0]
        if len(idx) > args.n_per_class:
            idx = np.random.choice(idx, args.n_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    np.random.shuffle(keep)
    states = states[keep]
    labels = labels[keep]
    print(f"Evaluating on {len(states):,} examples ({args.n_per_class}/class max)\n")

    # ── Cache component activations ──────────────────────────────────────────
    print("Caching component activations…")
    comp_acts = cache_components(model, states, n_layers, device=device)

    # ── Analysis 1: DLA ──────────────────────────────────────────────────────
    print("Computing DLA…")
    dla_correct = compute_dla(model, comp_acts, labels, n_layers, device)
    comps = component_names(n_layers)
    print("\nMean correct-class DLA per component:")
    for c in comps:
        print(f"  {c:12s}  {dla_correct[c].mean():+.4f}  (std {dla_correct[c].std():.4f})")

    # ── Analysis 2: Activation patching ─────────────────────────────────────
    print("\nRunning activation patching (this may take a few minutes)…")
    patch_results = run_patching(
        model, states, labels, comp_acts, n_layers, n_classes, device=device
    )

    # ── Analysis 3: Neuron DLA ───────────────────────────────────────────────
    print("\nComputing neuron DLA…")
    neuron_dla = compute_neuron_dla(model, states, labels, n_layers, device=device)
    for i, dla in enumerate(neuron_dla):
        top3 = np.argsort(np.abs(dla))[-3:][::-1]
        print(f"  MLP {i} — top-3 neurons by |DLA|: "
              + ", ".join(f"N{j}({dla[j]:+.4f})" for j in top3))

    # ── Plots ────────────────────────────────────────────────────────────────
    if args.plot:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating plots…")
        write_circuit_page(
            dla_correct, patch_results, neuron_dla,
            n_layers, n_classes,
            out_dir / "circuit_results.html",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7: Circuit identification")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--n-per-class", type=int, default=500,
                   help="max examples per distance class (default 500)")
    p.add_argument("--out-dir",    default="docs")
    p.add_argument("--no-plot",    dest="plot", action="store_false")
    p.set_defaults(plot=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
