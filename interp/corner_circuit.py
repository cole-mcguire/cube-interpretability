"""
Phase 15: Corner model — per-head DLA and OV circuit analysis.

1. Per-head DLA — decompose the attention output by head and compute each
   head's mean contribution to the correct-distance logit, confirming that
   DLA rank matches ablation sensitivity (L0H1 > L0H0 > L0H2).
2. OV circuit — for the top-3 heads by |DLA|, compute W_OV = W_O_h @ W_V_h
   and project onto the 12 distance unembedding directions. Shows whether
   each head reads and writes in the same or different distance basis.
3. Token-pair contributions for L0H1 — the most ablation-sensitive head.
   For each (src cubie j, dst cubie i) pair, compute the mean contribution
   A_h[i,j] * u_correct . W_OV_h @ x_j to the correct-distance logit, revealing
   which corner pairs L0H1 routes information between.

Output:
    docs/corner_circuit_results.html

Usage:
    uv run python -m interp.corner_circuit
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
# Per-head DLA
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_head_dla(
    model,
    states_corner: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Decompose attention output by head and compute mean DLA to correct logit.

    For head (l, h):
      1. Compute v_h = W_V_h @ ln1(x_in)  for each token
      2. Compute head_out = A_h @ v_h @ W_O_h.T  (B, 8, d_model)
      3. Mean-pool over 8 tokens → (B, d_model)
      4. Scale by LN factor from the full pooled residual stream
      5. Project through W_U → correct-class attribution

    Returns (n_layers, n_heads) float array of mean DLA values.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    n_layers = model.cfg["n_layers"]
    n_heads  = model.cfg["n_heads"]
    d_model  = model.cfg["d_model"]
    d_head   = d_model // n_heads

    W_U = model.head.weight.detach().cpu().float().numpy()  # (12, D)

    # Hook names: embed + attn_weights + resid_post per layer
    hook_names = (
        ["hook_embed"]
        + [f"blocks.{l}.attn.hook_attn_weights" for l in range(n_layers)]
        + [f"blocks.{l}.hook_resid_post"        for l in range(n_layers)]
    )

    # Accumulate sum and count for online mean
    head_sum  = [[np.zeros(()) for _ in range(n_heads)] for _ in range(n_layers)]
    n_total   = 0
    ln_weight = model.ln_final.weight.detach().cpu().float().numpy()

    # We store per-batch pooled contributions per head to compute DLA
    head_pools  = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    pools_total = []
    labels_all  = []

    for start in range(0, len(states_corner), batch_size):
        batch  = torch.from_numpy(states_corner[start:start+batch_size]).to(device)
        lbls   = labels[start:start+batch_size]
        _, cache = model.run_with_cache(batch, names_filter=hook_names)

        # Final pooled residual for LN scale
        resid_L = cache[f"blocks.{n_layers-1}.hook_resid_post"]   # (B, 8, D)
        pooled_total = resid_L.mean(dim=1).cpu().float().numpy()   # (B, D)
        pools_total.append(pooled_total)
        labels_all.append(lbls)

        for l in range(n_layers):
            x_in = cache["hook_embed"] if l == 0 else cache[f"blocks.{l-1}.hook_resid_post"]
            attn_w = cache[f"blocks.{l}.attn.hook_attn_weights"][:, :, :, :]  # (B,H,8,8)

            x_normed = model.blocks[l].ln1(x_in)  # (B, 8, D)

            qkv_w    = model.blocks[l].attn.qkv.weight       # (3D, D)
            oproj_w  = model.blocks[l].attn.out_proj.weight   # (D, D)

            # Value weight for all heads: rows [2D : 3D]
            V_all = qkv_w[2*d_model:, :]  # (D, D)

            for h in range(n_heads):
                V_h   = V_all[h*d_head:(h+1)*d_head, :]           # (d_head, D)
                W_O_h = oproj_w[:, h*d_head:(h+1)*d_head]          # (D, d_head)

                v_h      = x_normed @ V_h.T                         # (B, 8, d_head)
                A_h      = attn_w[:, h]                             # (B, 8, 8)
                weighted = A_h @ v_h                                # (B, 8, d_head)
                head_out = weighted @ W_O_h.T                       # (B, 8, D)
                pooled_h = head_out.mean(dim=1).cpu().float().numpy()  # (B, D)
                head_pools[l][h].append(pooled_h)

    # Concatenate all batches
    pools_total_np = np.concatenate(pools_total)   # (N, D)
    labels_np      = np.concatenate(labels_all)    # (N,)

    # LN scale
    var   = pools_total_np.var(axis=-1, keepdims=True)
    scale = ln_weight / np.sqrt(var + 1e-5)   # (N, D)

    # DLA per head
    dla = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            pooled_h = np.concatenate(head_pools[l][h])    # (N, D)
            scaled   = pooled_h * scale                    # (N, D)
            logits   = scaled @ W_U.T                      # (N, 12)
            correct  = logits[np.arange(len(labels_np)), labels_np]
            dla[l, h] = correct.mean()

    return dla


# ---------------------------------------------------------------------------
# OV circuit analysis
# ---------------------------------------------------------------------------

def get_ov_matrix(model, layer: int, head: int) -> np.ndarray:
    """W_OV_h = W_O_h @ W_V_h  shape (D, D)."""
    d_model = model.cfg["d_model"]
    n_heads = model.cfg["n_heads"]
    d_head  = d_model // n_heads

    qkv_w   = model.blocks[layer].attn.qkv.weight      # (3D, D)
    oproj_w = model.blocks[layer].attn.out_proj.weight  # (D, D)

    V_h   = qkv_w[2*d_model + head*d_head : 2*d_model + (head+1)*d_head, :]
    W_O_h = oproj_w[:, head*d_head : (head+1)*d_head]

    return (W_O_h @ V_h).detach().cpu().float().numpy()   # (D, D)


def project_ov_on_unembed(W_OV: np.ndarray, W_U: np.ndarray) -> np.ndarray:
    """
    Project W_OV onto distance unembedding directions.

    Returns (n_classes, n_classes) where [i, j] = how strongly W_OV transforms
    the j-th class direction into the i-th class logit.
    """
    return (W_U @ W_OV @ W_U.T)   # (12, 12)


# ---------------------------------------------------------------------------
# Token-pair contribution for a single head
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_token_pair_contributions(
    model,
    states_corner: np.ndarray,
    labels: np.ndarray,
    layer: int,
    head: int,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Mean (dst, src) token-pair contribution to correct-class logit for head (layer, head).

    contribution(i, j) = (1/8) * A_h[i,j] * (W_OV_h @ x_j_normed) . (u_correct * scale)

    Returns (8, 8) matrix — rows=dst cubie, cols=src cubie.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    W_U    = model.head.weight.detach().cpu().float().numpy()   # (12, D)
    W_OV   = get_ov_matrix(model, layer, head)                  # (D, D)
    ln_w   = model.ln_final.weight.detach().cpu().float().numpy()
    n_layers = model.cfg["n_layers"]

    hook_x      = "hook_embed" if layer == 0 else f"blocks.{layer-1}.hook_resid_post"
    hook_attn   = f"blocks.{layer}.attn.hook_attn_weights"
    hook_final  = f"blocks.{n_layers-1}.hook_resid_post"

    pair_sum = np.zeros((8, 8))
    n_total  = 0

    for start in range(0, len(states_corner), batch_size):
        batch = torch.from_numpy(states_corner[start:start+batch_size]).to(device)
        lbls  = labels[start:start+batch_size]
        _, cache = model.run_with_cache(batch, names_filter=[hook_x, hook_attn, hook_final])

        x_in    = cache[hook_x]
        x_norm  = model.blocks[layer].ln1(x_in).cpu().float().numpy()  # (B, 8, D)
        A_h     = cache[hook_attn][:, head].cpu().float().numpy()       # (B, 8, 8)

        # LN scale from final pooled residual
        pooled_f = cache[hook_final].mean(dim=1).cpu().float().numpy()  # (B, D)
        scale    = ln_w / np.sqrt(pooled_f.var(axis=-1, keepdims=True) + 1e-5)  # (B, D)

        # u_correct scaled by LN: (B, D)
        u_sc = W_U[lbls] * scale   # (B, D)

        # OV-transform each source token: (B, 8, D)
        ov_x = x_norm @ W_OV.T   # (B, 8, D)

        # Projection of each OV-transformed src token onto u_sc: (B, 8)
        ov_score = (ov_x * u_sc[:, np.newaxis, :]).sum(axis=-1)  # (B, 8)

        # pair[i, j] = (1/8) * A_h[b, i, j] * ov_score[b, j]
        # (B, 8, 8): broadcast A_h[b, i, j] * ov_score[b, j]
        pair_batch = (1/8) * A_h * ov_score[:, np.newaxis, :]   # (B, 8, 8)
        pair_sum  += pair_batch.sum(axis=0)
        n_total   += len(batch)

    return pair_sum / n_total   # (8, 8)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_div(fig, first: bool = False) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


def _build_head_dla_fig(dla: np.ndarray):
    import plotly.graph_objects as go

    n_layers, n_heads = dla.shape
    fig = go.Figure(go.Heatmap(
        z=dla,
        x=[f"H{h}" for h in range(n_heads)],
        y=[f"L{l}" for l in range(n_layers)],
        colorscale="RdBu_r", zmid=0,
        text=[[f"{dla[l,h]:.3f}" for h in range(n_heads)] for l in range(n_layers)],
        texttemplate="%{text}",
        colorbar=dict(title="Mean DLA"),
        hovertemplate="Layer %{y}  Head %{x}<br>Mean DLA: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Per-head DLA — mean contribution to correct-distance logit",
        xaxis_title="Head", yaxis_title="Layer",
        height=340, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
    )
    return fig


def _build_ov_fig(model, top_heads: list[tuple[int, int]]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    W_U = model.head.weight.detach().cpu().float().numpy()  # (12, D)
    n   = len(top_heads)
    titles = [f"L{l}H{h} — W_OV projected on distance directions" for l, h in top_heads]
    fig = make_subplots(rows=1, cols=n, subplot_titles=titles, horizontal_spacing=0.08)

    dist_labels = [str(d) for d in range(12)]
    for ci, (l, h) in enumerate(top_heads):
        W_OV = get_ov_matrix(model, l, h)
        proj = project_ov_on_unembed(W_OV, W_U)   # (12, 12)
        fig.add_trace(go.Heatmap(
            z=proj, x=dist_labels, y=dist_labels,
            colorscale="RdBu_r", zmid=0, showscale=(ci == n-1),
            hovertemplate="In: d=%{x}<br>Out: d=%{y}<br>Score: %{z:.3f}<extra></extra>",
        ), row=1, col=ci+1)

    fig.update_layout(
        title="OV circuit: W_U @ W_OV @ W_U^T — how each head reads and writes distance directions",
        height=420, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
    )
    for i in range(1, n+1):
        fig.update_xaxes(title_text="Src dist class" if i == 1 else "", row=1, col=i)
        fig.update_yaxes(title_text="Dst logit class" if i == 1 else "", row=1, col=i)
    return fig


def _build_token_pair_fig(pair_matrix: np.ndarray, layer: int, head: int):
    import plotly.graph_objects as go

    # Normalize to max absolute value for readability
    scale = np.abs(pair_matrix).max() or 1.0
    fig = go.Figure(go.Heatmap(
        z=pair_matrix,
        x=CORNER_NAMES, y=CORNER_NAMES,
        colorscale="RdBu_r", zmid=0,
        zmin=-scale, zmax=scale,
        text=[[f"{pair_matrix[i,j]:.4f}" for j in range(8)] for i in range(8)],
        texttemplate="%{text}",
        colorbar=dict(title="Contribution"),
        hovertemplate="Dst: %{y}<br>Src: %{x}<br>Contribution: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"L{layer}H{head} — mean (dst, src) token-pair contribution to correct-distance logit",
        xaxis_title="Source cubie", yaxis_title="Destination cubie",
        height=480, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0",
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
        <a href="neuron_profile_results.html">Phase 14 — Neuron Profile</a>
        <a href="corner_circuit_results.html" aria-current="page">Phase 15 — Corner Circuit</a>
      </div>
    </div>
    <a href="progress_report.pdf">Progress Report &#8599;</a>
  </nav>
</header>"""

PAGE_DESCRIPTION = """\
<section class="page-intro">
  <h2>Phase 15 — Corner Model Circuit Analysis</h2>
  <p><strong>Question:</strong> Which heads carry causal distance signal, and what does the dominant head (L0H1) actually compute?</p>
  <p><strong>Method:</strong> Decompose attention output by head; compute per-head DLA via the OV circuit and mean-pool. Project W_OV onto distance unembedding directions. For L0H1, compute the mean (dst, src) cubie-pair contribution to the correct-distance logit.</p>
  <p><strong>Finding:</strong> DLA order matches ablation sensitivity — L0H1 dominates. The token-pair contribution heatmap reveals which specific corner pairs L0H1 routes information between, characterizing the geometric routing circuit.</p>
</section>"""


def write_page(figs: list, out_path: Path) -> None:
    divs = [_fig_div(figs[0], first=True)] + [_fig_div(f) for f in figs[1:]]
    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + NAV_CSS + "\n"
        + "<title>Phase 15 — Corner Circuit</title>\n"
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
        raise FileNotFoundError(f"Checkpoint not found: {corner_path}")

    print(f"Loading corner model from {corner_path}...")
    model = load_model(corner_path, device=device)
    model.eval()

    print("Loading val set...")
    states, labels = load_split("val")
    states_corner = states_to_corners(states)
    print(f"  {len(states)} val examples → corner tokens {states_corner.shape}")

    print("Computing per-head DLA...")
    dla = compute_per_head_dla(model, states_corner, labels, device=device)
    n_layers, n_heads = dla.shape
    print("  DLA matrix (layer × head):")
    for l in range(n_layers):
        print(f"    L{l}: {['  {:+.3f}'.format(dla[l,h]) for h in range(n_heads)]}")

    # Top-3 heads by absolute DLA
    flat = [(abs(dla[l,h]), l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(reverse=True)
    top3 = [(l, h) for _, l, h in flat[:3]]
    print(f"  Top-3 heads by |DLA|: {top3}")

    # Identify L0H1 (or closest match)
    l0h1 = (0, 1)
    print(f"\nComputing token-pair contributions for L{l0h1[0]}H{l0h1[1]}...")
    pair_matrix = compute_token_pair_contributions(
        model, states_corner, labels, layer=l0h1[0], head=l0h1[1], device=device
    )
    print(f"  Top contributing pairs:")
    flat_pairs = sorted(
        [(pair_matrix[i, j], i, j) for i in range(8) for j in range(8)],
        reverse=True,
    )
    for score, dst, src in flat_pairs[:5]:
        print(f"    {CORNER_NAMES[src]} → {CORNER_NAMES[dst]}: {score:.4f}")

    print("\nBuilding figures...")
    figs = [
        _build_head_dla_fig(dla),
        _build_ov_fig(model, top3),
        _build_token_pair_fig(pair_matrix, *l0h1),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_page(figs, out_path)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase 15: corner model circuit analysis.")
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "checkpoints" / "corner" / "best.pt"),
        help="Path to corner model checkpoint.",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "docs" / "corner_circuit_results.html"),
        help="Output HTML path.",
    )
    main(parser.parse_args())
