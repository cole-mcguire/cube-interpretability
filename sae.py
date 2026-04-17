"""
Phase 5 (Option 3): Sparse Autoencoder (SAE) on residual stream activations.

Trains an overcomplete sparse autoencoder on each layer's residual stream to
find a dictionary of sparse features, then checks whether any features align
with known interpretable concepts (face_solved, corner_oriented, optimal_distance).

Architecture:
    pre-bias:  b_pre ∈ R^d_model  (subtracted before encoding, added after decoding)
    encoder:   ReLU(W_enc @ (x - b_pre) + b_enc)     → h ∈ R^d_hidden  (sparse)
    decoder:   W_dec @ h + b_pre                      → x̂ ∈ R^d_model
    loss:      ||x - x̂||² + λ·||h||₁
    constraint: decoder columns kept unit-norm after each step

Training follows Bricken et al. (2023) with neuron resampling to revive dead features.

Usage:
    uv run python sae.py                          # all layers, expansion=4
    uv run python sae.py --expansion 8            # larger dictionary
    uv run python sae.py --layer L2               # single layer
    uv run python sae.py --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_split
from model import load_model


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model  = d_model
        self.d_hidden = d_hidden

        self.b_pre   = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=False)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        # Initialize decoder columns to unit norm
        nn.init.kaiming_uniform_(self.decoder.weight)
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_pre))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h) + self.b_pre

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h    = self.encode(x)
        x_hat = self.decode(h)
        l2   = (x - x_hat).pow(2).sum(dim=-1)   # (B,) per-sample
        l1   = h.sum(dim=-1)                      # (B,) per-sample L1
        return h, x_hat, l2, l1


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
) -> dict[str, torch.Tensor]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    accum: dict[str, list[torch.Tensor]] = {k: [] for k in hook_names}
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for name in hook_names:
            accum[name].append(cache[name].cpu().float())
    return {n: torch.cat(accum[n]) for n in hook_names}


# ---------------------------------------------------------------------------
# Dead-feature resampling
# ---------------------------------------------------------------------------

def resample_dead_features(
    sae: SparseAutoencoder,
    acts: torch.Tensor,
    dead_mask: torch.Tensor,
    l1_coeff: float,
    device: torch.device,
    n_sample: int = 2048,
) -> int:
    """
    Reinitialize dead encoder rows using high-reconstruction-error inputs.
    Returns number of features resampled.
    """
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return 0

    sae.eval()
    idx = torch.randperm(len(acts))[:n_sample]
    x   = acts[idx].to(device)
    with torch.no_grad():
        _, _, l2, l1 = sae(x)
        loss_per = l2 + l1_coeff * l1                   # (n_sample,)
    # Top-loss inputs as resampling candidates
    top_idx = loss_per.topk(min(n_dead, n_sample)).indices
    x_top   = x[top_idx]                                 # (k, d_model)
    # Normalize to unit vectors
    x_norm  = F.normalize(x_top - sae.b_pre.detach(), dim=-1)

    dead_idx = dead_mask.nonzero(as_tuple=True)[0][:len(x_norm)]
    with torch.no_grad():
        sae.encoder.weight.data[dead_idx] = x_norm
        sae.encoder.bias.data[dead_idx]   = 0.0
        sae.decoder.weight.data[:, dead_idx] = x_norm.T
        sae._normalize_decoder()

    sae.train()
    return n_dead


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sae(
    acts_train: torch.Tensor,           # (N, d_model)
    d_hidden:   int,
    l1_coeff:   float,
    device:     torch.device,
    epochs:     int   = 50,
    lr:         float = 1e-3,
    batch_size: int   = 512,
    resample_interval: int = 2000,      # steps between resampling checks
    dead_threshold:    float = 1e-4,    # EMA activation freq below = dead
) -> tuple[SparseAutoencoder, list[dict]]:
    d_model = acts_train.shape[1]
    sae     = SparseAutoencoder(d_model, d_hidden).to(device)
    opt     = torch.optim.Adam(sae.parameters(), lr=lr)

    ds = TensorDataset(acts_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Exponential moving average of per-feature activation frequency
    ema_freq  = torch.zeros(d_hidden, device=device)
    ema_decay = 0.99
    step = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        sae.train()
        epoch_l2 = epoch_l1 = epoch_l0 = 0.0
        n_batches = 0

        for (x_batch,) in dl:
            x_batch = x_batch.to(device)
            h, x_hat, l2, l1 = sae(x_batch)

            loss = (l2 + l1_coeff * l1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae._normalize_decoder()

            with torch.no_grad():
                freq = (h > 0).float().mean(dim=0)
                ema_freq = ema_decay * ema_freq + (1 - ema_decay) * freq
                epoch_l2 += l2.mean().item()
                epoch_l1 += l1.mean().item()
                epoch_l0 += (h > 0).float().sum(dim=-1).mean().item()

            step += 1
            if step % resample_interval == 0:
                dead_mask = ema_freq < dead_threshold
                n_resampled = resample_dead_features(
                    sae, acts_train, dead_mask, l1_coeff, device
                )
                if n_resampled:
                    # Reset optimizer state for resampled params
                    opt = torch.optim.Adam(sae.parameters(), lr=lr)
                    ema_freq[dead_mask] = 0.0

            n_batches += 1

        dead_count = int((ema_freq < dead_threshold).sum().item())
        history.append({
            "epoch": epoch,
            "l2":    epoch_l2 / n_batches,
            "l1":    epoch_l1 / n_batches,
            "l0":    epoch_l0 / n_batches,
            "dead":  dead_count,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    epoch {epoch:3d}  l2={history[-1]['l2']:.4f}"
                f"  l1={history[-1]['l1']:.4f}"
                f"  l0={history[-1]['l0']:.1f}"
                f"  dead={dead_count}/{d_hidden}"
            )

    return sae, history


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_features(
    sae: SparseAutoencoder,
    acts_val: torch.Tensor,             # (N, d_model)
    concepts: dict[str, np.ndarray],    # name → (N,) or (N, k) float/bool
    device: torch.device,
    batch_size: int = 512,
    top_pct: float = 0.10,
) -> dict:
    """
    For each SAE feature:
      - Pearson correlation with each concept label
      - Mean concept value for top-activating vs all samples
    Returns analysis dict.
    """
    sae.eval()
    feat_acts_list: list[torch.Tensor] = []
    recon_l2_list:  list[torch.Tensor] = []

    for start in range(0, len(acts_val), batch_size):
        x = acts_val[start : start + batch_size].to(device)
        h, x_hat, l2, _ = sae(x)
        feat_acts_list.append(h.cpu())
        recon_l2_list.append(l2.cpu())

    feat_acts = torch.cat(feat_acts_list).numpy()   # (N, d_hidden)
    recon_l2  = torch.cat(recon_l2_list).numpy()    # (N,)

    N, d_hidden = feat_acts.shape

    # Reconstruction stats
    x_np    = acts_val.numpy()
    var_total = np.var(x_np, axis=0).sum()
    var_resid = np.mean(recon_l2)
    r_squared = 1.0 - var_resid / var_total if var_total > 0 else 0.0

    l0_mean = float((feat_acts > 0).sum(axis=1).mean())
    l0_med  = float(np.median((feat_acts > 0).sum(axis=1)))

    dead = int((feat_acts.max(axis=0) == 0).sum())

    # Pearson correlation: feature activations × concept labels
    # feat_acts: (N, d_hidden), concept: (N,)
    correlations: dict[str, np.ndarray] = {}
    top_diffs:    dict[str, np.ndarray] = {}    # mean(top) - mean(all)

    k = max(1, int(N * top_pct))

    for cname, clabels in concepts.items():
        if clabels.ndim == 2:
            # Multi-column concept: aggregate to mean across columns
            clabels_flat = clabels.mean(axis=1).astype(np.float32)
        else:
            clabels_flat = clabels.astype(np.float32)

        # Pearson r for each feature
        fa_centered = feat_acts - feat_acts.mean(axis=0, keepdims=True)
        cl_centered = clabels_flat - clabels_flat.mean()
        cov = (fa_centered * cl_centered[:, None]).mean(axis=0)
        std_f = feat_acts.std(axis=0) + 1e-8
        std_c = clabels_flat.std() + 1e-8
        correlations[cname] = cov / (std_f * std_c)     # (d_hidden,)

        # Top-activating mean concept vs global mean
        global_mean = clabels_flat.mean()
        top_means = np.zeros(d_hidden, dtype=np.float32)
        for fi in range(d_hidden):
            top_idx = np.argpartition(feat_acts[:, fi], -k)[-k:]
            top_means[fi] = clabels_flat[top_idx].mean()
        top_diffs[cname] = top_means - global_mean      # (d_hidden,)

    return {
        "feat_acts":   feat_acts,
        "r_squared":   r_squared,
        "l0_mean":     l0_mean,
        "l0_med":      l0_med,
        "dead":        dead,
        "correlations": correlations,   # concept → (d_hidden,)
        "top_diffs":    top_diffs,      # concept → (d_hidden,)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device=device)
    model.eval()
    n_layers = model.cfg["n_layers"]
    d_model  = model.cfg["d_model"]
    d_hidden = d_model * args.expansion

    all_hook_names   = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    all_layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    # Filter to requested layer(s)
    if args.layer == "all":
        hook_names   = all_hook_names
        layer_labels = all_layer_labels
    else:
        idx = all_layer_labels.index(args.layer)
        hook_names   = [all_hook_names[idx]]
        layer_labels = [args.layer]

    print(f"\nSAE config: d_model={d_model}  d_hidden={d_hidden} ({args.expansion}×)  λ={args.l1}")

    # Load data
    print("\nLoading data...")
    train_data = load_split(Path(args.data) / "train.npz")
    val_data   = load_split(Path(args.data) / "val.npz")

    # Concept labels for feature analysis
    FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
    concepts = {
        "optimal_distance":  val_data["optimal_distance"].astype(np.float32),
        "face_solved_mean":  val_data["face_solved"].astype(np.float32),
        "corner_orient_mean": val_data["corner_oriented"].astype(np.float32),
        **{f"face_{fn}": val_data["face_solved"][:, fi].astype(np.float32)
           for fi, fn in enumerate(FACE_NAMES)},
        **{f"corner_{ci}": val_data["corner_oriented"][:, ci].astype(np.float32)
           for ci in range(8)},
    }

    # Extract activations
    print("Extracting train activations...")
    train_acts = extract_activations(model, train_data["states"], hook_names, device=device)
    print("Extracting val activations...")
    val_acts   = extract_activations(model, val_data["states"],   hook_names, device=device)

    all_results: dict[str, dict] = {}
    trained_saes: dict[str, SparseAutoencoder] = {}

    for hook, lbl in zip(hook_names, layer_labels):
        print(f"\n{'─'*60}")
        print(f"Training SAE on layer: {lbl}  ({len(train_acts[hook]):,} train samples)")
        print(f"{'─'*60}")

        sae, history = train_sae(
            train_acts[hook], d_hidden, args.l1, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        )
        trained_saes[lbl] = sae

        print(f"\nAnalyzing features on val set...")
        results = analyze_features(sae, val_acts[hook], concepts, device)
        all_results[lbl] = results

        print(f"\n  Reconstruction R² = {results['r_squared']:.4f}")
        print(f"  Mean L0 (active features/sample): {results['l0_mean']:.1f} / {d_hidden}")
        print(f"  Median L0: {results['l0_med']:.1f}")
        print(f"  Dead features: {results['dead']} / {d_hidden}")

        # Top concept-aligned features per concept
        print(f"\n  Top features by |Pearson r| with each concept:")
        for cname, corrs in results["correlations"].items():
            if "_mean" in cname:
                continue
            top3_idx = np.argsort(np.abs(corrs))[-3:][::-1]
            top3_str = "  ".join(
                f"feat{i}(r={corrs[i]:+.3f})" for i in top3_idx
            )
            print(f"    {cname:>20}: {top3_str}")

    # ------------------------------------------------------------------
    # Summary: most concept-aligned features across all layers
    # ------------------------------------------------------------------
    print(f"\n\n── Concept alignment summary ────────────────────────────────────────")
    print(f"  Max |r| between any SAE feature and each concept, per layer\n")
    concept_names = [c for c in concepts if "_mean" not in c]
    header = f"  {'concept':>22}" + "".join(f"  {lbl:>8}" for lbl in layer_labels)
    print(header)
    for cname in concept_names:
        row = f"  {cname:>22}"
        for lbl in layer_labels:
            max_r = float(np.abs(all_results[lbl]["correlations"][cname]).max())
            row += f"  {max_r:8.3f}"
        print(row)

    # ------------------------------------------------------------------
    # Plotly
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            n_layers_shown = len(layer_labels)

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Reconstruction R² by layer",
                    "Mean L0 (active features / sample)",
                    "Max |Pearson r| with optimal_distance",
                    "Feature correlation heatmap (top 50 features, best layer)",
                ],
            )

            # R² and L0 by layer
            r2s = [all_results[lbl]["r_squared"]  for lbl in layer_labels]
            l0s = [all_results[lbl]["l0_mean"]    for lbl in layer_labels]
            fig.add_trace(go.Bar(x=layer_labels, y=r2s, showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=layer_labels, y=l0s, showlegend=False), row=1, col=2)

            # Max |r| with optimal_distance per layer
            max_r_dist = [
                float(np.abs(all_results[lbl]["correlations"]["optimal_distance"]).max())
                for lbl in layer_labels
            ]
            fig.add_trace(go.Bar(x=layer_labels, y=max_r_dist, showlegend=False), row=2, col=1)

            # Heatmap: concept × top-50 features for best layer
            best_lbl = layer_labels[int(np.argmax(r2s))]
            corr_matrix = np.stack([
                all_results[best_lbl]["correlations"][c] for c in concept_names
            ])                                              # (n_concepts, d_hidden)
            top50_idx = np.argsort(np.abs(corr_matrix).max(axis=0))[-50:]
            corr_sub  = corr_matrix[:, top50_idx]          # (n_concepts, 50)

            fig.add_trace(go.Heatmap(
                z=corr_sub,
                x=[f"f{i}" for i in top50_idx],
                y=concept_names,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="r", len=0.45, y=0.1),
            ), row=2, col=2)

            fig.update_yaxes(title_text="R²",         row=1, col=1)
            fig.update_yaxes(title_text="mean L0",    row=1, col=2)
            fig.update_yaxes(title_text="max |r|",    row=2, col=1)
            fig.update_layout(
                title=f"Phase 5 (Option 3): SAE — {args.expansion}× expansion  λ={args.l1}",
                height=850,
            )

            out = Path("sae_results.html")
            fig.write_html(str(out))
            print(f"\nPlot saved → {out}")
        except ImportError:
            print("\nplotly not installed; skipping plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5 Option 3: Sparse Autoencoder.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data",       default="data")
    p.add_argument("--layer",      default="all",
                   help="Layer to train SAE on: embed, L0, L1, L2, L3, or all")
    p.add_argument("--expansion",  type=int,   default=4,    help="d_hidden = expansion × d_model")
    p.add_argument("--l1",         type=float, default=2e-4, help="L1 sparsity coefficient")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=512,  dest="batch_size")
    p.add_argument("--no-plot",    action="store_true", dest="no_plot")
    run(p.parse_args())


if __name__ == "__main__":
    main()
