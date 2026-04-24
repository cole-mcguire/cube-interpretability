"""
Phase 5 (Option 1): Logit lens + Tuned lens.

Logit lens  — apply the model's final LN+head directly to each layer's
              residual stream. No training; free interpretability baseline.

Tuned lens  — per-layer affine transform T_l: d_model → d_model (init=identity)
              trained so that head(LN(T_l(h_l))) predicts optimally.
              Follows Belrose et al. 2023, trained against true labels.

Key questions:
  - At which layer does the model first predict the correct distance?
  - How does confidence grow across layers?
  - Do later layers only matter for hard (high-distance) inputs?

Usage:
    uv run python tuned_lens.py
    uv run python tuned_lens.py --checkpoint checkpoints/best.pt --no-plot
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
    """Return dict of hook_name → (N, d_model) float32 tensor (on CPU)."""
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
# Logit lens (no training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_lens(
    model,
    acts: dict[str, torch.Tensor],
    hook_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Apply model's final LN+head directly to each layer's residual stream."""
    model.eval()
    result: dict[str, torch.Tensor] = {}
    for name in hook_names:
        h = acts[name].to(device)
        logits = model.head(model.ln_final(h)).cpu()
        result[name] = logits                           # (N, n_classes)
    return result


# ---------------------------------------------------------------------------
# Tuned lens (one affine per layer, trained)
# ---------------------------------------------------------------------------

class TunedLensLayer(nn.Module):
    """Affine transform d_model → d_model, initialized to identity."""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_tuned_lens(
    model,
    acts: dict[str, torch.Tensor],
    hook_names: list[str],
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> dict[str, TunedLensLayer]:
    """Train one TunedLensLayer per hook, returns trained layers (on CPU)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = model.cfg["d_model"]
    trained: dict[str, TunedLensLayer] = {}

    for hook in hook_names:
        lens = TunedLensLayer(d_model).to(device)
        opt  = torch.optim.Adam(lens.parameters(), lr=lr)
        h    = acts[hook]                               # (N, d_model) on CPU
        ds   = TensorDataset(h, labels)
        dl   = DataLoader(ds, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            lens.train()
            total_loss = 0.0
            for x_batch, y_batch in dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model.head(model.ln_final(lens(x_batch)))
                loss   = F.cross_entropy(logits, y_batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += loss.item()

        trained[hook] = lens.cpu()
        print(f"  {hook:40s}  final loss = {total_loss / len(dl):.4f}")

    for p in model.parameters():
        p.requires_grad_(True)

    return trained


@torch.no_grad()
def apply_tuned_lens(
    model,
    acts: dict[str, torch.Tensor],
    lens_layers: dict[str, TunedLensLayer],
    hook_names: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    result: dict[str, torch.Tensor] = {}
    for name in hook_names:
        h      = acts[name].to(device)
        lens   = lens_layers[name].to(device)
        logits = model.head(model.ln_final(lens(h))).cpu()
        result[name] = logits
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def layer_accuracy(logits_dict: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, float]:
    return {
        name: float((logits.argmax(dim=-1) == labels).float().mean())
        for name, logits in logits_dict.items()
    }


def layer_confidence(logits_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Mean max softmax probability across samples."""
    return {
        name: float(logits.softmax(dim=-1).max(dim=-1).values.mean())
        for name, logits in logits_dict.items()
    }


def accuracy_by_distance(
    logits_dict: dict[str, torch.Tensor],
    labels: torch.Tensor,
    opt_dist: torch.Tensor,
) -> dict[str, dict[int, float]]:
    """Per true-distance accuracy for each layer."""
    result: dict[str, dict[int, float]] = {}
    for name, logits in logits_dict.items():
        preds = logits.argmax(dim=-1)
        correct = (preds == labels)
        by_d: dict[int, float] = {}
        for d in opt_dist.unique().tolist():
            mask = opt_dist == d
            if mask.sum() > 0:
                by_d[int(d)] = float(correct[mask].float().mean())
        result[name] = by_d
    return result


def commitment_layer(
    ll_logits: dict[str, torch.Tensor],
    final_preds: torch.Tensor,
    hook_names: list[str],
) -> torch.Tensor:
    """
    For each sample, first layer where logit-lens prediction matches final model output.
    Returns (N,) int tensor with layer index (0=embed, 1=L0, ...).
    """
    N = final_preds.shape[0]
    commit = torch.full((N,), len(hook_names) - 1, dtype=torch.long)
    for li, name in enumerate(hook_names):
        matches = ll_logits[name].argmax(dim=-1) == final_preds
        not_yet = commit == len(hook_names) - 1
        commit[matches & not_yet] = li
    return commit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device=device)
    model.eval()
    n_layers  = model.cfg["n_layers"]
    n_classes = model.cfg["n_classes"]

    hook_names   = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    # ── load data ──────────────────────────────────────────────────────────
    data_dir = Path(args.data)
    print(f"\nLoading splits...")
    train_data = load_split(data_dir / "train.npz")
    val_data   = load_split(data_dir / "val.npz")

    train_states = train_data["states"]
    val_states   = val_data["states"]
    train_labels = torch.from_numpy(train_data["optimal_distance"].astype(np.int64))
    val_labels   = torch.from_numpy(val_data["optimal_distance"].astype(np.int64))
    val_opt_dist = val_labels.clone()
    print(f"  train: {len(train_states):,}  val: {len(val_states):,}")

    # ── extract activations ────────────────────────────────────────────────
    print("\nExtracting train activations...")
    train_acts = extract_activations(model, train_states, hook_names, device=device)
    print("Extracting val activations...")
    val_acts   = extract_activations(model, val_states,   hook_names, device=device)

    # ── logit lens ────────────────────────────────────────────────────────
    print("\nComputing logit lens...")
    ll_logits_val = logit_lens(model, val_acts, hook_names, device)

    ll_acc  = layer_accuracy(ll_logits_val, val_labels)
    ll_conf = layer_confidence(ll_logits_val)

    # Final model predictions (for commitment analysis)
    with torch.no_grad():
        final_logits_list = []
        for start in range(0, len(val_states), 512):
            b = torch.from_numpy(val_states[start : start + 512]).to(device)
            final_logits_list.append(model(b).cpu().float())
    final_logits = torch.cat(final_logits_list)
    final_preds  = final_logits.argmax(dim=-1)

    # ── tuned lens ────────────────────────────────────────────────────────
    print("\nTraining tuned lens (one layer per hook)...")
    lens_layers = train_tuned_lens(
        model, train_acts, hook_names, train_labels, device,
        epochs=args.lens_epochs, lr=args.lens_lr,
    )

    print("\nApplying tuned lens to val set...")
    tl_logits_val = apply_tuned_lens(model, val_acts, lens_layers, hook_names, device)

    tl_acc  = layer_accuracy(tl_logits_val, val_labels)
    tl_conf = layer_confidence(tl_logits_val)

    # ── commitment analysis ────────────────────────────────────────────────
    commit = commitment_layer(ll_logits_val, final_preds, hook_names)
    commit_frac = [(commit == li).float().mean().item() for li in range(len(hook_names))]

    # ── accuracy by distance ───────────────────────────────────────────────
    ll_by_dist = accuracy_by_distance(ll_logits_val, val_labels, val_opt_dist)
    tl_by_dist = accuracy_by_distance(tl_logits_val, val_labels, val_opt_dist)
    final_acc  = float((final_preds == val_labels).float().mean())

    # ── print results ──────────────────────────────────────────────────────
    print(f"\n── Logit lens vs Tuned lens — val accuracy ────────────────────────")
    print(f"  {'layer':>10}  {'logit_lens':>10}  {'tuned_lens':>10}  {'ll_conf':>8}  {'tl_conf':>8}")
    for name, lbl in zip(hook_names, layer_labels):
        print(
            f"  {lbl:>10}  {ll_acc[name]*100:9.1f}%  {tl_acc[name]*100:9.1f}%"
            f"  {ll_conf[name]*100:7.1f}%  {tl_conf[name]*100:7.1f}%"
        )
    print(f"  {'final':>10}  {final_acc*100:9.1f}%  {'—':>10}")

    print(f"\n── Commitment layer (first layer logit-lens matches final pred) ────")
    for li, (lbl, frac) in enumerate(zip(layer_labels, commit_frac)):
        bar = "█" * round(frac * 40)
        print(f"  {lbl:>8}: {frac*100:5.1f}%  {bar}")

    print(f"\n── Logit lens accuracy by true distance (val) ─────────────────────")
    distances = sorted(ll_by_dist[hook_names[0]].keys())
    header = f"  {'dist':>4}" + "".join(f"  {lbl:>7}" for lbl in layer_labels) + "   final"
    print(header)
    for d in distances:
        row = f"  {d:4d}"
        for name in hook_names:
            acc = ll_by_dist[name].get(d, float("nan"))
            row += f"  {acc*100:6.1f}%"
        row += f"  {(final_preds[val_opt_dist == d] == val_labels[val_opt_dist == d]).float().mean()*100:6.1f}%"
        print(row)

    # ── plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Accuracy by layer",
                    "Confidence by layer",
                    "Commitment layer distribution",
                    "Logit lens accuracy by distance × layer",
                ],
            )

            # Accuracy
            fig.add_trace(go.Scatter(x=layer_labels,
                y=[ll_acc[n] * 100 for n in hook_names],
                name="logit lens", mode="lines+markers"), row=1, col=1)
            fig.add_trace(go.Scatter(x=layer_labels,
                y=[tl_acc[n] * 100 for n in hook_names],
                name="tuned lens", mode="lines+markers"), row=1, col=1)
            fig.add_hline(y=final_acc * 100, line_dash="dash",
                          annotation_text="final model", row=1, col=1)

            # Confidence
            fig.add_trace(go.Scatter(x=layer_labels,
                y=[ll_conf[n] * 100 for n in hook_names],
                name="logit lens", mode="lines+markers", showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=layer_labels,
                y=[tl_conf[n] * 100 for n in hook_names],
                name="tuned lens", mode="lines+markers", showlegend=False), row=1, col=2)

            # Commitment
            fig.add_trace(go.Bar(x=layer_labels, y=[f * 100 for f in commit_frac],
                                 showlegend=False), row=2, col=1)

            # Accuracy by distance heatmap
            z = [[ll_by_dist[n].get(d, float("nan")) * 100 for n in hook_names]
                 for d in distances]
            fig.add_trace(go.Heatmap(
                z=z, x=layer_labels, y=[str(d) for d in distances],
                colorscale="RdYlGn", zmin=0, zmax=100,
                colorbar=dict(title="%", len=0.45, y=0.1),
                showscale=True,
            ), row=2, col=2)

            fig.update_yaxes(title_text="accuracy (%)", row=1, col=1)
            fig.update_yaxes(title_text="max softmax (%)", row=1, col=2)
            fig.update_yaxes(title_text="fraction of samples (%)", row=2, col=1)
            fig.update_yaxes(title_text="true distance", row=2, col=2)
            fig.update_layout(
                title="Phase 5 (Option 1): Logit Lens + Tuned Lens",
                height=850,
            )

            out = Path("tuned_lens_results.html")
            fig.write_html(str(out))
            print(f"\nPlot saved → {out}")
        except ImportError:
            print("\nplotly not installed; skipping plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5 Option 1: Logit lens + Tuned lens.")
    p.add_argument("--checkpoint",   default="checkpoints/best.pt")
    p.add_argument("--data",         default="data")
    p.add_argument("--lens-epochs",  type=int,   default=20,  dest="lens_epochs")
    p.add_argument("--lens-lr",      type=float, default=1e-3, dest="lens_lr")
    p.add_argument("--no-plot",      action="store_true", dest="no_plot")
    run(p.parse_args())


if __name__ == "__main__":
    main()
