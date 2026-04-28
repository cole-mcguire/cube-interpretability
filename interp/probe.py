"""
Phase 4: Linear probing across transformer layers.

Trains logistic regression / ridge probes on residual stream activations
at each layer to test whether the model encodes structured cube state.

Usage:
    uv run python probe.py                        # defaults
    uv run python probe.py --checkpoint checkpoints/best.pt
    uv run python probe.py --split test           # probe on test set
    uv run python probe.py --no-plot              # skip Plotly output

Probe targets:
    face_solved      (N, 6)  bool   — 6 binary LogisticRegression probes
    corner_oriented  (N, 8)  bool   — 8 binary LogisticRegression probes
    optimal_distance (N,)    int    — Ridge regression (MAE metric)
    scramble_depth   (N,)    int    — Ridge regression (MAE metric)

Activation layers probed:
    hook_embed  (after embedding, before blocks)
    blocks.{i}.hook_resid_post  for i in 0..n_layers-1
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


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model,
    states: np.ndarray,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Return dict of hook_name → (N, d_model) float32 numpy arrays."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    n_layers = model.cfg["n_layers"]
    hook_names = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    accum: dict[str, list[np.ndarray]] = {k: [] for k in hook_names}

    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)
        for name in hook_names:
            accum[name].append(cache[name].cpu().float().numpy())

    return {name: np.concatenate(accum[name], axis=0) for name in hook_names}


# ---------------------------------------------------------------------------
# Probing helpers
# ---------------------------------------------------------------------------

def probe_binary(
    acts: np.ndarray, labels: np.ndarray, train_frac: float = 0.8
) -> float:
    """Train logistic regression, return test accuracy."""
    n = len(labels)
    n_train = int(n * train_frac)
    X_tr, X_te = acts[:n_train], acts[n_train:]
    y_tr, y_te = labels[:n_train], labels[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def probe_regression(
    acts: np.ndarray, labels: np.ndarray, train_frac: float = 0.8
) -> float:
    """Train ridge regression, return test MAE."""
    n = len(labels)
    n_train = int(n * train_frac)
    X_tr, X_te = acts[:n_train], acts[n_train:]
    y_tr, y_te = labels[:n_train].astype(float), labels[n_train:].astype(float)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    return float(np.mean(np.abs(preds - y_te)))


# ---------------------------------------------------------------------------
# Main probing logic
# ---------------------------------------------------------------------------

def run_probes(args: argparse.Namespace) -> None:
    device = torch.device("cpu")  # probing uses small batches; CPU is fine
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    model.eval()
    n_layers = model.cfg["n_layers"]
    hook_names = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    data_path = Path(args.data) / f"{args.split}.npz"
    print(f"Loading {args.split} split from {data_path}...")
    data = load_split(data_path)
    states          = data["states"]
    face_solved     = data["face_solved"]          # (N, 6) bool
    corner_oriented = data["corner_oriented"]      # (N, 8) bool
    opt_dist        = data["optimal_distance"].astype(np.int32)   # (N,)
    scr_depth       = data["scramble_depth"].astype(np.int32)     # (N,)

    print(f"Samples: {len(states):,}  |  Extracting activations for {len(hook_names)} hooks...")
    acts = extract_activations(model, states, batch_size=args.batch_size, device=device)
    print("Done.\n")

    # ------------------------------------------------------------------
    # 1. face_solved probes (6 binary)
    # ------------------------------------------------------------------
    FACE_NAMES = ["U", "D", "F", "B", "L", "R"]
    face_results: dict[str, list[float]] = {}  # face_name → [acc per layer]

    print("── face_solved probes (LogisticRegression) ─────────────────────────")
    header = f"{'layer':>10}" + "".join(f"  {fn:>6}" for fn in FACE_NAMES) + "  mean"
    print(header)
    print("─" * len(header))

    face_layer_acc: list[list[float]] = []  # [layer][face]
    for hook, lbl in zip(hook_names, layer_labels):
        accs = [probe_binary(acts[hook], face_solved[:, fi]) for fi in range(6)]
        face_layer_acc.append(accs)
        face_results[lbl] = accs
        row = f"{lbl:>10}" + "".join(f"  {a*100:5.1f}%" for a in accs) + f"  {np.mean(accs)*100:5.1f}%"
        print(row)

    # ------------------------------------------------------------------
    # 2. corner_oriented probes (8 binary)
    # ------------------------------------------------------------------
    print("\n── corner_oriented probes (LogisticRegression) ─────────────────────")
    corner_header = f"{'layer':>10}" + "".join(f"  C{i:>4}" for i in range(8)) + "  mean"
    print(corner_header)
    print("─" * len(corner_header))

    corner_layer_acc: list[list[float]] = []
    for hook, lbl in zip(hook_names, layer_labels):
        accs = [probe_binary(acts[hook], corner_oriented[:, ci]) for ci in range(8)]
        corner_layer_acc.append(accs)
        row = f"{lbl:>10}" + "".join(f"  {a*100:5.1f}%" for a in accs) + f"  {np.mean(accs)*100:5.1f}%"
        print(row)

    # ------------------------------------------------------------------
    # 3. optimal_distance probe (Ridge)
    # ------------------------------------------------------------------
    print("\n── optimal_distance probe (Ridge regression, MAE) ──────────────────")
    print(f"  Baseline MAE (predict-mean): {np.mean(np.abs(opt_dist.astype(float) - opt_dist.mean())):.3f}")
    opt_maes: list[float] = []
    for hook, lbl in zip(hook_names, layer_labels):
        mae = probe_regression(acts[hook], opt_dist)
        opt_maes.append(mae)
        print(f"  {lbl:>10}:  MAE = {mae:.3f}")

    # ------------------------------------------------------------------
    # 4. scramble_depth probe (Ridge)
    # ------------------------------------------------------------------
    print("\n── scramble_depth probe (Ridge regression, MAE) ────────────────────")
    print(f"  Baseline MAE (predict-mean): {np.mean(np.abs(scr_depth.astype(float) - scr_depth.mean())):.3f}")
    scr_maes: list[float] = []
    for hook, lbl in zip(hook_names, layer_labels):
        mae = probe_regression(acts[hook], scr_depth)
        scr_maes.append(mae)
        print(f"  {lbl:>10}:  MAE = {mae:.3f}")

    # ------------------------------------------------------------------
    # Plotly visualization
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "face_solved accuracy by layer",
                    "corner_oriented accuracy by layer",
                    "optimal_distance MAE by layer",
                    "scramble_depth MAE by layer",
                ],
            )

            for fi, fn in enumerate(FACE_NAMES):
                fig.add_trace(
                    go.Scatter(x=layer_labels, y=[face_layer_acc[li][fi] * 100 for li in range(len(hook_names))],
                               name=f"face {fn}", mode="lines+markers"),
                    row=1, col=1,
                )
            # mean line
            fig.add_trace(
                go.Scatter(x=layer_labels, y=[np.mean(face_layer_acc[li]) * 100 for li in range(len(hook_names))],
                           name="mean", mode="lines+markers", line=dict(dash="dash", width=3, color="black")),
                row=1, col=1,
            )

            for ci in range(8):
                fig.add_trace(
                    go.Scatter(x=layer_labels, y=[corner_layer_acc[li][ci] * 100 for li in range(len(hook_names))],
                               name=f"corner {ci}", mode="lines+markers", showlegend=True),
                    row=1, col=2,
                )
            fig.add_trace(
                go.Scatter(x=layer_labels, y=[np.mean(corner_layer_acc[li]) * 100 for li in range(len(hook_names))],
                           name="mean", mode="lines+markers", line=dict(dash="dash", width=3, color="black"),
                           showlegend=False),
                row=1, col=2,
            )

            fig.add_trace(
                go.Scatter(x=layer_labels, y=opt_maes, mode="lines+markers", showlegend=False),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(x=layer_labels, y=scr_maes, mode="lines+markers", showlegend=False),
                row=2, col=2,
            )

            fig.update_yaxes(title_text="accuracy (%)", row=1, col=1)
            fig.update_yaxes(title_text="accuracy (%)", row=1, col=2)
            fig.update_yaxes(title_text="MAE", row=2, col=1)
            fig.update_yaxes(title_text="MAE", row=2, col=2)
            fig.update_layout(
                title="Phase 4: Linear Probe Results by Layer",
                height=800,
            )

            out_html = Path("probe_results.html")
            fig.write_html(str(out_html))
            print(f"\nPlot saved → {out_html}")
        except ImportError:
            print("\nplotly not installed; skipping plot. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4: Linear probing across transformer layers.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data",       default="data")
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=1024, dest="batch_size")
    p.add_argument("--no-plot",    action="store_true", dest="no_plot")
    run_probes(p.parse_args())


if __name__ == "__main__":
    main()
