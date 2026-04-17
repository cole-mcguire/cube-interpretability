"""
Phase 5 (Option 2): Activation patching — causal test of concept directions.

For each binary concept (face_solved × 6, corner_oriented × 8) and each layer:
  1. Fit a logistic regression probe → extract weight direction w
  2. For positive examples (concept=1), mean-ablate: replace the concept
     component of the activation with the negative-class mean projection
  3. Run model.run_with_hooks() with patched activation at that layer
  4. Report: prediction flip rate, mean distance shift

Core question: at which layer does each concept have the largest causal
influence on the model's optimal_distance prediction?

Usage:
    uv run python patch.py
    uv run python patch.py --checkpoint checkpoints/best.pt --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    return {n: np.concatenate(accum[n]) for n in hook_names}


def fit_probe(
    acts: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.8,
) -> tuple[np.ndarray, StandardScaler, float]:
    """Return (probe_weight_in_original_space, scaler, train_acc)."""
    n_train = int(len(labels) * train_frac)
    X_tr, y_tr = acts[:n_train], labels[:n_train]
    X_te, y_te = acts[n_train:], labels[n_train:]

    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    acc = float(clf.score(scaler.transform(X_te), y_te))

    # Map coef back to original (unscaled) space: w_orig = coef / scale
    w_orig = clf.coef_[0] / scaler.scale_
    return w_orig, scaler, acc


def mean_ablate(
    acts: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    For all samples, replace the concept component of acts with the
    negative-class (label=0) mean projection. Returns patched activations.
    """
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    neg_proj_mean = float((acts[labels == 0] @ w_hat).mean())
    curr_proj = acts @ w_hat                                    # (N,)
    delta = (neg_proj_mean - curr_proj)[:, None] * w_hat[None, :]
    return (acts + delta).astype(np.float32)


@torch.no_grad()
def patched_logits(
    model,
    states: np.ndarray,
    hook_name: str,
    patched_acts: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run model with activation at hook_name replaced by patched_acts."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_logits: list[np.ndarray] = []

    for start in range(0, len(states), batch_size):
        batch_states  = torch.from_numpy(states[start : start + batch_size]).to(device)
        batch_patched = torch.from_numpy(patched_acts[start : start + batch_size]).to(device)

        def hook_fn(act, hook, _p=batch_patched):
            return _p

        logits = model.run_with_hooks(batch_states, fwd_hooks=[(hook_name, hook_fn)])
        all_logits.append(logits.cpu().float().numpy())

    return np.concatenate(all_logits)                           # (N, n_classes)


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_patching(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device=device)
    model.eval()
    n_layers  = model.cfg["n_layers"]
    n_classes = model.cfg["n_classes"]

    hook_names   = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    print(f"Loading {args.split} split from {args.data}/{args.split}.npz...")
    data = load_split(Path(args.data) / f"{args.split}.npz")

    states          = data["states"]                            # (N, 144) float32
    face_solved     = data["face_solved"]                      # (N, 6) bool
    corner_oriented = data["corner_oriented"]                  # (N, 8) bool
    opt_dist        = data["optimal_distance"].astype(np.int32)

    print(f"Samples: {len(states):,}  |  Extracting activations...")
    acts = extract_activations(model, states, hook_names, device=device)

    # Baseline: clean predictions
    with torch.no_grad():
        clean_logits_list = []
        for start in range(0, len(states), 512):
            batch = torch.from_numpy(states[start : start + 512]).to(device)
            clean_logits_list.append(model(batch).cpu().float().numpy())
    clean_logits = np.concatenate(clean_logits_list)           # (N, n_classes)
    clean_pred   = clean_logits.argmax(axis=1)                 # (N,)

    FACE_NAMES   = ["U", "D", "F", "B", "L", "R"]
    concepts = (
        [(f"face_{fn}", face_solved[:, fi]) for fi, fn in enumerate(FACE_NAMES)]
        + [(f"corner_{ci}", corner_oriented[:, ci]) for ci in range(8)]
    )

    # Results: concept → layer → {flip_rate, dist_shift, probe_acc}
    results: dict[str, list[dict]] = {}

    for concept_name, labels in concepts:
        labels = labels.astype(np.int32)
        pos_mask = labels == 1
        n_pos = pos_mask.sum()
        if n_pos < 20:
            print(f"  {concept_name}: skipped (only {n_pos} positive examples)")
            continue

        print(f"\n── {concept_name}  (n_pos={n_pos:,}) ──────────────────────────────────")
        print(f"  {'layer':>10}  {'probe_acc':>9}  {'flip_rate':>9}  {'dist_shift':>10}  {'mean_dist_clean':>15}  {'mean_dist_patch':>15}")

        layer_results: list[dict] = []

        for hook, lbl in zip(hook_names, layer_labels):
            layer_acts = acts[hook]                             # (N, d_model)
            w, scaler, probe_acc = fit_probe(layer_acts, labels)

            # Ablate all positive examples
            patched = mean_ablate(layer_acts, labels, w)        # (N, d_model)

            # Only evaluate on positive examples (we patched their concept)
            pos_states  = states[pos_mask]
            pos_patched = patched[pos_mask]
            pos_clean_pred = clean_pred[pos_mask]
            pos_opt_dist   = opt_dist[pos_mask].astype(float)

            p_logits = patched_logits(model, pos_states, hook, pos_patched, device=device)
            p_pred   = p_logits.argmax(axis=1)

            flip_rate  = float((p_pred != pos_clean_pred).mean())
            dist_shift = float(p_pred.mean() - pos_clean_pred.mean())
            mean_clean = float(pos_clean_pred.mean())
            mean_patch = float(p_pred.mean())

            print(
                f"  {lbl:>10}  {probe_acc*100:8.1f}%  {flip_rate*100:8.1f}%"
                f"  {dist_shift:+10.3f}  {mean_clean:15.3f}  {mean_patch:15.3f}"
            )
            layer_results.append({
                "layer": lbl, "probe_acc": probe_acc,
                "flip_rate": flip_rate, "dist_shift": dist_shift,
                "mean_clean": mean_clean, "mean_patch": mean_patch,
            })

        results[concept_name] = layer_results

    # ------------------------------------------------------------------
    # Summary: most causally influential layer per concept
    # ------------------------------------------------------------------
    print("\n\n── Causal summary (layer with highest flip_rate per concept) ────────")
    print(f"  {'concept':>20}  {'best_layer':>10}  {'flip_rate':>9}  {'dist_shift':>10}")
    for concept_name, layer_results in results.items():
        best = max(layer_results, key=lambda r: r["flip_rate"])
        print(
            f"  {concept_name:>20}  {best['layer']:>10}"
            f"  {best['flip_rate']*100:8.1f}%  {best['dist_shift']:+10.3f}"
        )

    # ------------------------------------------------------------------
    # Experiment 2: Distance counterfactual patching
    #
    # For pairs of distance groups (src_d → tgt_d), swap the full residual
    # stream activation at each layer and measure how often the model's
    # prediction shifts toward the target group. This tests *where* distance
    # information is causally encoded — not just linearly decodable.
    # ------------------------------------------------------------------
    print("\n\n── Distance counterfactual patching ─────────────────────────────────")
    print("  Replacing activation of high-distance states with mean of low-distance")
    print("  states at each layer. Metric: fraction whose predicted distance drops.\n")

    PATCH_PAIRS = [(5, 1), (4, 1), (3, 1)]  # (src_group, tgt_group)

    cf_results: dict[tuple[int, int], list[dict]] = {}

    for src_d, tgt_d in PATCH_PAIRS:
        src_mask = opt_dist == src_d
        tgt_mask = opt_dist == tgt_d
        n_src, n_tgt = src_mask.sum(), tgt_mask.sum()
        if n_src < 10 or n_tgt < 10:
            print(f"  d={src_d}→{tgt_d}: skipped (n_src={n_src}, n_tgt={n_tgt})")
            continue

        src_states = states[src_mask]
        src_clean_pred = clean_pred[src_mask]

        print(f"  d={src_d}→{tgt_d}  (n_src={n_src:,}, n_tgt={n_tgt:,})")
        print(f"  {'layer':>10}  {'flip_rate':>9}  {'pred_before':>11}  {'pred_after':>10}  {'Δ pred':>8}")

        pair_results: list[dict] = []

        for hook, lbl in zip(hook_names, layer_labels):
            # Mean activation of target group at this layer
            tgt_mean = acts[hook][tgt_mask].mean(axis=0, keepdims=True)  # (1, d_model)
            src_patched = np.broadcast_to(tgt_mean, acts[hook][src_mask].shape).copy().astype(np.float32)

            p_logits = patched_logits(model, src_states, hook, src_patched, device=device)
            p_pred   = p_logits.argmax(axis=1)

            flip_rate   = float((p_pred != src_clean_pred).mean())
            pred_before = float(src_clean_pred.mean())
            pred_after  = float(p_pred.mean())
            delta       = pred_after - pred_before

            print(f"  {lbl:>10}  {flip_rate*100:8.1f}%  {pred_before:11.3f}  {pred_after:10.3f}  {delta:+8.3f}")
            pair_results.append({
                "layer": lbl, "flip_rate": flip_rate,
                "pred_before": pred_before, "pred_after": pred_after, "delta": delta,
            })

        cf_results[(src_d, tgt_d)] = pair_results
        print()

    # ------------------------------------------------------------------
    # Plotly visualization
    # ------------------------------------------------------------------
    if not args.no_plot:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            face_concepts   = [c for c in results if c.startswith("face_")]
            corner_concepts = [c for c in results if c.startswith("corner_")]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "face_solved — flip rate by layer",
                    "corner_oriented — flip rate by layer",
                    "face_solved — distance shift by layer",
                    "corner_oriented — distance shift by layer",
                ],
            )

            for cn in face_concepts:
                xs = [r["layer"] for r in results[cn]]
                fig.add_trace(
                    go.Scatter(x=xs, y=[r["flip_rate"] * 100 for r in results[cn]],
                               name=cn, mode="lines+markers"),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=xs, y=[r["dist_shift"] for r in results[cn]],
                               name=cn, mode="lines+markers", showlegend=False),
                    row=2, col=1,
                )

            for cn in corner_concepts:
                xs = [r["layer"] for r in results[cn]]
                fig.add_trace(
                    go.Scatter(x=xs, y=[r["flip_rate"] * 100 for r in results[cn]],
                               name=cn, mode="lines+markers"),
                    row=1, col=2,
                )
                fig.add_trace(
                    go.Scatter(x=xs, y=[r["dist_shift"] for r in results[cn]],
                               name=cn, mode="lines+markers", showlegend=False),
                    row=2, col=2,
                )

            fig.update_yaxes(title_text="flip rate (%)", row=1, col=1)
            fig.update_yaxes(title_text="flip rate (%)", row=1, col=2)
            fig.update_yaxes(title_text="dist shift", row=2, col=1)
            fig.update_yaxes(title_text="dist shift", row=2, col=2)

            # --- second figure: counterfactual patching ---
            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "Distance counterfactual — flip rate by layer",
                    "Distance counterfactual — Δ predicted distance",
                ],
            )
            for (src_d, tgt_d), pr in cf_results.items():
                xs    = [r["layer"] for r in pr]
                label = f"d={src_d}→{tgt_d}"
                fig2.add_trace(
                    go.Scatter(x=xs, y=[r["flip_rate"] * 100 for r in pr],
                               name=label, mode="lines+markers"),
                    row=1, col=1,
                )
                fig2.add_trace(
                    go.Scatter(x=xs, y=[r["delta"] for r in pr],
                               name=label, mode="lines+markers", showlegend=False),
                    row=1, col=2,
                )
            fig2.update_yaxes(title_text="flip rate (%)", row=1, col=1)
            fig2.update_yaxes(title_text="Δ predicted distance", row=1, col=2)
            fig2.update_layout(
                title="Phase 5: Distance Counterfactual Patching",
                height=450,
            )

            fig.update_layout(
                title="Phase 5: Concept-Direction Patching (Null Result)",
                height=800,
            )

            out = Path("patch_results.html")
            fig.write_html(str(out))
            out2 = Path("patch_cf_results.html")
            fig2.write_html(str(out2))
            print(f"\nPlots saved → {out}  |  {out2}")
        except ImportError:
            print("\nplotly not installed; skipping plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5: Activation patching.")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--data",       default="data")
    p.add_argument("--split",      default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=512, dest="batch_size")
    p.add_argument("--no-plot",    action="store_true", dest="no_plot")
    run_patching(p.parse_args())


if __name__ == "__main__":
    main()
