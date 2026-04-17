"""
Training script for CubeTransformer — next-move prediction on 2x2x2 cube states.

Usage:
    uv run cube-train                         # defaults (30 epochs, d_model=128, 4L 4H)
    uv run cube-train --epochs 20 --lr 5e-4
    uv run cube-train --help

Outputs:
    checkpoints/best.pt   — best checkpoint by val accuracy
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_split
from model import CubeTransformer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR   = "data"
DEFAULT_OUT_DIR    = "checkpoints"
DEFAULT_TASK       = "optimal_distance"   # or "next_move"
DEFAULT_D_MODEL    = 128
DEFAULT_N_LAYERS   = 4
DEFAULT_N_HEADS    = 4
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS     = 30
DEFAULT_LR         = 1e-3
DEFAULT_WD         = 1e-4

TASK_CONFIG = {
    "optimal_distance": {"label_key": "optimal_distance", "n_classes": 12},
    "next_move":        {"label_key": "next_moves",       "n_classes": 18},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(data: dict, label_key: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data["states"]),
        torch.from_numpy(data[label_key]).long(),
        torch.from_numpy(data["optimal_distance"].astype("int64")),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(
    model: CubeTransformer,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    total_loss = total_correct = total = 0
    depth_correct: dict[int, int] = defaultdict(int)
    depth_total:   dict[int, int] = defaultdict(int)

    for states, targets, depths in loader:
        states, targets = states.to(device), targets.to(device)
        logits  = model(states)
        loss    = F.cross_entropy(logits, targets, weight=class_weights, reduction="sum")
        correct = (logits.argmax(dim=-1) == targets).cpu()

        total_loss    += loss.item()
        total_correct += correct.sum().item()
        total         += len(targets)

        for d, c in zip(depths.numpy(), correct.numpy()):
            depth_correct[int(d)] += int(c)
            depth_total[int(d)]   += 1

    return (
        total_loss / total,
        total_correct / total,
        {d: depth_correct[d] / depth_total[d] for d in sorted(depth_total)},
    )


def save_checkpoint(
    model: CubeTransformer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    history: list[dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config":      model.cfg,
            "model_state": model.state_dict(),
            "epoch":       epoch,
            "val_loss":    val_loss,
            "val_acc":     val_acc,
            "history":     history,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(args.data)
    if not (data_dir / "train.npz").exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}/. Run `uv run cube-dataset` first."
        )

    task_cfg   = TASK_CONFIG[args.task]
    label_key  = task_cfg["label_key"]
    n_classes  = task_cfg["n_classes"]

    print("Loading dataset...", end=" ", flush=True)
    train_loader = make_loader(load_split(data_dir / "train.npz"), label_key, args.batch_size, shuffle=True)
    val_loader   = make_loader(load_split(data_dir / "val.npz"),   label_key, args.batch_size, shuffle=False)
    print(f"train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")

    model = CubeTransformer(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_classes=n_classes,
    ).to(device)

    print(f"\nTask: {args.task}  ({n_classes} classes)")
    print(f"CubeTransformer — {args.n_layers}L  {args.n_heads}H  d_model={args.d_model}")
    print(f"Parameters: {model.n_params:,}  ({model.n_params * 4 / 1e6:.2f} MB)")
    print(f"Random baseline accuracy: {100/n_classes:.1f}%\n")

    # Class weights: inverse frequency so rare distances are not ignored.
    train_labels = torch.from_numpy(load_split(data_dir / "train.npz")[label_key].astype("int64"))
    counts = torch.bincount(train_labels, minlength=n_classes).float()
    class_weights = (1.0 / (counts + 1)).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes
    print(f"Class weights (min={class_weights.min():.2f}  max={class_weights.max():.2f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 20
    )

    best_val_acc = 0.0
    history: list[dict] = []
    out_path = Path(args.out) / "best.pt"

    header = f"{'Epoch':>6}  {'train':>8}  {'val':>8}  {'acc':>7}  {'lr':>10}  {'time':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = running_loss = 0.0
        t0 = time.perf_counter()

        for states, targets, _ in train_loader:
            states, targets = states.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(states), targets, weight=class_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, depth_acc = evaluate(model, val_loader, device, class_weights)
        elapsed = time.perf_counter() - t0
        lr = scheduler.get_last_lr()[0]

        improved = val_acc > best_val_acc
        print(
            f"{epoch:6d}  {train_loss:8.4f}  {val_loss:8.4f}  "
            f"{val_acc*100:6.1f}%  {lr:10.2e}  {elapsed:5.1f}s"
            + (" *" if improved else "")
        )

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if improved:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, val_loss, val_acc, history, out_path)

    print(f"\nBest val acc: {best_val_acc*100:.1f}%  →  {out_path}")

    label = "optimal distance" if args.task == "optimal_distance" else "scramble depth"
    print(f"\nAccuracy by {label} (val, final epoch):")
    for d, acc in depth_acc.items():
        bar = "█" * round(acc * 30)
        print(f"  {d:2d}: {acc*100:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train CubeTransformer.")
    p.add_argument("--data",       default=DEFAULT_DATA_DIR)
    p.add_argument("--out",        default=DEFAULT_OUT_DIR)
    p.add_argument("--task",       default=DEFAULT_TASK, choices=list(TASK_CONFIG))
    p.add_argument("--d-model",    type=int,   default=DEFAULT_D_MODEL,    dest="d_model")
    p.add_argument("--n-layers",   type=int,   default=DEFAULT_N_LAYERS,   dest="n_layers")
    p.add_argument("--n-heads",    type=int,   default=DEFAULT_N_HEADS,    dest="n_heads")
    p.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE, dest="batch_size")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",         type=float, default=DEFAULT_LR)
    p.add_argument("--wd",         type=float, default=DEFAULT_WD)
    train(p.parse_args())


if __name__ == "__main__":
    main()
