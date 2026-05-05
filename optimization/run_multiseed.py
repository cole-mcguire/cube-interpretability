"""
Multi-seed repeat runs for optimizer and schedule comparisons.
Saves results to results/multi_seed_{name}_s{seed}.json.
Prints a mean ± std summary table when done.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_split
from model import CubeTransformer

NOTEBOOK_DIR = ROOT / "optimization"
RESULTS_DIR  = NOTEBOOK_DIR / "results"

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Device: {device}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
train_data = load_split(DATA_DIR / "train.npz")
val_data   = load_split(DATA_DIR / "val.npz")
BATCH_SIZE = 512

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(train_data["states"].astype("float32")),
                  torch.from_numpy(train_data["optimal_distance"].astype("int64"))),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(val_data["states"].astype("float32")),
                  torch.from_numpy(val_data["optimal_distance"].astype("int64"))),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

train_labels = torch.from_numpy(train_data["optimal_distance"].astype("int64"))
counts = torch.bincount(train_labels, minlength=12).float()
class_weights = (1.0 / (counts + 1))
class_weights = class_weights / class_weights.sum() * 12

# ── Training helper ───────────────────────────────────────────────────────────
def run_one(name, seed, optimizer_type, scheduler_type, lr, wd=1e-4, epochs=30):
    out = RESULTS_DIR / f"multi_seed_{name}_s{seed}.json"
    if out.exists():
        print(f"  [cached] {name} seed={seed}")
        return json.loads(out.read_text())

    print(f"  [run]    {name} seed={seed}  ({optimizer_type}, {scheduler_type}, lr={lr:.0e})")
    torch.manual_seed(seed)

    model = CubeTransformer(d_model=128, n_layers=4, n_heads=4, n_classes=12).to(device)

    if optimizer_type == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:  # sgd
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    if scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/20)
    elif scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)

    cw = class_weights.to(device)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for states, targets in train_loader:
            states, targets = states.to(device), targets.to(device)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(states), targets, weight=cw).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for states, targets in val_loader:
                states, targets = states.to(device), targets.to(device)
                correct += (model(states).argmax(1) == targets).sum().item()
                total   += len(targets)
        val_acc = correct / total
        history.append({"epoch": epoch, "val_acc": val_acc})

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}  val_acc={val_acc*100:.1f}%")

    out.write_text(json.dumps(history))
    return history

# ── Configurations ────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2]

optimizer_configs = {
    "adamw": dict(optimizer_type="adamw", scheduler_type="cosine", lr=1e-3),
    "adam":  dict(optimizer_type="adam",  scheduler_type="cosine", lr=1e-3),
    "sgd":   dict(optimizer_type="sgd",   scheduler_type="cosine", lr=0.05),
}

schedule_configs = {
    "cosine":   dict(optimizer_type="adamw", scheduler_type="cosine",   lr=1e-3),
    "step":     dict(optimizer_type="adamw", scheduler_type="step",     lr=1e-3),
    "constant": dict(optimizer_type="adamw", scheduler_type="constant", lr=1e-3),
}

# ── Run ───────────────────────────────────────────────────────────────────────
print("=== Optimizer comparison (3 seeds) ===")
opt_results = {}
for name, cfg in optimizer_configs.items():
    opt_results[name] = []
    for seed in SEEDS:
        hist = run_one(name, seed, **cfg)
        opt_results[name].append(hist[-1]["val_acc"] * 100)

print("\n=== Schedule ablation (3 seeds) ===")
sched_results = {}
for name, cfg in schedule_configs.items():
    sched_results[name] = []
    for seed in SEEDS:
        hist = run_one(name, seed, **cfg)
        sched_results[name].append(hist[-1]["val_acc"] * 100)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n=== SUMMARY TABLE ===")
print(f"{'Configuration':<22} {'Seed 0':>8} {'Seed 1':>8} {'Seed 2':>8} {'Mean':>8} {'Std':>6}")
print("-" * 66)

print("Optimizer comparison:")
labels = {"adamw": "AdamW + cosine", "adam": "Adam + cosine", "sgd": "SGD + cosine"}
for name, accs in opt_results.items():
    m, s = np.mean(accs), np.std(accs)
    print(f"  {labels[name]:<20} {accs[0]:>7.1f}% {accs[1]:>7.1f}% {accs[2]:>7.1f}% {m:>7.1f}% {s:>5.2f}pp")

print("Schedule ablation (AdamW):")
slabels = {"cosine": "Cosine annealing", "step": "Step decay", "constant": "Constant"}
for name, accs in sched_results.items():
    m, s = np.mean(accs), np.std(accs)
    print(f"  {slabels[name]:<20} {accs[0]:>7.1f}% {accs[1]:>7.1f}% {accs[2]:>7.1f}% {m:>7.1f}% {s:>5.2f}pp")

# Save combined summary for reference
summary = {"optimizer": opt_results, "schedule": sched_results}
(RESULTS_DIR / "multi_seed_summary.json").write_text(json.dumps(summary, indent=2))
print("\nSaved multi_seed_summary.json")
