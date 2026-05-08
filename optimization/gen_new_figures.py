"""Generate two new figures for the optimization report:
  1. training_dynamics.pdf  — per-epoch val_acc (mean ± std) for all 3 optimizers
  2. overestimation.pdf     — per-distance admissibility breakdown (stacked bar)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"

COLORS   = {"adamw": "#1f77b4", "adam": "#ff7f0e", "sgd": "#2ca02c"}
LABELS   = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD + momentum"}
LINESTYLE = {"adamw": "-", "adam": "--", "sgd": ":"}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
})

# ── Figure 1: Training dynamics ───────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))

epochs = np.arange(1, 31)
for opt in ["adamw", "adam", "sgd"]:
    accs = []
    for s in range(3):
        hist = json.load(open(RESULTS / f"multi_seed_{opt}_s{s}.json"))
        accs.append([h["val_acc"] * 100 for h in hist])
    accs = np.array(accs)           # shape (3, 30)
    mean = accs.mean(axis=0)
    std  = accs.std(axis=0)
    c = COLORS[opt]
    ax.plot(epochs, mean, color=c, label=LABELS[opt],
            linestyle=LINESTYLE[opt], linewidth=1.8)
    ax.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.15)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy (%)")
ax.set_xlim(1, 30)
ax.set_ylim(50, 83)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.legend(loc="lower right")
ax.grid(True, linewidth=0.4, alpha=0.5)
ax.set_title("Per-epoch validation accuracy (mean ± std, 3 seeds)")

# annotate: no sudden phase transition
ax.annotate("smooth, monotone\nconvergence — no phase transition",
            xy=(20, 76), xytext=(14, 60),
            fontsize=7.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))

fig.tight_layout()
out = FIGURES / "training_dynamics.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
plt.close(fig)

# ── Figure 2: Per-distance admissibility breakdown ────────────────────────────
hist = json.load(open(RESULTS / "opt_adamw.json"))
depth_errors = hist[-1]["depth_errors"]   # dict: str(dist) -> list of (pred-true)

distances  = list(range(11))
n_over     = []
n_correct  = []
n_under    = []
totals     = []

for d in distances:
    errs = np.array(depth_errors.get(str(d), []))
    n = len(errs)
    totals.append(n)
    if n == 0:
        n_over.append(0); n_correct.append(0); n_under.append(0)
    else:
        n_over.append(int((errs > 0).sum()))
        n_correct.append(int((errs == 0).sum()))
        n_under.append(int((errs < 0).sum()))

# Convert to percentages
pct_over    = [100 * o / t if t else 0 for o, t in zip(n_over,    totals)]
pct_correct = [100 * c / t if t else 0 for c, t in zip(n_correct, totals)]
pct_under   = [100 * u / t if t else 0 for u, t in zip(n_under,   totals)]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

# Left: stacked bar (admissibility breakdown)
ax = axes[0]
x = np.array(distances)
b1 = ax.bar(x, pct_correct, color="#4ade80", label="Correct")
b2 = ax.bar(x, pct_over,    bottom=pct_correct, color="#f87171", label="Overestimate (inadmissible)")
b3 = ax.bar(x, pct_under,   bottom=[c + o for c, o in zip(pct_correct, pct_over)],
            color="#93c5fd", label="Underestimate (admissible)")

ax.set_xlabel("True distance $d^*$")
ax.set_ylabel("Fraction of validation examples (%)")
ax.set_xticks(distances)
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=7.5)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)
ax.set_title("Admissibility breakdown by distance class")

# annotate inadmissible zone
ax.annotate("inadmissible\nzone (d=4–7)",
            xy=(5, pct_over[5] / 2 + pct_correct[5]),
            xytext=(7.2, 70),
            fontsize=7.5, color="#b91c1c",
            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9))

# Right: overestimation rate line plot
ax2 = axes[1]
ax2.bar(x, pct_over, color="#f87171", alpha=0.85, label="Overestimation rate")
overall_rate = 100 * sum(n_over) / sum(totals)
ax2.axhline(overall_rate, color="#991b1b", linestyle="--", linewidth=1.2,
            label=f"Overall rate ({overall_rate:.1f}%)")
ax2.set_xlabel("True distance $d^*$")
ax2.set_ylabel("Overestimation rate (%)")
ax2.set_xticks(distances)
ax2.set_ylim(0, 65)
ax2.legend(fontsize=7.5)
ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
ax2.set_title("Overestimation rate by distance (inadmissibility profile)")

# annotate d=9-10 admissible
ax2.annotate("admissible\n(d=8–10)", xy=(9, 2), xytext=(7.5, 20),
             fontsize=7.5, color="#166534",
             arrowprops=dict(arrowstyle="->", color="#166534", lw=0.9))

fig.tight_layout()
out = FIGURES / "overestimation.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
plt.close(fig)
