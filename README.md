# Rubik's Cube Mechanistic Interpretability

A mechanistic interpretability project using the 2×2×2 Rubik's cube as a controlled setting. We train a small transformer on optimal-distance classification, then probe and patch its residual stream to understand what it learns and how. A parallel line of work evaluates text representations of cube state for pre-trained LLM reasoning.

**Progress report:** [`docs/progress_report.pdf`](docs/progress_report.pdf) (source: [`docs/progress_report.tex`](docs/progress_report.tex)).

## What this project does

1. **Simulates** a 2×2×2 cube with a one-hot state encoding (24 stickers × 6 colors = 144-dim vector)
2. **Solves** any cube state optimally via BFS distance-table lookup, and generates scramble/solution pairs
3. **Generates** a dataset of scrambled cube states with BFS-computed optimal distances (God's number = 11)
4. **Trains** a small TransformerLens transformer to classify optimal distance (0–11) from state
5. **Probes** each residual stream layer with logistic regression to find linearly decodable features
6. **Patches** activations causally to distinguish features the model *uses* from those it merely encodes
7. **Tuned lens** — trains a per-layer affine transform to read off predictions at each intermediate layer, revealing how the model builds up its answer
8. **Sparse autoencoder** — trains an overcomplete SAE on each layer's residual stream to find monosemantic features and check alignment with known concepts
9. **Text representations** — evaluates five ways to describe cube state in natural language and tests whether a pre-trained LLM can solve scrambles from each

## Key findings

**Phase 4 — Linear probing:**
- `face_solved` is 98% decodable from every layer, including the raw embedding
- `corner_oriented` degrades from 100% at the embedding to ~92% at the final layer — the model transforms it away as it focuses on distance
- Optimal-distance MAE drops from 0.87 at the embedding to 0.40 at L0, with minimal improvement in later layers

**Phase 5 — Activation patching:**
- Ablating the `face_solved` and `corner_oriented` probe directions causes **zero** prediction change — these features are epiphenomenal, not causally active
- Swapping the full residual stream between distance groups produces **100% flip rate at every layer**, including before any transformer block runs
- Conclusion: the linear embedding layer encodes optimal distance almost completely; the transformer blocks refine hard cases but don't restructure the distance representation

**Phase 5 — Tuned lens:**
- Logit lens accuracy grows from 20% at the embedding to 78% at L3, meaning each block is doing meaningful work (unlike the patching experiment suggested)
- The gap between logit lens (~20–35%) and tuned lens (~60–80%) at early layers shows the residual stream holds the right information in a rotated basis the final head can't yet read — the tuned lens learns that rotation
- The gap closes by L3 (78.5% vs 81.3%), confirming the final block aligns the representation to the head's reading direction
- Distance 1 and 2 show a near-zero → ~100% accuracy jump at L2, a striking phase transition where the representation snaps into place
- Distances 5–9 remain hard at every layer, consistent with the class-imbalance in training data

The model is effectively a linear classifier over its own embedded state, but the transformer blocks matter for hard inputs.

**Phase 5 — Sparse autoencoder (SAE, 4× expansion, 512 features):**
- Face and distance features are strongly monosemantic: dedicated SAE features correlate with `face_solved` (r ≈ 0.77–0.87) and `optimal_distance` (r ≈ 0.81–0.89) at every layer
- Symmetric face pairs (U/D, F/B, L/R) consistently share the same top SAE features — the model treats geometrically equivalent face pairs identically
- Corner orientation features are weaker (r ≈ 0.5–0.63) and strongest at the embedding layer, again confirming that corner orientation is encoded early and then transformed away
- The final layer (L3) is naturally sparser (mean L0 ≈ 200 vs ~400 in earlier layers), consistent with the model compressing to a decision

**Phase 6 — Text representations:**

Five representations tested on GPT (chat interface), 1 trial each at optimal distances d=3, 5, 7:

| Representation | d=3 | d=5 | d=7 |
|---|---|---|---|
| `face_grid` — unfolded net diagram | ✓ | ✗ | ✗ |
| `compact_string` — 24-char flat string | ✓ | ✗ | ✓ |
| `corner_cubies` — colors per face per position | ✓ | ✗ | ✗ |
| **`piece_identity`** — piece name + current position + W/Y face | ✓ | **✓** | ✗ |
| `move_sequence` — the scramble itself (degenerate baseline) | ✓ | ✓ | ✓ |
| `piece_identity` + chain-of-thought | ✓ | ✗ | ✗ |

`piece_identity` is the only state-based representation to solve d=5, by making the permutation explicit (naming each piece by its solved-state colors and stating where it is now). d=7 is a hard wall across all representations: the model cannot simulate multi-move sequences reliably from text alone, even with chain-of-thought prompting.

## File map

```
cube-interpretability/
│
├── cube.py                 Core cube simulator
│   ├── Cube                State machine: apply_move, scramble, encode/decode
│   ├── solve()             Optimal BFS-table solver → list of move indices
│   ├── generate_scramble_solution_pairs()
│   ├── compute_optimal_distances()   Full BFS (~25–30 min, cached as data/distances.npz)
│   └── generate_dataset()  Scramble sequences with labels
│
├── cube_visualizer.py      Interactive tkinter visualizer
│   ├── 2D net view         Flat unfolded cube with per-sticker color labels
│   ├── 3D view             Drag-to-rotate orthographic projection (no extra deps)
│   └── Solver panel        Load table, Step / ▶ Play through optimal solution
│
├── dataset.py              Generate and save train/val/test splits (.npz)
├── model.py                CubeTransformer — TransformerLens HookedRootModule
│                           d_model=128, n_layers=4, n_heads=4 (~200k params)
├── train.py                AdamW + cosine annealing, class-weighted CE
│
├── analysis/               Post-training interpretability passes (run with python -m analysis.<name>)
│   ├── probe.py            Phase 4 — linear probes on residual stream
│   │                       LogisticRegression (face_solved, corner_oriented)
│   │                       Ridge regression (optimal_distance, scramble_depth)
│   ├── patch.py            Phase 5a — activation patching
│   │                       Concept-direction ablation + counterfactual swap
│   ├── tuned_lens.py       Phase 5b — logit lens & trained per-layer affine lens
│   └── sae.py              Phase 5c — sparse autoencoder (4× expansion, 512 features)
│                           Dead-feature resampling, Pearson alignment analysis
│
├── print_test_cases.py     Phase 6 — generates text representations of scrambled
│                           states for manual LLM testing; outputs 5 formats
│                           (face_grid, compact_string, corner_cubies,
│                           piece_identity, move_sequence) + CoT variant
│
├── walkthrough.ipynb       Executed notebook: all phases with inline plots
│
├── testing_transcripts/    LLM responses from manual text-representation tests
│   ├── No_CoT.txt          GPT responses: all 5 representations × 3 distances
│   └── CoT.txt             GPT responses: piece_identity + CoT × 3 distances
│
├── docs/                   Write-up and bibliography
│   ├── progress_report.tex Full progress report (LaTeX)
│   ├── progress_report.pdf Compiled PDF (11 pages)
│   ├── references.bib      Bibliography
│   └── proposal.tex        Original project proposal
│
├── data/                   (gitignored — regenerate with uv run cube-dataset)
│   ├── distances.npz       BFS table: sorted uint64 packed keys + int8 distances (mmap-loaded)
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
│
├── checkpoints/            (gitignored — regenerate with uv run cube-train)
│   └── best.pt
│
└── pyproject.toml          uv project; scripts: cube-tests, cube-visualizer,
                            cube-dataset, cube-train
```

## Solver API

`cube.py` exposes two functions for optimal solving (requires the BFS distance table):

```python
import pickle
from cube import Cube, MOVE_NAMES, solve, generate_scramble_solution_pairs

with open("data/distances_cache.pkl", "rb") as f:
    distances = pickle.load(f)

# Solve a single cube state
cube = Cube()
cube.scramble(9)
solution = solve(cube.state, distances)
print([MOVE_NAMES[m] for m in solution])   # e.g. ['R', "U'", 'F2', 'D', 'B2', 'L']

# Generate (scramble, solution) pairs in bulk
pairs = generate_scramble_solution_pairs(1000, distances, max_scramble=11)
for scramble_moves, solution_moves in pairs:
    ...
```

`solve()` uses greedy descent on the BFS distance table — because the table stores true shortest-path distances, the greedy choice is always optimal (HTM-optimal, ≤ 11 moves).

The visualizer (`uv run cube-visualizer`) includes a **Solver** panel: click **Solve** to compute the optimal solution for the current cube state, then step through it move-by-move or auto-play at a configurable speed. The distance table is loaded once in a background thread on first use.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Generate dataset (computes BFS distances on first run, ~25–30 min; cached afterward)
uv run cube-dataset

# Train the transformer (~30 epochs, ~5 min on MPS/GPU)
uv run cube-train

# Phase 4: linear probing
uv run python -m analysis.probe

# Phase 5a: activation patching
uv run python -m analysis.patch

# Phase 5b: logit lens + tuned lens
uv run python -m analysis.tuned_lens

# Phase 5c: sparse autoencoder
uv run python -m analysis.sae

# Phase 6: generate text representation test cases for manual LLM evaluation
uv run python print_test_cases.py
```

Output files: `data/` (splits + BFS cache), `checkpoints/best.pt`, and HTML visualizations for each analysis step.

## Model architecture

```
Input:   (batch, 144)  float32 one-hot cube state
Embed:   Linear(144 → d_model)          hook_embed
Blocks:  TransformerBlock × n_layers
           LN → Attention → residual    hook_resid_mid
           LN → MLP      → residual     hook_resid_post
Head:    LayerNorm → Linear(d_model, 12)
```

Default config: `d_model=128, n_layers=4, n_heads=4` (~200k parameters).

Note: input is a single token, so attention weights are always 1.0. Computation flows almost entirely through the MLP sublayers. All hook points follow TransformerLens naming, so `model.run_with_cache()` works out of the box.

**Training result:** 78.5% val accuracy on 12-class distance classification (random baseline: 8.3%).

## Dataset

Each `.npz` split contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `states` | `(N, 144)` float32 | One-hot encoded cube state |
| `optimal_distance` | `(N,)` int8 | BFS distance to solved (0–10) |
| `next_moves` | `(N,)` int64 | Move applied to reach this state |
| `scramble_depth` | `(N,)` int32 | Steps from solved in the scramble sequence |
| `face_solved` | `(N, 6)` bool | Per-face solved status |
| `corner_oriented` | `(N, 8)` bool | Per-corner U/D orientation |

Default split sizes: 50k / 5k / 5k sequences → ~300k / 30k / 30k samples.

The BFS reaches the full 88M-state space from a single solved seed (face moves connect every rotationally-equivalent solved state). The one-time BFS takes ~25–30 min; the result is packed into a sorted `(uint64, int8)` array pair and written to `data/distances.npz` (~760 MB). Subsequent loads are memory-mapped and effectively instant (~0.1 s), and lookup is `O(log N)` via `np.searchsorted`.

## Move vocabulary

18 moves = 6 faces × 3 turn types (CW, CCW, 180°):

```
U U' U2 | D D' D2 | F F' F2 | B B' B2 | L L' L2 | R R' R2
```

## References

- Alain & Bengio (2016) — [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644)
- Belrose et al. (2023) — [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)
- Meng et al. (2022) — [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (activation patching methodology)
- Nanda et al. (2023) — [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)
- Bricken et al. (2023) — [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
- Variengien et al. (2024) — [Transformers Represent Belief State Geometry in their Residual Stream](https://arxiv.org/abs/2405.15943)
- Gupta et al. (2024) — [Better World Models Can Lead to Better Post-Training Performance](https://arxiv.org/abs/2512.03400)
- Agostinelli et al. (2019) — [Solving the Rubik's Cube with Deep Reinforcement Learning and Search](https://www.nature.com/articles/s42256-019-0070-z)
- Takano (2023) — [Self-Supervision is All You Need for Solving Rubik's Cube](https://arxiv.org/abs/2106.03157)
