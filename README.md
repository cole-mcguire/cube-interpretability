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
10. **Circuit identification** — DLA, activation patching, and neuron DLA identify which components carry the distance signal and which individual neurons are most distance-tuned
11. **Weight-level analysis** — SVD of the embedding matrix, direct read-out path accuracy, and per-neuron input/output profiles close the loop on the circuit
12. **Training dynamics** — re-trains with per-epoch checkpoints, then tracks per-distance accuracy, layer-wise probe MAE, and component DLA over time to reveal phase transitions
13. **Next-move prediction variant** — re-trains on next-move prediction (18 classes) and probes for optimal distance, testing whether distance representation emerges under a different supervision signal
14. **Superposition analysis** — sweeps SAE expansion ratios (1×–16×) to estimate how many true features the model represents; R² plateau and dead-feature count identify the effective feature count per layer

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
├── interp/                 Post-training interpretability passes (run with python -m interp.<name>)
│   ├── probe.py            Phase 4 — linear probes on residual stream
│   │                       LogisticRegression (face_solved, corner_oriented)
│   │                       Ridge regression (optimal_distance, scramble_depth)
│   ├── patch.py            Phase 5a — activation patching
│   │                       Concept-direction ablation + counterfactual swap
│   ├── tuned_lens.py       Phase 5b — logit lens & trained per-layer affine lens
│   ├── sae.py              Phase 5c — sparse autoencoder (4× expansion, 512 features)
│   │                       Dead-feature resampling, Pearson alignment analysis
│   ├── circuit.py          Phase 7 — circuit identification
│   │                       DLA, activation patching, neuron DLA
│   ├── weights.py          Phase 8 — weight-level analysis
│   │                       Embedding SVD, direct read-out path, top neuron profiles
│   ├── grokking.py         Phase 9 — training dynamics / grokking
│   │                       Per-distance accuracy, layer probe MAE, DLA over epochs
│   ├── next_move.py        Phase 10 — next-move prediction variant
│   │                       Probes next-move model for optimal distance vs distance model
│   └── superposition.py    Phase 11 — SAE expansion sweep
│                           R², active features, dead fraction vs expansion ratio
│
├── print_test_cases.py     Phase 6 — generates text representations of scrambled
│                           states for manual LLM testing; outputs 5 formats
│                           (face_grid, compact_string, corner_cubies,
│                           piece_identity, move_sequence) + CoT variant
│
├── test_representations.py Phase 6 — automated API evaluation of all 5 representations
│                           across multiple LLMs (OpenAI, Gemini, Anthropic, Groq)
│
├── walkthrough.ipynb       Executed notebook: all phases with inline plots
│
├── docs/                   GitHub Pages dashboard + write-up (https://cole-mcguire.github.io/cube-interpretability/)
│   ├── index.html          Interactive visualizer: 2D net + 3D cube, IDA* solver, scramble/solution log
│   ├── cube_core.js        Cube logic + IDA* heuristic tables compiled for the browser
│   ├── solver.worker.js    Web Worker wrapper — runs IDA* off the main thread
│   ├── probe_results.html  Phase 4 — Probing interactive Plotly charts
│   ├── patch_results.html  Phase 5a — Activation patching results
│   ├── patch_cf_results.html  Phase 5a — Counterfactual patching results
│   ├── tuned_lens_results.html  Phase 5b — Tuned Lens results
│   ├── sae_results.html    Phase 5c — SAE results
│   ├── circuit_results.html  Phase 7 — Circuit identification results
│   ├── weights_results.html  Phase 8 — Weight-level analysis results
│   ├── grokking_results.html Phase 9 — Training dynamics results (generated after retraining)
│   ├── next_move_results.html Phase 10 — Next-move variant comparison
│   ├── superposition_results.html Phase 11 — SAE expansion sweep results
│   ├── progress_report.tex Full progress report (LaTeX)
│   ├── progress_report.pdf Compiled PDF (11 pages)
│   ├── references.bib      Bibliography
│   └── proposal.tex        Original project proposal
│
├── data/                   (gitignored — regenerate with uv run cube-dataset)
│   ├── distances.npz       BFS table: sorted uint64 packed keys + int8 distances (mmap-loaded)
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   ├── No_CoT.txt          LLM responses: all 5 representations × 3 distances (no chain-of-thought)
│   ├── CoT.txt             LLM responses: piece_identity + CoT × 3 distances
│   ├── Gemini_2_5_Pro.txt  Gemini 2.5 Pro responses: all 5 representations × distances 1–11
│   ├── GPT_4o.txt          GPT-4o responses: all 5 representations × distances 1–11
│   ├── GPT_5_2.txt         GPT-5.2 responses: all 5 representations × distances 1–11
│   ├── GPT_5_4.txt         GPT-5.4 responses: all 5 representations × distances 1–11
│   └── Llama3.3_70B_Groq.txt  Llama 3.3 70B (Groq) responses: all 5 representations × distances 1–11
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
uv run python -m interp.probe

# Phase 5a: activation patching
uv run python -m interp.patch

# Phase 5b: logit lens + tuned lens
uv run python -m interp.tuned_lens

# Phase 5c: sparse autoencoder
uv run python -m interp.sae

# Phase 6: generate text representation test cases for manual LLM evaluation
uv run python print_test_cases.py

# Phase 7: circuit identification (DLA, activation patching, neuron DLA)
uv run python -m interp.circuit

# Phase 8: weight-level analysis (embedding SVD, direct read-out, neuron profiles)
uv run python -m interp.weights

# Phase 9: training dynamics / grokking (requires re-training with --save-every 1)
uv run cube-train --save-every 1
uv run python -m interp.grokking

# Phase 10: next-move prediction variant
uv run python train.py --task next_move --out checkpoints/next_move
uv run python -m interp.next_move

# Phase 11: superposition analysis (SAE expansion sweep)
uv run python -m interp.superposition
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

An initial pilot (GPT-5 via chat, 1 case per distance, no code prohibition — the model frequently wrote Python to find solutions) gave:

| Representation | d=3 | d=5 | d=7 |
|---|---|---|---|
| `face_grid` — unfolded net diagram | ✓ | ✗ | ✗ |
| `compact_string` — 24-char flat string | ✓ | ✗ | ✓ |
| `corner_cubies` — colors per face per position | ✓ | ✗ | ✗ |
| **`piece_identity`** — piece name + current position + W/Y face | ✓ | **✓** | ✗ |
| `move_sequence` — the scramble itself (degenerate baseline) | ✓ | ✓ | ✓ |
| `piece_identity` + chain-of-thought | ✓ | ✗ | ✗ |

A subsequent automated evaluation (API, 10 cases per distance, d=3–11, code explicitly prohibited) tested all five representations including `piece_identity` and found that **all state-based representations score 0% across every model and every distance**. Only `move_sequence` (trivial inversion) varies:

| Model | d=3 | d=5 | d=7 | d=9 | d=11 | Total |
|---|---|---|---|---|---|---|
| GPT-5.4 | 10/10 | 10/10 | 7/10 | 6/10 | 9/10 | 42/50 (84%) |
| GPT-5.2 | 9/10 | 9/10 | 8/10 | 8/10 | 9/10 | 43/50 (86%) |
| Gemini 2.5 Pro | 9/10 | 8/10 | 9/10 | 8/10 | 1/2* | 35/42 (83%) |
| GPT-4o | 5/10 | 1/10 | 1/10 | 0/10 | 1/10 | 8/50 (16%) |
| Llama 3.3 70B | 3/10 | 1/10 | 1/10 | 0/10 | 0/2* | 5/42 (12%) |

\* Only 2 d=11 cases captured in that run.

Frontier models (GPT-5.x, Gemini 2.5 Pro) reliably invert move sequences (83–86%); GPT-4o and Llama trail at 12–16%, reflecting differences in notation parsing rather than cube understanding. The true barrier for state-based representations is move simulation itself — not the choice of representation and not the scramble depth.

**Phase 7 — Circuit identification:**
- `mlp_0` has by far the highest DLA (+5.6), making it the dominant component in the circuit; all attention layers contribute negatively (−0.2 to −1.4)
- Activation patching confirms this: patching `mlp_0` produces the largest prediction shift; attention patching has almost no effect
- Top individual neurons are L3N460 and L3N7 (DLA +2.36/+2.30) followed by L0N399 and L0N224 — a small handful of neurons account for a disproportionate share of the distance signal

**Phase 8 — Weight-level analysis:**
- The direct read-out path W_U @ W_E achieves only 8.4% accuracy (random baseline 8.3%) — the embedding alone, without any MLP computation, cannot linearly predict distance; the transformer blocks are necessary
- The top-8 embedding singular values are nearly equal (3.32–2.87), suggesting the embedding is roughly isotropic and does not preferentially align with any single input direction
- Top neuron input profiles (fc1.weight[n] @ W_E, reshaped to 24×6) show structured sticker-color selectivity; output profiles (W_U @ fc2.weight[:,n]) reveal which neurons promote specific distance classes vs. suppress them

**Phase 10 — Next-move prediction variant:**
- A model trained on next-move prediction (18 random scramble moves) achieves only 6.3% val accuracy — barely above the 5.6% random baseline — because scramble moves are unpredictable from cube state alone
- Despite essentially random training signal, the residual stream encodes *some* distance information: optimal-distance probe MAE drops from 1.77 (predict-mean) to 0.98 at the embedding, then only to 0.82 at L3
- The distance-supervised model, by contrast, reaches MAE 0.40 at L3 from the same 0.98 embedding start — nearly 2× better
- Conclusion: the embedding layer independently captures substantial distance structure (both models start at MAE 0.98), but the transformer blocks only build on this when directly supervised; distance representation does not emerge as a byproduct of next-move training

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
