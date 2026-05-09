# Rubik's Cube Mechanistic Interpretability

A mechanistic interpretability project using the 2×2×2 Rubik's cube as a controlled setting. The project has two parallel lines of work:

- **Interpretability track** — train a small transformer on optimal-distance classification, then probe, patch, and dissect its residual stream to understand what it learns and how
- **LLM evaluation track** — benchmark seven pre-trained LLMs across eight text representations of cube state to test whether in-context reasoning alone can solve scrambles

**Website:** [cole-mcguire.github.io/cube-interpretability](https://cole-mcguire.github.io/cube-interpretability)

**Final report:** [`docs/progress_report.pdf`](docs/progress_report.pdf) (source: [`docs/progress_report.tex`](docs/progress_report.tex)).

## TL;DR results

| Finding | Result |
|---------|--------|
| Distance classification accuracy | **78.5%** val (random baseline 8.3%) |
| Linear probes | `face_solved` decodable ≥98% at every layer; distance MAE drops from 0.87 → 0.40 |
| Causal patching | Swapping full residual stream between distance classes flips predictions at 100% — distance is linearly decodable from the embedding before any block runs, though not yet aligned with the final readout basis |
| Circuit | `mlp_0` dominates (DLA +5.6); attention heads contribute negatively; ≤10 neurons explain most of the signal |
| Superposition | No evidence of classical superposition — SAE R² ≥0.999 at 1× expansion; representations are near-orthogonal |
| LLM eval | State-based representations score effectively 0% (1 corner_cubies trial solved out of 1,190 state-based trials); only `move_sequence` (trivial inversion) varies (GPT-5.x: 100%, Claude/GPT-4o: 32–48%) |
| Corner model | 76.2% accuracy; L0 heads are critical (ablating L0H1 drops 44.9 pp); attention sharpens monotonically with scramble depth |

**Start here:**
- **Interpretability results** → [website](https://cole-mcguire.github.io/cube-interpretability) or `docs/probe_results.html`, `docs/circuit_results.html`
- **LLM evaluation results** → `docs/llm_eval_results.html`
- **Full write-up** → `docs/progress_report.pdf`
- **Interactive cube + solver** → `docs/index.html` or `uv run cube-visualizer`

## What this project does

1. **Simulates** a 2×2×2 cube with a one-hot state encoding (24 stickers × 6 colors = 144-dim vector)
2. **Solves** any cube state optimally via BFS distance-table lookup, and generates scramble/solution pairs
3. **Generates** a dataset of scrambled cube states with BFS-computed optimal distances (God's number = 11)
4. **Trains** a small TransformerLens transformer to classify optimal distance (0–11) from state
5. **Probes** each residual stream layer with logistic regression to find linearly decodable features
6. **Patches** activations causally to distinguish features the model *uses* from those it merely encodes
7. **Tuned lens** — trains a per-layer affine transform to read off predictions at each intermediate layer, revealing how the model builds up its answer
8. **Sparse autoencoder** — trains an overcomplete SAE on each layer's residual stream to find monosemantic features and check alignment with known concepts
9. **Text representations** — evaluates eight ways to describe cube state in natural language and tests whether a pre-trained LLM can solve scrambles from each
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
│   ├── compute_optimal_distances()   Full BFS (~10 min, cached as data/distances.npz)
│   └── generate_dataset()  Scramble sequences with labels
│
├── cube_visualizer.py      Interactive tkinter visualizer
│   ├── 2D net view         Flat unfolded cube with per-sticker color labels
│   ├── 3D view             Drag-to-rotate orthographic projection (no extra deps)
│   └── Solver panel        Load table, Step / ▶ Play through optimal solution
│
├── dataset.py              Generate and save train/val/test splits (.npz)
├── model.py                CubeTransformer — TransformerLens HookedRootModule
│                           d_model=128, n_layers=4, n_heads=4 (~800k params)
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
│   ├── superposition.py    Phase 11 — SAE expansion sweep
│   │                       R², active features, dead fraction vs expansion ratio
│   ├── corner_analysis.py  Phase 12 — corner-tokenized model analysis
│   │                       Accuracy vs flat model, 8×8 attention heatmaps, entropy by distance
│   └── corner_attn.py      Phase 13 — attention head ablation and specialization
│                           Accuracy drop per head, mean 8×8 patterns, U/D layer bias, distance modulation
│
├── print_test_cases.py     Phase 6 — generates text representations of scrambled
│                           states for manual LLM testing; outputs 8 formats
│                           (face_grid, compact_string, corner_cubies,
│                           piece_identity, move_sequence, natural_language,
│                           perm_orient, cycle_notation) + CoT variant
│
├── test_representations.py Phase 17 — automated API evaluation of all 8 representations
│                           across 7 LLMs (OpenAI, Gemini, Anthropic, Groq);
│                           results cached in data/llm_eval_cache.json
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
│   ├── corner_results.html Phase 12 — corner-tokenized model results
│   ├── corner_attn_results.html Phase 13 — head ablation and specialization
│   ├── progress_report.tex Full final report (LaTeX)
│   ├── progress_report.pdf Compiled PDF (18 pages)
│   └── references.bib      Bibliography
│
├── data/                   (.npz files gitignored — regenerate with uv run cube-dataset;
│   │                        JSON and .txt files are tracked in git)
│   ├── distances.npz       BFS table: sorted uint64 packed keys + int8 distances (gitignored)
│   ├── train.npz           (gitignored)
│   ├── val.npz             (gitignored)
│   ├── test.npz            (gitignored)
│   ├── llm_eval_cache.json LLM eval cache: 7 models × 8 representations × d=3–11 (tracked)
│   │                       (N=100/distance for move_sequence; N=10/distance for state reps)
│   ├── No_CoT.txt          Early pilot responses (tracked)
│   ├── CoT.txt             Early pilot responses with chain-of-thought (tracked)
│   ├── Gemini_2_5_Pro.txt  Early pilot: Gemini 2.5 Pro (tracked)
│   ├── GPT_4o.txt          Early pilot: GPT-4o (tracked)
│   ├── GPT_5_2.txt         Early pilot: GPT-5.2 (tracked)
│   ├── GPT_5_4.txt         Early pilot: GPT-5.4 (tracked)
│   └── Llama3.3_70B_Groq.txt  Early pilot: Llama 3.3 70B (tracked)
│
├── checkpoints/            (gitignored — regenerate with uv run cube-train)
│   ├── best.pt             main distance-classifier checkpoint
│   ├── epoch_001.pt ...    per-epoch checkpoints for training dynamics (Phase 9)
│   ├── corner/best.pt      corner-tokenized model (Phases 12–15)
│   └── next_move/best.pt   next-move variant (Phase 10)
│
└── pyproject.toml          uv project; scripts: cube-tests, cube-visualizer,
                            cube-dataset, cube-train
```

## Solver API

`cube.py` exposes two functions for optimal solving (requires the BFS distance table):

```python
from cube import Cube, DistanceTable, MOVE_NAMES, solve, generate_scramble_solution_pairs

distances = DistanceTable.load("data/distances.npz")  # mmap, ~0.1 s

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
# Install core dependencies
uv sync

# Install LLM API clients (needed for Phase 17 / test_representations.py)
uv sync --extra llm

# Run unit tests (cube simulator + BFS correctness)
uv run cube-tests

# Generate dataset (computes BFS distances on first run, ~10 min; cached afterward)
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

# Phase 6/17: generate text representation test cases for manual LLM evaluation
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

# Phase 12: corner-tokenized transformer (train + analyze)
uv run python train.py --arch corner --out checkpoints/corner
uv run python -m interp.corner_analysis

# Phase 13: attention head ablation and specialization
uv run python -m interp.corner_attn

# Phase 17: LLM evaluation (requires API keys; results cached in data/llm_eval_cache.json)
# Set env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY as needed
uv sync --extra llm
uv run python test_representations.py --models claude-sonnet-4-6 --n 10
uv run python test_representations.py --models gpt-4o --reps move_sequence --n 100
```

Output files: `data/` (splits + BFS cache), `checkpoints/best.pt`, and HTML visualizations for each analysis step.

## Browser model regeneration

The interactive visualizer in `docs/index.html` uses a compiled ONNX model (`docs/model.onnx`). To regenerate it after retraining:

```bash
uv run python scripts/export_onnx.py          # exports checkpoints/best.pt → docs/model.onnx
```

The script uses the legacy TorchScript ONNX exporter (`dynamo=False`) with embedded weights for browser compatibility and includes an `onnxruntime` sanity check if installed.

## Model architecture

```
Input:   (batch, 144)  float32 one-hot cube state
Embed:   Linear(144 → d_model)          hook_embed
Blocks:  TransformerBlock × n_layers
           LN → Attention → residual    hook_resid_mid
           LN → MLP      → residual     hook_resid_post
Head:    LayerNorm → Linear(d_model, 12)
```

Default config: `d_model=128, n_layers=4, n_heads=4` (~800k parameters).

Note: input is a single token, so attention weights are always 1.0. Computation flows almost entirely through the MLP sublayers. All hook points follow TransformerLens naming, so `model.run_with_cache()` works out of the box.

**Training result:** 78.5% val accuracy on 12-class distance classification (random baseline: 8.3%).

## Dataset

Each `.npz` split contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `states` | `(N, 144)` float32 | One-hot encoded cube state |
| `optimal_distance` | `(N,)` int8 | BFS distance to solved (0–11) |
| `next_moves` | `(N,)` int64 | Move applied to reach this state |
| `scramble_depth` | `(N,)` int32 | Steps from solved in the scramble sequence |
| `face_solved` | `(N, 6)` bool | Per-face solved status |
| `corner_oriented` | `(N, 8)` bool | Per-corner U/D orientation |

Default split sizes: 50k / 5k / 5k sequences → ~300k / 30k / 30k samples.

The BFS reaches the full 88M-state space from a single solved seed (face moves connect every rotationally-equivalent solved state). The one-time BFS takes ~10 min (chunked vectorised numpy); the result is packed into a sorted `(uint64, int8)` array pair and written to `data/distances.npz` (~760 MB). Subsequent loads are memory-mapped and effectively instant (~0.1 s), and lookup is `O(log N)` via `np.searchsorted`.

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

**Phase 17 — LLM evaluation of text representations:**

An initial pilot (GPT-5 via chat, 1 case per distance, no code prohibition — the model frequently wrote Python to find solutions) gave:

| Representation | d=3 | d=5 | d=7 |
|---|---|---|---|
| `face_grid` — unfolded net diagram | ✓ | ✗ | ✗ |
| `compact_string` — 24-char flat string | ✓ | ✗ | ✓ |
| `corner_cubies` — colors per face per position | ✓ | ✗ | ✗ |
| **`piece_identity`** — piece name + current position + W/Y face | ✓ | **✓** | ✗ |
| `move_sequence` — the scramble itself (degenerate baseline) | ✓ | ✓ | ✓ |
| `piece_identity` + chain-of-thought | ✓ | ✗ | ✗ |

A large-scale automated evaluation (API, 100 cases/distance for `move_sequence`, 10 cases/distance for state reps, d=3–11, code prohibited) extended the study to **8 representations** and **7 models**. State-based representations score effectively 0% — one `corner_cubies` trial (Claude Sonnet 4.6, d=3) solved out of 1,190 state-based trials total. This includes three new representations (`natural_language`, `perm_orient`, `cycle_notation`). Only `move_sequence` (trivial inversion) varies:

| Model | d=3 | d=5 | d=7 | d=9 | d=11 | Total |
|---|---|---|---|---|---|---|
| GPT-5.5 | 100/100 | 100/100 | 100/100 | 100/100 | 48/48 | 448/448 (100%) |
| GPT-5.4 | 100/100 | 99/100 | 100/100 | 100/100 | 47/48 | 446/448 (100%) |
| Gemini 2.5 Pro | 98/100 | 92/100 | 84/100 | 72/100 | 33/48 | 379/448 (85%) |
| Claude Sonnet 4.6 | 62/100 | 58/100 | 45/100 | 41/100 | 11/48 | 217/448 (48%) |
| Claude Opus 4.7 | 55/100 | 41/100 | 27/100 | 15/100 | 4/48 | 142/448 (32%) |
| GPT-4o | 62/100 | 44/100 | 18/100 | 18/100 | 3/48 | 145/448 (32%) |
| Llama 3.3 70B | — | — | — | — | — | insufficient data† |

† Groq free-tier rate limits prevented completing the N=100 run for Llama.

Frontier reasoning models (GPT-5.x) achieve near-perfect move-sequence inversion; Gemini 2.5 Pro scores 85%; Claude models and GPT-4o trail at 32–48%; Llama at 12% in the earlier pilot. The true barrier for state-based representations is move simulation itself — not the choice of representation and not the scramble depth. No representation, including verbose natural-language or group-theoretic encodings, meaningfully enabled cube solving (1/1,190 state-based trials).

**Phase 7 — Circuit identification:**
- `mlp_0` has by far the highest DLA (+5.6), making it the dominant component in the circuit; all attention layers contribute negatively (−0.2 to −1.4)
- Activation patching confirms this: patching `mlp_0` produces the largest prediction shift; attention patching has almost no effect
- Top individual neurons are L3N460 and L3N7 (DLA +2.36/+2.30) followed by L0N399 and L0N224 — a small handful of neurons account for a disproportionate share of the distance signal

**Phase 8 — Weight-level analysis:**
- The direct read-out path W_U @ W_E achieves only 8.4% accuracy (random baseline 8.3%) — the embedding alone, without any MLP computation, cannot linearly predict distance; the transformer blocks are necessary
- The top-8 embedding singular values are nearly equal (3.32–2.87), suggesting the embedding is roughly isotropic and does not preferentially align with any single input direction
- Top neuron input profiles (fc1.weight[n] @ W_E, reshaped to 24×6) show structured sticker-color selectivity; output profiles (W_U @ fc2.weight[:,n]) reveal which neurons promote specific distance classes vs. suppress them

**Phase 9 — Training dynamics:**
- No sharp grokking-style phase transition — accuracy builds gradually with a mild plateau around epochs 5–8 before continuing to improve
- `mlp_0` DLA grows monotonically throughout training; distance probes improve smoothly at every layer — consistent with incremental learning rather than sudden generalization
- Per-distance accuracy heatmap shows easy distances (d=0–2) converge first; hard distances (d=5–9) improve slowly and uniformly across epochs

**Phase 10 — Next-move prediction variant:**
- A model trained on next-move prediction (18 random scramble moves) achieves only 6.3% val accuracy — barely above the 5.6% random baseline — because scramble moves are unpredictable from cube state alone
- Despite essentially random training signal, the residual stream encodes *some* distance information: optimal-distance probe MAE drops from 1.77 (predict-mean) to 0.98 at the embedding, then only to 0.82 at L3
- The distance-supervised model, by contrast, reaches MAE 0.40 at L3 from the same 0.98 embedding start — nearly 2× better
- Conclusion: the embedding layer independently captures substantial distance structure (both models start at MAE 0.98), but the transformer blocks only build on this when directly supervised; distance representation does not emerge as a byproduct of next-move training

**Phase 11 — Superposition analysis (SAE expansion sweep 1×–16×):**

| Expansion | embed R² | L0 R² | L1 R² | L2 R² | L3 R² |
|---|---|---|---|---|---|
| 1× (128) | 0.9994 | 0.9917 | 0.9961 | 0.9978 | 0.9982 |
| 4× (512) | 0.9996 | 0.9804 | 0.9988 | 0.9996 | 0.9990 |
| 16× (2048) | 0.9993 | 0.9311 | 0.9973 | 0.9927 | 0.9927 |

- R² is >0.999 at **1× expansion** for embed/L1–L3 — residual streams are low-dimensional enough that a same-size dictionary nearly perfectly reconstructs them; there is no reconstruction gap for larger dictionaries to fill
- L0 is anomalous: R² *degrades* above 4× due to training instability (L1 loss diverges at 16×), reflecting L0's much denser activation distribution as the dominant MLP layer
- L1, L2, L3 show **zero dead features** at every expansion ratio through 8×; L2 and L3 have zero dead even at 16× (2048 features in 128-dim space), meaning every dictionary element gets used and active count scales linearly with expansion
- There is **no plateau** in active feature count — no identifiable cutoff that would indicate a finite set of "true features" in superposition
- Conclusion: this model does not exhibit classical superposition. Small size (~800k params, 12 classes) and dense residual streams mean features are not packed above dimensionality; the representations are already near-orthogonal rather than superimposed

**Phase 12 — Corner-tokenized transformer:**
- Replaces the single 144-dim cube token with 8 per-corner tokens (18-dim each: 3 stickers × 6-color one-hot), making multi-head attention non-degenerate for the first time
- Corner model achieves **76.2%** val accuracy vs flat model's **79.8%** — a small gap given the structural advantage of per-cubie tokenization; more training or a larger model would likely close it
- Attention entropy is very low (L0–L3: 0.28–0.37 nats) vs uniform ceiling of 2.08 nats, indicating heads attend sharply to specific corner pairs rather than diffusely; the model learns concentrated inter-cubie routing
- Distance-stratified 8×8 attention heatmaps reveal which corner pairs the heads track at each solve distance

**Phase 13 — Attention head ablation and specialization:**
- Layer 0 dominates: ablating L0H1 drops accuracy by **44.9 pp** (76.2% → 31.3%), L0H0 by **30.7 pp**, L0H2 by **25.1 pp**; all L1–L3 heads are individually small (≤7.7 pp drop)
- The three critical L0 heads are also the sharpest: sharpness scores 0.760, 0.495, 0.701 (mean max attention weight) vs. the relatively diffuse L0H3 (0.570) and near-uniform L1–L3 heads
- Specialization analysis (U-layer: UFR,UFL,UBL,UBR vs D-layer: DFR,DFL,DBL,DBR) reveals which heads exhibit within-layer vs cross-layer routing preferences, surfacing geometric structure in the inter-cubie attention circuit

## Reproducibility notes

- **BFS distance table** (`data/distances.npz`, ~760 MB) is gitignored. Regenerate with `uv run cube-dataset` (~10 min, ~2 GB peak RAM). The table is memory-mapped on subsequent loads (~0.1 s, `O(log N)` lookup via `np.searchsorted`).
- **Model checkpoints** (`checkpoints/best.pt`) are gitignored. Regenerate with `uv run cube-train` (~30 epochs, ~5 min on MPS/GPU). The corner model and next-move model require separate training runs (see Quickstart).
- **LLM evaluation API keys** are not committed. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `GROQ_API_KEY` as environment variables before running `test_representations.py`. Results are cached in `data/llm_eval_cache.json` so re-runs are free.
- **LLM eval cost warning**: running the full automated eval (7 models × 8 reps × 100 cases/distance) incurs non-trivial API costs. The cached results in `data/llm_eval_cache.json` cover the full run; use them instead of re-running.
- **Move geometry** is generated from face normal vectors at startup (not hand-coded), so adding new cube sizes only requires changing the face list.

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
