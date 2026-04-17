# Rubik's Cube Mechanistic Interpretability

A mechanistic interpretability project using the 2×2×2 Rubik's cube as a controlled setting. We train a small transformer on optimal-distance classification, then probe and patch its residual stream to understand what it learns and how.

## What this project does

1. **Simulates** a 2×2×2 cube with a one-hot state encoding (24 stickers × 6 colors = 144-dim vector)
2. **Generates** a dataset of scrambled cube states with BFS-computed optimal distances (God's number = 11)
3. **Trains** a small TransformerLens transformer to classify optimal distance (0–11) from state
4. **Probes** each residual stream layer with logistic regression to find linearly decodable features
5. **Patches** activations causally to distinguish features the model *uses* from those it merely encodes
6. **Tuned lens** — trains a per-layer affine transform to read off predictions at each intermediate layer, revealing how the model builds up its answer
7. **Sparse autoencoder** — trains an overcomplete SAE on each layer's residual stream to find monosemantic features and check alignment with known concepts

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

## Project structure

```
cube.py            — 2×2×2 simulator, one-hot encoding, BFS distance computation
cube_visualizer.py — Interactive tkinter visualizer
dataset.py         — Dataset generation, split saving/loading, BFS cache management
model.py           — CubeTransformer (TransformerLens HookedRootModule)
train.py           — Training loop (AdamW + cosine annealing, class-weighted CE)
probe.py           — Phase 4: linear probes on residual stream activations
patch.py           — Phase 5a: concept-direction and counterfactual activation patching
tuned_lens.py      — Phase 5b: logit lens + trained per-layer affine lens
sae.py             — Phase 5c: sparse autoencoder, feature alignment analysis
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Generate dataset (computes BFS distances on first run, ~25 min; cached afterward)
uv run cube-dataset

# Train the transformer (~30 epochs, ~5 min on MPS/GPU)
uv run cube-train

# Phase 4: linear probing
uv run python probe.py

# Phase 5a: activation patching
uv run python patch.py

# Phase 5b: logit lens + tuned lens
uv run python tuned_lens.py

# Phase 5c: sparse autoencoder
uv run python sae.py
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

The BFS over the full 88M-state space (24 × 3.7M, since global orientation isn't fixed) takes ~25 minutes and is cached to `data/distances_cache.pkl`.

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
