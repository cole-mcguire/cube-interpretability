# Rubik's Cube Interpretability — Project Notes

Design decisions and context from initial planning session. Drop this file in your repo
and share it with Claude Code at the start of each session for full context.

---

## Project goal

Use the 2×2×2 Rubik's cube as a small, controlled setting for mechanistic interpretability.
Train a small transformer on next-move prediction, then use linear probes on the hidden
states to detect whether the model learns structured internal representations of cube
progress (scramble depth, face solved, corner orientation). Optionally follow up with
tuned lens or activation patching.

---

## Project timeline (5 phases)

| Phase | Task | Est. time |
|-------|------|-----------|
| 1 | Cube representation + simulator | 1 week — **done** |
| 2 | Dataset design + probe label definition | 1 week |
| 3 | Train small transformer (next-move prediction) | 1–2 weeks |
| 4 | Linear probing across layers | 1–2 weeks |
| 5 | Extensions: tuned lens, activation patching, SAE | 1 week each, pick one |

---

## Phase 1 decisions — cube representation

### Sticker layout

24 stickers, indices 0..23, grouped by face in this order:

```
U (top):   stickers  0– 3   color W (white),  index 0
D (bottom):stickers  4– 7   color Y (yellow), index 1
F (front): stickers  8–11   color O (orange), index 2
B (back):  stickers 12–15   color R (red),    index 3
L (left):  stickers 16–19   color G (green),  index 4
R (right): stickers 20–23   color B (blue),   index 5
```

Within each face, stickers are numbered in reading order:
top-left → top-right → bottom-left → bottom-right.

### Color-to-index map

```python
COLOR_TO_IDX = {"W": 0, "Y": 1, "O": 2, "R": 3, "G": 4, "B": 5}
```

### State encoding: one-hot per sticker (chosen)

Each sticker is one-hot encoded into a 6-dim binary vector.
All 24 stickers concatenated → **144-dim float32 vector** per state.

```python
# One line in numpy:
encoded = np.eye(6, dtype=np.float32)[state].flatten()  # state: (24,) int → (144,) float32
```

Alternatives considered and rejected:
- Flat integer array (24 ints): requires embedding layer, less transparent for probing
- Cubie-based encoding (8 corners × 3 orientations): compact but harder to implement;
  geometric abstraction may interfere with probing

### Transformer input: single token (chosen)

The 144-dim encoded state is treated as **one token** projected via a linear layer into
the model's hidden dim. Simplest to implement and debug for a first pass.

Alternative considered: 24 tokens of dim 6 (one per sticker). More expressive, lets
attention operate over individual stickers. Can revisit if single-token probes are weak.

### Move vocabulary

18 moves = 6 faces × 3 turn types (CW, CCW, 180°):

```
Index: 0=U  1=U' 2=U2  3=D  4=D' 5=D2  6=F  7=F' 8=F2
       9=B 10=B' 11=B2 12=L 13=L' 14=L2 15=R 16=R' 17=R2
```

`move_idx // 3` → face, `move_idx % 3` → turn type (0=CW, 1=CCW, 2=180°).

---

## Implemented: `cube.py`

File: `cube.py`  
Run tests: `python cube.py`  
All 10 unit tests pass.

### Key classes and functions

```python
Cube()                          # solved cube
Cube(state)                     # from (24,) int8 array

cube.apply_move(move_idx)       # in-place, returns self for chaining
cube.apply_move_name("U'")      # by name
cube.scramble(n, rng)           # random n moves, returns move list
Cube.from_scramble(n, rng)      # classmethod → (cube, moves)

cube.encode()                   # → (144,) float32 one-hot vector
Cube.decode(encoded)            # → Cube (inverts encode)

cube.is_solved()                # → bool
cube.face_solved()              # → (6,) bool — one per face
cube.corner_oriented()          # → (8,) bool — U/D color facing up/down

generate_dataset(n_sequences, max_scramble_depth)
# → dict with keys:
#     states:           float32 (N, 144)
#     next_moves:       int64   (N,)
#     scramble_depth:   int32   (N,)
#     face_solved:      bool    (N, 6)
#     corner_oriented:  bool    (N, 8)
```

### Known fragility

`B_CYCLES` and `R_CYCLES` permutation tables are the trickiest to get right.
If probe accuracy on back/right face lags badly, audit those tables first.
Sanity check: apply `U F U' F'` and compare result to a physical cube or online simulator.

---

## Phase 2 plan — dataset and task design

### Training task

**Next-move prediction**: given a cube state, predict which of the 18 moves was applied
next in the scramble sequence. Cross-entropy loss over 18 classes.

Alternative considered: move-quality scoring (binary: does this move help or hurt?).
Simpler but less information-rich. Stick with next-move prediction unless training is slow.

### Probe targets (already extracted in `generate_dataset`)

| Label | Type | Shape | Notes |
|-------|------|-------|-------|
| `scramble_depth` | int (0–11) | `(N,)` | Continuous or bucketed regression target |
| `face_solved` | bool | `(N, 6)` | One probe per face, or multi-label |
| `corner_oriented` | bool | `(N, 8)` | One probe per corner |

Simple concepts (scramble depth, face solved) expected to be easier to decode than
full cube state. Later layers expected to be more informative than earlier ones —
this is the main hypothesis to test.

### Suggested dataset size

- ~50k–100k sequences, scramble depth 1–11
- Should be fast to generate (pure numpy, no GPU needed)

---

## Phase 3 plan — transformer architecture

### Suggested starting architecture

```
Input:        (144,) float32 one-hot state
Linear proj:  144 → hidden_dim (e.g. 64 or 128)
+ learned positional encoding (single token, so optional)
Transformer:  2–4 layers, 2–4 heads, hidden_dim 64–128
MLP head:     hidden_dim → 18 (logits over moves)
Loss:         CrossEntropyLoss
```

Small on purpose — interpretability is the goal, not performance.
Use TransformerLens for the model so hooks are available from day one.

### Validation criteria

Before moving to probing, confirm the model:
1. Predicts legal moves (all 18 are valid, so this is trivially true)
2. Does better than random (random = 1/18 ≈ 5.6% accuracy)
3. Shows some sensitivity to scramble depth (harder positions → different move distribution)

---

## Phase 4 plan — linear probing

### Approach

For each layer's residual stream activations (hidden_dim vector per state):
- Train a logistic regression probe for each binary label (face_solved × 6, corner_oriented × 8)
- Train a linear regression probe for scramble_depth
- Compare accuracy across layers → expect later layers to be more informative

Use TransformerLens hooks to extract activations:
```python
# Example hook pattern
cache = {}
model.run_with_cache(encoded_states)  # fills cache with per-layer activations
acts = cache["resid_post", layer_idx]  # shape (batch, hidden_dim)
```

### Tools

- `sklearn.linear_model.LogisticRegression` for binary probes
- `sklearn.linear_model.Ridge` for depth regression
- Existing `probe_viz.py` / Plotly dashboard pattern from your `linear_probe` project

---

## Phase 5 options (pick one)

1. **Tuned lens**: train an affine transform per layer that maps residual stream →
   logit space; see how predictions evolve across layers. Uses the `tuned-lens` library
   or manual implementation following Belrose et al. 2025.

2. **Activation patching**: identify the probe direction for a concept (e.g. "face 0 solved"),
   then patch activations along that direction and observe whether the model's move
   preferences change. Causal test of whether the model *uses* the feature.

3. **Sparse autoencoder (SAE)**: train an SAE on the residual stream; inspect whether
   sparse features align with interpretable cube concepts. More ambitious scope.

---

## References

- Agostinelli et al. 2019 — solving with deep RL
- Takano 2023 — self-supervised cube solving
- Chasmai 2023 — CubeTR, transformer-based solver
- Alain & Bengio 2016 — linear classifier probes
- Belrose et al. 2025 — tuned lens
- Belinkov 2022 — probing classifiers: promises and shortcomings

---

## Environment

- Python with `uv` as package manager
- TransformerLens for model + hooks
- NumPy, scikit-learn, Plotly
- Local Mac development