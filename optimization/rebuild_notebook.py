"""
Generates optimization/optimization_experiments.ipynb as a fully self-contained
notebook that only requires numpy, torch, and matplotlib.
Run with:  uv run python optimization/rebuild_notebook.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

# ─────────────────────────────────────────────────────────────────
# Cell 1 — Title markdown
# ─────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "# Optimization Experiments — 2×2×2 Rubik’s Cube\n\n"
    "This notebook is **fully self-contained**: it requires only `numpy`, `torch`, and\n"
    "`matplotlib` — no external project files.\n\n"
    "**Experiments:**\n"
    "1. Optimizer comparison (AdamW vs Adam vs SGD with momentum)\n"
    "2. Learning rate schedule ablation (cosine annealing vs step decay vs constant)\n"
    "3. Hyperparameter sensitivity (d_model sweep, weight decay sweep)\n"
    "4. Learned heuristic quality (admissibility, MAE by distance, per-distance accuracy)\n"
    "5. Training dynamics — multi-seed comparison\n"
    "6. Loss landscape (filter-normalized, Li et al. 2018)\n"
    "7. A* search with learned heuristic vs Dijkstra\n\n"
    "**Expected runtimes (first run, no cache):**\n"
    "- BFS distance table: ~25–30 min\n"
    "- Dataset generation: ~3 min\n"
    "- Training (all experiments): ~60 min total\n"
    "- Loss landscape: ~5–10 min\n\n"
    "Results are cached to `results/` as JSON / npz so re-running the notebook skips\n"
    "already-completed work.\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 2 — Standard imports
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "import json, time, heapq\n"
    "from pathlib import Path\n"
    "from collections import defaultdict, deque\n"
    "from typing import Optional\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import DataLoader, TensorDataset\n"
    "import matplotlib\n"
    'matplotlib.use("Agg")\n'
    "import matplotlib.pyplot as plt\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 3 — Cube simulator
# We build this as a list of lines to avoid nested triple-quote issues.
# ─────────────────────────────────────────────────────────────────
_cube_cell_lines = [
    "# ---------------------------------------------------------------------------",
    "# Constants",
    "# ---------------------------------------------------------------------------",
    "",
    "NUM_STICKERS = 24",
    "NUM_COLORS   = 6",
    "STATE_DIM    = NUM_STICKERS * NUM_COLORS  # 144",
    "",
    'COLOR_TO_IDX = {"W": 0, "Y": 1, "O": 2, "R": 3, "G": 4, "B": 5}',
    "IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}",
    "",
    'MOVE_NAMES = ["U", "U\'", "U2", "D", "D\'", "D2",',
    '              "F", "F\'", "F2", "B", "B\'", "B2",',
    '              "L", "L\'", "L2", "R", "R\'", "R2"]',
    "NUM_MOVES      = len(MOVE_NAMES)  # 18",
    "MOVE_NAME_TO_IDX = {name: idx for idx, name in enumerate(MOVE_NAMES)}",
    "",
    "# Solved state: sticker i has color SOLVED_STATE[i]",
    "# U=W(0), D=Y(1), F=O(2), B=R(3), L=G(4), R=B(5)",
    "SOLVED_STATE = np.array(",
    "    [0]*4 +   # U face: stickers 0-3  -> White",
    "    [1]*4 +   # D face: stickers 4-7  -> Yellow",
    "    [4]*4 +   # F face: stickers 8-11 -> Green",
    "    [5]*4 +   # B face: stickers 12-15 -> Blue",
    "    [2]*4 +   # L face: stickers 16-19 -> Orange",
    "    [3]*4,    # R face: stickers 20-23 -> Red",
    "    dtype=np.int8",
    ")",
    "",
    "# ---------------------------------------------------------------------------",
    "# Move permutations",
    "# ---------------------------------------------------------------------------",
    "",
    'FACE_ORDER = ["U", "D", "F", "B", "L", "R"]',
    "FACE_AXES = {",
    '    "U": {"normal": (0, 1, 0), "right": (1, 0, 0), "down": (0, 0, 1)},',
    '    "D": {"normal": (0, -1, 0), "right": (1, 0, 0), "down": (0, 0, -1)},',
    '    "F": {"normal": (0, 0, 1), "right": (1, 0, 0), "down": (0, -1, 0)},',
    '    "B": {"normal": (0, 0, -1), "right": (-1, 0, 0), "down": (0, -1, 0)},',
    '    "L": {"normal": (-1, 0, 0), "right": (0, 0, 1), "down": (0, -1, 0)},',
    '    "R": {"normal": (1, 0, 0), "right": (0, 0, -1), "down": (0, -1, 0)},',
    "}",
    "",
    "",
    "def _v_add(a, b):",
    "    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])",
    "",
    "",
    "def _v_scale(v, k):",
    "    return (v[0] * k, v[1] * k, v[2] * k)",
    "",
    "",
    "def _v_neg(v):",
    "    return (-v[0], -v[1], -v[2])",
    "",
    "",
    "def _rotate_positive_axis(vec, axis, quarter_turns):",
    "    quarter_turns %= 4",
    "    if quarter_turns == 0:",
    "        return vec",
    "",
    "    x, y, z = vec",
    "    if axis == 0:",
    "        rotations = {",
    "            1: (x, -z, y),",
    "            2: (x, -y, -z),",
    "            3: (x, z, -y),",
    "        }",
    "    elif axis == 1:",
    "        rotations = {",
    "            1: (z, y, -x),",
    "            2: (-x, y, -z),",
    "            3: (-z, y, x),",
    "        }",
    "    else:",
    "        rotations = {",
    "            1: (-y, x, z),",
    "            2: (-x, -y, z),",
    "            3: (y, -x, z),",
    "        }",
    "    return rotations[quarter_turns]",
    "",
    "",
    "def _rotate_about_normal(vec, normal, quarter_turns):",
    "    axis = next(i for i, value in enumerate(normal) if value != 0)",
    "    sign = normal[axis]",
    "    return _rotate_positive_axis(vec, axis, quarter_turns * sign)",
    "",
    "",
    "def _cw_turn_for_face(face_name):",
    "    spec = FACE_AXES[face_name]",
    '    up = _v_neg(spec["down"])',
    '    if _rotate_about_normal(up, spec["normal"], 1) == spec["right"]:',
    "        return 1",
    "    return -1",
    "",
    "",
    "def _build_sticker_descriptors():",
    "    descriptors = []",
    "    for face_name in FACE_ORDER:",
    "        spec = FACE_AXES[face_name]",
    '        normal = spec["normal"]',
    '        right = spec["right"]',
    '        down = spec["down"]',
    "        for row in range(2):",
    "            for col in range(2):",
    "                row_sign = -1 if row == 0 else 1",
    "                col_sign = -1 if col == 0 else 1",
    "                position = _v_add(",
    "                    normal,",
    "                    _v_add(_v_scale(right, col_sign), _v_scale(down, row_sign)),",
    "                )",
    "                descriptors.append((position, normal))",
    "    return descriptors",
    "",
    "",
    "def _build_cw_permutation(face_name):",
    "    spec = FACE_AXES[face_name]",
    '    normal = spec["normal"]',
    "    axis = next(i for i, value in enumerate(normal) if value != 0)",
    "    layer_value = normal[axis]",
    "    quarter_turn = _cw_turn_for_face(face_name)",
    "",
    "    permutation = np.arange(NUM_STICKERS, dtype=np.int8)",
    "    for old_idx, (position, sticker_normal) in enumerate(STICKER_DESCRIPTORS):",
    "        if position[axis] != layer_value:",
    "            continue",
    "        new_position = _rotate_about_normal(position, normal, quarter_turn)",
    "        new_normal = _rotate_about_normal(sticker_normal, normal, quarter_turn)",
    "        new_idx = STICKER_INDEX[(new_position, new_normal)]",
    "        permutation[new_idx] = old_idx",
    "    return permutation",
    "",
    "",
    "def _build_global_rotation_permutation(axis, quarter_turns):",
    "    permutation = np.arange(NUM_STICKERS, dtype=np.int8)",
    "    for old_idx, (position, sticker_normal) in enumerate(STICKER_DESCRIPTORS):",
    "        new_position = _rotate_positive_axis(position, axis, quarter_turns)",
    "        new_normal = _rotate_positive_axis(sticker_normal, axis, quarter_turns)",
    "        new_idx = STICKER_INDEX[(new_position, new_normal)]",
    "        permutation[new_idx] = old_idx",
    "    return permutation",
    "",
    "",
    "def _build_cube_rotation_permutations():",
    "    identity = np.arange(NUM_STICKERS, dtype=np.int8)",
    "    generators = [",
    "        _build_global_rotation_permutation(axis=0, quarter_turns=1),",
    "        _build_global_rotation_permutation(axis=1, quarter_turns=1),",
    "        _build_global_rotation_permutation(axis=2, quarter_turns=1),",
    "    ]",
    "",
    "    permutations = [identity]",
    "    seen = {tuple(int(idx) for idx in identity)}",
    "    queue = [identity]",
    "",
    "    while queue:",
    "        current = queue.pop()",
    "        for generator in generators:",
    "            composed = current[generator]",
    "            key = tuple(int(idx) for idx in composed)",
    "            if key in seen:",
    "                continue",
    "            seen.add(key)",
    "            permutations.append(composed)",
    "            queue.append(composed)",
    "",
    "    return permutations",
    "",
    "",
    "STICKER_DESCRIPTORS = _build_sticker_descriptors()",
    "STICKER_INDEX = {descriptor: idx for idx, descriptor in enumerate(STICKER_DESCRIPTORS)}",
    "BASE_PERMUTATIONS = [_build_cw_permutation(face_name) for face_name in FACE_ORDER]",
    "INV_PERMUTATIONS = [np.argsort(p).astype(np.int8) for p in BASE_PERMUTATIONS]",
    "CUBE_ROTATION_PERMUTATIONS = _build_cube_rotation_permutations()",
    "SOLVED_SYMMETRY_STATES = np.stack(",
    "    [SOLVED_STATE[permutation] for permutation in CUBE_ROTATION_PERMUTATIONS]",
    ")",
    "",
    "# Base-6 packing: a (24,) state in [0..5] fits in uint64",
    "_POWERS_OF_6 = (6 ** np.arange(NUM_STICKERS, dtype=np.uint64)).astype(np.uint64)",
    "",
    "",
    "def pack_state(state):",
    '    "Pack a (24,) int8 state with values in [0..5] to a unique uint64."',
    "    return int((np.asarray(state, dtype=np.uint64) * _POWERS_OF_6).sum())",
    "",
    "",
    "def _pack_states_batch(states):",
    '    "Vectorized pack_state over a (N, 24) array."',
    "    return (states.astype(np.uint64) * _POWERS_OF_6).sum(axis=1)",
    "",
    "",
    "class DistanceTable:",
    '    "Dict-like wrapper over sorted uint64 keys + int8 distances."',
    "",
    "    def __init__(self, keys, values):",
    "        self._keys = np.asarray(keys, dtype=np.uint64)",
    "        self._values = np.asarray(values, dtype=np.int8)",
    "",
    "    def _as_packed(self, key):",
    "        if isinstance(key, (bytes, bytearray, memoryview)):",
    "            return pack_state(np.frombuffer(bytes(key), dtype=np.int8))",
    "        if isinstance(key, np.ndarray):",
    "            return pack_state(key)",
    "        return int(key)",
    "",
    "    def __getitem__(self, key):",
    "        k = self._as_packed(key)",
    "        idx = int(np.searchsorted(self._keys, np.uint64(k)))",
    "        if idx < len(self._keys) and int(self._keys[idx]) == k:",
    "            return int(self._values[idx])",
    "        raise KeyError(key)",
    "",
    "    def get(self, key, default=None):",
    "        try:",
    "            return self[key]",
    "        except KeyError:",
    "            return default",
    "",
    "    def __contains__(self, key):",
    "        return self.get(key, None) is not None",
    "",
    "    def __len__(self):",
    "        return int(len(self._keys))",
    "",
    "    def values(self):",
    "        return (int(v) for v in self._values)",
    "",
    "    @classmethod",
    "    def load(cls, path):",
    '        "Memory-map an .npz file saved by save()."',
    '        data = np.load(path, mmap_mode="r")',
    "        return cls(data[\"keys\"], data[\"values\"])",
    "",
    "    def save(self, path):",
    '        "Write to .npz."',
    "        np.savez(path, keys=self._keys, values=self._values)",
    "",
    "",
    "CORNER_STICKERS = [",
    "    (3, 9, 20),   # UFR",
    "    (2, 8, 17),   # UFL",
    "    (0, 13, 16),  # UBL",
    "    (1, 12, 21),  # UBR",
    "    (5, 11, 22),  # DFR",
    "    (4, 10, 19),  # DFL",
    "    (6, 15, 18),  # DBL",
    "    (7, 14, 23),  # DBR",
    "]",
    "VALID_CORNER_COLOR_SETS = {",
    "    tuple(sorted(int(SOLVED_STATE[idx]) for idx in stickers))",
    "    for stickers in CORNER_STICKERS",
    "}",
    "",
    "# ---------------------------------------------------------------------------",
    "# Cube class",
    "# ---------------------------------------------------------------------------",
    "",
    "class Cube:",
    '    "2x2x2 Rubik\'s Cube."',
    "",
    "    def __init__(self, state=None):",
    "        if state is not None:",
    "            if state.shape != (NUM_STICKERS,):",
    "                raise ValueError(f\"State must have shape ({NUM_STICKERS},), got {state.shape}\")",
    "            self.state = state.astype(np.int8)",
    "        else:",
    "            self.state = SOLVED_STATE.copy()",
    "",
    "    def apply_move(self, move_idx):",
    "        if not 0 <= move_idx < NUM_MOVES:",
    "            raise ValueError(f\"Invalid move index {move_idx}\")",
    "        face_idx = move_idx // 3",
    "        turn_type = move_idx % 3",
    "",
    "        if turn_type == 0:",
    "            self.state = self.state[BASE_PERMUTATIONS[face_idx]]",
    "        elif turn_type == 1:",
    "            self.state = self.state[INV_PERMUTATIONS[face_idx]]",
    "        else:",
    "            p = BASE_PERMUTATIONS[face_idx]",
    "            self.state = self.state[p][p]",
    "",
    "        return self",
    "",
    "    def apply_move_name(self, name):",
    "        if name not in MOVE_NAME_TO_IDX:",
    "            raise ValueError(f\"Unknown move name {name!r}\")",
    "        return self.apply_move(MOVE_NAME_TO_IDX[name])",
    "",
    "    def scramble(self, n, rng=None):",
    "        if rng is None:",
    "            rng = np.random.default_rng()",
    "",
    "        moves = []",
    "        last_face = -1",
    "        for _ in range(n):",
    "            candidates = [m for m in range(NUM_MOVES) if m // 3 != last_face]",
    "            move = int(rng.choice(candidates))",
    "            self.apply_move(move)",
    "            moves.append(move)",
    "            last_face = move // 3",
    "        return moves",
    "",
    "    @classmethod",
    "    def from_scramble(cls, n, rng=None):",
    "        cube = cls()",
    "        moves = cube.scramble(n, rng)",
    "        return cube, moves",
    "",
    "    def encode(self):",
    "        return np.eye(NUM_COLORS, dtype=np.float32)[self.state].flatten()",
    "",
    "    @staticmethod",
    "    def decode(encoded):",
    "        if encoded.shape != (STATE_DIM,):",
    "            raise ValueError(f\"Encoded state must have shape ({STATE_DIM},), got {encoded.shape}\")",
    "        oh = encoded.reshape(NUM_STICKERS, NUM_COLORS)",
    "        state = oh.argmax(axis=1).astype(np.int8)",
    "        return Cube(state)",
    "",
    "    def is_solved(self):",
    "        return bool(np.any(np.all(SOLVED_SYMMETRY_STATES == self.state, axis=1)))",
    "",
    "    def face_solved(self):",
    "        solved = np.zeros(6, dtype=bool)",
    "        for f in range(6):",
    "            face_stickers = self.state[f*4 : f*4+4]",
    "            solved[f] = bool((face_stickers == face_stickers[0]).all())",
    "        return solved",
    "",
    "    def corner_oriented(self):",
    '        ud_colors = {COLOR_TO_IDX["W"], COLOR_TO_IDX["Y"]}',
    "        oriented = np.zeros(8, dtype=bool)",
    "        for i, (ud_idx, _, _) in enumerate(CORNER_STICKERS):",
    "            oriented[i] = self.state[ud_idx] in ud_colors",
    "        return oriented",
    "",
    "    def copy(self):",
    "        return Cube(self.state.copy())",
    "",
    "    def __repr__(self):",
    "        colors = [IDX_TO_COLOR[int(c)] for c in self.state]",
    '        faces = ["U", "D", "F", "B", "L", "R"]',
    "        lines = []",
    "        for f, name in enumerate(faces):",
    "            row = colors[f*4 : f*4+4]",
    "            lines.append(f\"{name}: [{row[0]}{row[1]}|{row[2]}{row[3]}]\")",
    '        return "  ".join(lines)',
    "",
    "",
    "# ---------------------------------------------------------------------------",
    "# Optimal distance (BFS)",
    "# ---------------------------------------------------------------------------",
    "",
    "def compute_optimal_distances(verbose=True):",
    '    "BFS over every reachable state (~88M). Returns a DistanceTable."',
    "    distances_dict = {}",
    "    queue = deque()",
    "",
    "    key = SOLVED_STATE.tobytes()",
    "    distances_dict[key] = 0",
    "    queue.append(key)",
    "",
    "    t0 = time.perf_counter()",
    "    while queue:",
    "        key = queue.popleft()",
    "        d = distances_dict[key]",
    "        state = np.frombuffer(key, dtype=np.int8)",
    "",
    "        for move_idx in range(NUM_MOVES):",
    "            face_idx = move_idx // 3",
    "            turn_type = move_idx % 3",
    "            if turn_type == 0:",
    "                new_state = state[BASE_PERMUTATIONS[face_idx]]",
    "            elif turn_type == 1:",
    "                new_state = state[INV_PERMUTATIONS[face_idx]]",
    "            else:",
    "                p = BASE_PERMUTATIONS[face_idx]",
    "                new_state = state[p][p]",
    "",
    "            new_key = new_state.tobytes()",
    "            if new_key not in distances_dict:",
    "                distances_dict[new_key] = d + 1",
    "                queue.append(new_key)",
    "",
    "    bfs_seconds = time.perf_counter() - t0",
    "    if verbose:",
    "        from collections import Counter",
    "        counts = Counter(distances_dict.values())",
    "        for dist in sorted(counts):",
    "            print(f\"  distance {dist:2d}: {counts[dist]:>12,} states\")",
    "        print(f\"  total: {len(distances_dict):,} states  ({bfs_seconds:.1f}s)\")",
    "",
    "    if verbose:",
    '        print("  packing and sorting for on-disk format...", end=" ", flush=True)',
    "    t_pack = time.perf_counter()",
    "    n = len(distances_dict)",
    "    state_array = np.frombuffer(b\"\".join(distances_dict.keys()), dtype=np.int8).reshape(n, NUM_STICKERS)",
    "    values = np.fromiter(distances_dict.values(), dtype=np.int8, count=n)",
    "    packed = _pack_states_batch(state_array)",
    "    order = np.argsort(packed)",
    "    table = DistanceTable(packed[order], values[order])",
    "    if verbose:",
    "        print(f\"{time.perf_counter() - t_pack:.1f}s\")",
    "    return table",
    "",
    "",
    "# ---------------------------------------------------------------------------",
    "# Dataset generation",
    "# ---------------------------------------------------------------------------",
    "",
    "def generate_dataset(n_sequences, max_scramble_depth=11, rng=None, distances=None):",
    '    "Generate a dataset of (state, next_move) pairs from random scramble sequences."',
    "    if rng is None:",
    "        rng = np.random.default_rng(42)",
    "",
    "    all_states, all_moves, all_depths = [], [], []",
    "    all_face_solved, all_corner_oriented = [], []",
    "    all_optimal = [] if distances is not None else None",
    "",
    "    for _ in range(n_sequences):",
    "        depth = int(rng.integers(1, max_scramble_depth + 1))",
    "        cube = Cube()",
    "        moves = cube.scramble(depth, rng)",
    "",
    "        replay = Cube()",
    "        for step, move in enumerate(moves):",
    "            all_states.append(replay.encode())",
    "            all_moves.append(move)",
    "            all_depths.append(step)",
    "            all_face_solved.append(replay.face_solved())",
    "            all_corner_oriented.append(replay.corner_oriented())",
    "            if distances is not None:",
    "                all_optimal.append(distances[replay.state.tobytes()])",
    "            replay.apply_move(move)",
    "",
    "    result = {",
    '        "states":          np.stack(all_states).astype(np.float32),',
    '        "next_moves":      np.array(all_moves, dtype=np.int64),',
    '        "scramble_depth":  np.array(all_depths, dtype=np.int32),',
    '        "face_solved":     np.stack(all_face_solved),',
    '        "corner_oriented": np.stack(all_corner_oriented),',
    "    }",
    "    if distances is not None:",
    '        result["optimal_distance"] = np.array(all_optimal, dtype=np.int8)',
    "    return result",
    "",
    "",
    'print("Cube simulator loaded.")',
    "print(f\"  STATE_DIM={STATE_DIM}, NUM_MOVES={NUM_MOVES}\")",
    "print(f\"  24 cube rotational symmetries: {len(CUBE_ROTATION_PERMUTATIONS)}\")",
]
cells.append(new_code_cell("\n".join(_cube_cell_lines)))

# ─────────────────────────────────────────────────────────────────
# Cell 4 — CubeTransformer model (stripped, no HookPoints)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "class Attention(nn.Module):\n"
    "    def __init__(self, d_model, n_heads):\n"
    "        super().__init__()\n"
    "        assert d_model % n_heads == 0\n"
    "        self.n_heads = n_heads\n"
    "        self.d_head = d_model // n_heads\n"
    "        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)\n"
    "        self.out_proj = nn.Linear(d_model, d_model, bias=False)\n"
    "\n"
    "    def forward(self, x):\n"
    "        B, D = x.shape\n"
    "        q, k, v = self.qkv(x.unsqueeze(1)).chunk(3, dim=-1)\n"
    "        q = q.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)\n"
    "        k = k.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)\n"
    "        v = v.view(B, 1, self.n_heads, self.d_head).transpose(1, 2)\n"
    "        attn = F.softmax(q @ k.transpose(-2, -1) * self.d_head ** -0.5, dim=-1)\n"
    "        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, D))\n"
    "\n"
    "\n"
    "class MLP(nn.Module):\n"
    "    def __init__(self, d_model, mlp_mult=4):\n"
    "        super().__init__()\n"
    "        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)\n"
    "        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.fc2(F.gelu(self.fc1(x)))\n"
    "\n"
    "\n"
    "class TransformerBlock(nn.Module):\n"
    "    def __init__(self, d_model, n_heads, mlp_mult=4):\n"
    "        super().__init__()\n"
    "        self.ln1 = nn.LayerNorm(d_model)\n"
    "        self.attn = Attention(d_model, n_heads)\n"
    "        self.ln2 = nn.LayerNorm(d_model)\n"
    "        self.mlp = MLP(d_model, mlp_mult)\n"
    "\n"
    "    def forward(self, x):\n"
    "        x = x + self.attn(self.ln1(x))\n"
    "        x = x + self.mlp(self.ln2(x))\n"
    "        return x\n"
    "\n"
    "\n"
    "class CubeTransformer(nn.Module):\n"
    "    def __init__(self, d_model=128, n_layers=4, n_heads=4, mlp_mult=4, n_classes=12):\n"
    "        super().__init__()\n"
    "        self.embed = nn.Linear(STATE_DIM, d_model)\n"
    "        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_mult) for _ in range(n_layers)])\n"
    "        self.ln_final = nn.LayerNorm(d_model)\n"
    "        self.head = nn.Linear(d_model, n_classes, bias=False)\n"
    "\n"
    "    def forward(self, x):\n"
    "        x = self.embed(x)\n"
    "        for block in self.blocks:\n"
    "            x = block(x)\n"
    "        return self.head(self.ln_final(x))\n"
    "\n"
    "    @property\n"
    "    def n_params(self):\n"
    "        return sum(p.numel() for p in self.parameters())\n"
    "\n"
    "\n"
    "print(f\"CubeTransformer defined. Default param count: {CubeTransformer().n_params:,}\")\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 5 — Paths, plotting config, device
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'NOTEBOOK_DIR = Path(".")\n'
    "RESULTS_DIR = NOTEBOOK_DIR / \"results\"\n"
    "FIGURES_DIR = NOTEBOOK_DIR / \"figures\"\n"
    "RESULTS_DIR.mkdir(exist_ok=True)\n"
    "FIGURES_DIR.mkdir(exist_ok=True)\n"
    "\n"
    "plt.rcParams.update({\n"
    '    "font.family": "serif",\n'
    '    "font.size": 10,\n'
    '    "axes.labelsize": 10,\n'
    '    "axes.titlesize": 11,\n'
    '    "xtick.labelsize": 8.5,\n'
    '    "ytick.labelsize": 8.5,\n'
    '    "axes.spines.top": False,\n'
    '    "axes.spines.right": False,\n'
    '    "figure.dpi": 130,\n'
    "})\n"
    "\n"
    "\n"
    "def get_device():\n"
    "    if torch.cuda.is_available():\n"
    '        return torch.device("cuda")\n'
    "    if torch.backends.mps.is_available():\n"
    '        return torch.device("mps")\n'
    '    return torch.device("cpu")\n'
    "\n"
    "\n"
    "device = get_device()\n"
    'print(f"Device: {device}")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 6 — Dataset Generation markdown
# ─────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## Dataset Generation — BFS + Scrambling\n\n"
    "The dataset is built in two steps:\n\n"
    "1. **BFS distance table** (~25–30 min): A full BFS from the solved state over all\n"
    "   ~88 million reachable 2×2×2 states, recording the optimal HTM distance for each.\n"
    "   Saved as a compact `.npz` (sorted uint64 keys + int8 values) for fast lookup.\n\n"
    "2. **Scramble-based dataset** (~3 min): Random scrambles of depth 1–11 are generated.\n"
    "   For each step of each scramble, the current state and the next move are recorded,\n"
    "   along with the optimal distance from the BFS table.\n\n"
    "Both outputs are cached to `results/`; subsequent runs load instantly.\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 7 — BFS + dataset generation with caching
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'DISTANCES_CACHE = RESULTS_DIR / "distances.npz"\n'
    'TRAIN_CACHE = RESULTS_DIR / "train.npz"\n'
    'VAL_CACHE   = RESULTS_DIR / "val.npz"\n'
    "\n"
    "# Step 1: BFS distance table\n"
    "if DISTANCES_CACHE.exists():\n"
    '    print("Loading cached distance table...")\n'
    "    distances = DistanceTable.load(DISTANCES_CACHE)\n"
    "    print(f\"  {len(distances):,} states loaded\")\n"
    "else:\n"
    '    print("Computing BFS distance table (25-30 min)...")\n'
    "    distances = compute_optimal_distances(verbose=True)\n"
    "    distances.save(DISTANCES_CACHE)\n"
    "    print(f\"Saved to {DISTANCES_CACHE}\")\n"
    "\n"
    "\n"
    "# Step 2: Generate train/val splits\n"
    "def _gen_and_save(path, n_sequences, seed):\n"
    "    rng = np.random.default_rng(seed)\n"
    "    data = generate_dataset(n_sequences, max_scramble_depth=11, rng=rng, distances=distances)\n"
    "    np.savez_compressed(path, **data)\n"
    "    return data\n"
    "\n"
    "\n"
    "if TRAIN_CACHE.exists() and VAL_CACHE.exists():\n"
    '    print("Loading cached train/val splits...")\n'
    "    train_data = {k: v for k, v in np.load(TRAIN_CACHE).items()}\n"
    "    val_data   = {k: v for k, v in np.load(VAL_CACHE).items()}\n"
    "else:\n"
    '    print("Generating train split (50,000 sequences)...")\n'
    "    train_data = _gen_and_save(TRAIN_CACHE, 50_000, seed=0)\n"
    '    print("Generating val split (5,000 sequences)...")\n'
    "    val_data   = _gen_and_save(VAL_CACHE,   5_000,  seed=1)\n"
    "\n"
    "print(f\"Train: {len(train_data['states']):,}  Val: {len(val_data['states']):,}\")\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 8 — Data loaders and class weights
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "def make_loader(data, batch_size, shuffle):\n"
    "    ds = TensorDataset(\n"
    '        torch.from_numpy(data["states"].astype("float32")),\n'
    '        torch.from_numpy(data["optimal_distance"].astype("int64")),\n'
    "    )\n"
    "    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)\n"
    "\n"
    "\n"
    "BATCH_SIZE = 512\n"
    "train_loader = make_loader(train_data, BATCH_SIZE, shuffle=True)\n"
    "val_loader   = make_loader(val_data,   BATCH_SIZE, shuffle=False)\n"
    "\n"
    "# Inverse-frequency class weights (same as train.py)\n"
    'train_labels = torch.from_numpy(train_data["optimal_distance"].astype("int64"))\n'
    "counts = torch.bincount(train_labels, minlength=12).float()\n"
    "class_weights = (1.0 / (counts + 1))\n"
    "class_weights = class_weights / class_weights.sum() * 12\n"
    "\n"
    "print(f\"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}\")\n"
    "print(f\"Class weights: min={class_weights.min():.2f}  max={class_weights.max():.2f}\")\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 9 — evaluate() and run_experiment()
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "@torch.no_grad()\n"
    "def evaluate(model, loader, device, class_weights):\n"
    '    "Returns (val_loss, val_acc, depth_acc, depth_mae, depth_pred_errors)."\n'
    "    model.eval()\n"
    "    cw = class_weights.to(device)\n"
    "    total_loss = total_correct = total = 0\n"
    "    depth_preds = defaultdict(list)\n"
    "\n"
    "    for states, targets in loader:\n"
    "        states, targets = states.to(device), targets.to(device)\n"
    "        logits = model(states)\n"
    "        loss   = F.cross_entropy(logits, targets, weight=cw, reduction=\"sum\")\n"
    "        preds  = logits.argmax(dim=-1)\n"
    "\n"
    "        total_loss    += loss.item()\n"
    "        total_correct += (preds == targets).sum().item()\n"
    "        total         += len(targets)\n"
    "\n"
    "        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):\n"
    "            depth_preds[int(t)].append(int(p))\n"
    "\n"
    "    acc  = total_correct / total\n"
    "    loss = total_loss / total\n"
    "    depth_acc = {d: float(np.mean(np.array(depth_preds[d]) == d))   for d in sorted(depth_preds)}\n"
    "    depth_mae = {d: float(np.mean(np.abs(np.array(depth_preds[d]) - d))) for d in sorted(depth_preds)}\n"
    "    depth_errors = {d: (np.array(depth_preds[d]) - d).tolist() for d in sorted(depth_preds)}\n"
    "\n"
    "    return loss, acc, depth_acc, depth_mae, depth_errors\n"
    "\n"
    "\n"
    "def run_experiment(\n"
    "    name,\n"
    '    optimizer_type="adamw",\n'
    '    scheduler_type="cosine",\n'
    "    d_model=128,\n"
    "    n_layers=4,\n"
    "    n_heads=4,\n"
    "    lr=1e-3,\n"
    "    wd=1e-4,\n"
    "    epochs=30,\n"
    "    force=False,\n"
    "):\n"
    '    "Train a model and return history. Caches to results/{name}.json."\n'
    "    out_path = RESULTS_DIR / f\"{name}.json\"\n"
    "    if out_path.exists() and not force:\n"
    "        print(f\"  [cached]  {name}\")\n"
    "        return json.loads(out_path.read_text())\n"
    "\n"
    "    print(f\"  [running] {name}  ({optimizer_type}, {scheduler_type}, d={d_model}, lr={lr:.0e}, wd={wd:.0e})\")\n"
    "    model = CubeTransformer(\n"
    "        d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_classes=12\n"
    "    ).to(device)\n"
    "\n"
    '    if optimizer_type == "adamw":\n'
    "        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)\n"
    '    elif optimizer_type == "adam":\n'
    "        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)\n"
    '    elif optimizer_type == "sgd":\n'
    "        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown optimizer: {optimizer_type}\")\n"
    "\n"
    '    if scheduler_type == "cosine":\n'
    "        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)\n"
    '    elif scheduler_type == "constant":\n'
    "        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)\n"
    '    elif scheduler_type == "step":\n'
    "        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown scheduler: {scheduler_type}\")\n"
    "\n"
    "    cw = class_weights.to(device)\n"
    "    history = []\n"
    "\n"
    "    for epoch in range(1, epochs + 1):\n"
    "        model.train()\n"
    "        t0 = time.perf_counter()\n"
    "        running_loss = 0.0\n"
    "\n"
    "        for states, targets in train_loader:\n"
    "            states, targets = states.to(device), targets.to(device)\n"
    "            opt.zero_grad(set_to_none=True)\n"
    "            loss = F.cross_entropy(model(states), targets, weight=cw)\n"
    "            loss.backward()\n"
    "            opt.step()\n"
    "            running_loss += loss.item()\n"
    "\n"
    "        sched.step()\n"
    "        train_loss = running_loss / len(train_loader)\n"
    "        val_loss, val_acc, depth_acc, depth_mae, depth_errors = evaluate(\n"
    "            model, val_loader, device, class_weights\n"
    "        )\n"
    "        elapsed = time.perf_counter() - t0\n"
    "        current_lr = sched.get_last_lr()[0]\n"
    "\n"
    "        history.append({\n"
    '            "epoch": epoch,\n'
    '            "train_loss": train_loss,\n'
    '            "val_loss": val_loss,\n'
    '            "val_acc": val_acc,\n'
    '            "depth_acc": depth_acc,\n'
    '            "depth_mae": depth_mae,\n'
    '            "depth_errors": depth_errors,\n'
    '            "lr": current_lr,\n'
    '            "elapsed": elapsed,\n'
    "        })\n"
    "\n"
    "        if epoch % 10 == 0 or epoch == epochs:\n"
    "            print(f\"    epoch {epoch:3d}  val_acc={val_acc*100:.1f}%  loss={val_loss:.4f}  lr={current_lr:.2e}  ({elapsed:.1f}s)\")\n"
    "\n"
    "    out_path.write_text(json.dumps(history, indent=2))\n"
    "    print(f\"  -> saved {out_path}\")\n"
    "    return history\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 10 — Experiment 1: Optimizer comparison (run)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'print("=== 1. Optimizer Comparison ===")\n'
    "results_opt = {\n"
    '    "adamw": run_experiment("opt_adamw", optimizer_type="adamw", scheduler_type="cosine", lr=1e-3),\n'
    '    "adam":  run_experiment("opt_adam",  optimizer_type="adam",  scheduler_type="cosine", lr=1e-3),\n'
    '    "sgd":   run_experiment("opt_sgd",   optimizer_type="sgd",   scheduler_type="cosine", lr=0.05),\n'
    "}\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 11 — Experiment 1: figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n"
    'colors = {"adamw": "#1f77b4", "adam": "#ff7f0e", "sgd": "#2ca02c"}\n'
    'labels = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD + momentum"}\n'
    "\n"
    "for key, hist in results_opt.items():\n"
    '    epochs_x = [h["epoch"]       for h in hist]\n'
    '    val_acc  = [h["val_acc"]*100 for h in hist]\n'
    '    val_loss = [h["val_loss"]    for h in hist]\n'
    "    c = colors[key]\n"
    "    axes[0].plot(epochs_x, val_acc,  color=c, label=labels[key], linewidth=1.8)\n"
    "    axes[1].plot(epochs_x, val_loss, color=c, label=labels[key], linewidth=1.8)\n"
    "\n"
    "for ax, ylabel, title in zip(axes,\n"
    '    ["Validation Accuracy (%)", "Validation Loss"],\n'
    '    ["Accuracy by Optimizer",   "Loss by Optimizer"]):\n'
    '    ax.set_xlabel("Epoch")\n'
    "    ax.set_ylabel(ylabel)\n"
    "    ax.set_title(title)\n"
    "    ax.legend()\n"
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "optimizer_comparison.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved optimizer_comparison.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 12 — Experiment 2: Schedule ablation (run)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'print("=== 2. LR Schedule Ablation ===")\n'
    "results_sched = {\n"
    '    "cosine":   run_experiment("sched_cosine",   optimizer_type="adamw", scheduler_type="cosine",   lr=1e-3),\n'
    '    "step":     run_experiment("sched_step",     optimizer_type="adamw", scheduler_type="step",     lr=1e-3),\n'
    '    "constant": run_experiment("sched_constant", optimizer_type="adamw", scheduler_type="constant", lr=1e-3),\n'
    "}\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 13 — Experiment 2: figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))\n"
    'colors_s = {"cosine": "#1f77b4", "step": "#d62728", "constant": "#9467bd"}\n'
    'labels_s = {"cosine": "Cosine annealing", "step": "Step decay", "constant": "Constant"}\n'
    "\n"
    "for key, hist in results_sched.items():\n"
    '    epochs_x = [h["epoch"]       for h in hist]\n'
    '    val_acc  = [h["val_acc"]*100 for h in hist]\n'
    '    val_loss = [h["val_loss"]    for h in hist]\n'
    '    lrs      = [h["lr"]          for h in hist]\n'
    "    c = colors_s[key]\n"
    "    axes[0].plot(epochs_x, val_acc,  color=c, label=labels_s[key], linewidth=1.8)\n"
    "    axes[1].plot(epochs_x, val_loss, color=c, label=labels_s[key], linewidth=1.8)\n"
    "    axes[2].plot(epochs_x, lrs,      color=c, label=labels_s[key], linewidth=1.8)\n"
    "\n"
    "for ax, ylabel, title in zip(axes,\n"
    '    ["Val Accuracy (%)", "Val Loss", "Learning Rate"],\n'
    '    ["Accuracy",         "Loss",     "LR Trajectory"]):\n'
    '    ax.set_xlabel("Epoch")\n'
    "    ax.set_ylabel(ylabel)\n"
    "    ax.set_title(title)\n"
    "    ax.legend(fontsize=8)\n"
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "schedule_comparison.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved schedule_comparison.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 14 — Training Dynamics markdown
# ─────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## Training Dynamics — Multi-Seed Comparison\n\n"
    "Train each of AdamW, Adam, and SGD+momentum for 3 independent seeds (0, 1, 2)\n"
    "to assess variance across initializations. Results are cached as\n"
    "`results/multi_seed_{opt}_s{seed}.json`.\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 15 — Multi-seed training
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "SEEDS = [0, 1, 2]\n"
    "\n"
    "optimizer_ms_configs = {\n"
    '    "adamw": dict(optimizer_type="adamw", scheduler_type="cosine", lr=1e-3),\n'
    '    "adam":  dict(optimizer_type="adam",  scheduler_type="cosine", lr=1e-3),\n'
    '    "sgd":   dict(optimizer_type="sgd",   scheduler_type="cosine", lr=0.05),\n'
    "}\n"
    "\n"
    "\n"
    "def run_multiseed_one(name, seed, optimizer_type, scheduler_type, lr, wd=1e-4, epochs=30):\n"
    '    "Train one model for one seed, caching to results/multi_seed_{name}_s{seed}.json."\n'
    "    out = RESULTS_DIR / f\"multi_seed_{name}_s{seed}.json\"\n"
    "    if out.exists():\n"
    "        print(f\"  [cached] {name} seed={seed}\")\n"
    "        return json.loads(out.read_text())\n"
    "\n"
    "    print(f\"  [run]    {name} seed={seed}  ({optimizer_type}, {scheduler_type}, lr={lr:.0e})\")\n"
    "    torch.manual_seed(seed)\n"
    "\n"
    "    model = CubeTransformer(d_model=128, n_layers=4, n_heads=4, n_classes=12).to(device)\n"
    "\n"
    '    if optimizer_type == "adamw":\n'
    "        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)\n"
    '    elif optimizer_type == "adam":\n'
    "        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)\n"
    "    else:  # sgd\n"
    "        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)\n"
    "\n"
    '    if scheduler_type == "cosine":\n'
    "        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)\n"
    '    elif scheduler_type == "step":\n'
    "        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)\n"
    "    else:\n"
    "        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0)\n"
    "\n"
    "    cw = class_weights.to(device)\n"
    "    history = []\n"
    "\n"
    "    for epoch in range(1, epochs + 1):\n"
    "        model.train()\n"
    "        for states_b, targets_b in train_loader:\n"
    "            states_b, targets_b = states_b.to(device), targets_b.to(device)\n"
    "            opt.zero_grad(set_to_none=True)\n"
    "            F.cross_entropy(model(states_b), targets_b, weight=cw).backward()\n"
    "            opt.step()\n"
    "        sched.step()\n"
    "\n"
    "        model.eval()\n"
    "        correct = total = 0\n"
    "        with torch.no_grad():\n"
    "            for states_b, targets_b in val_loader:\n"
    "                states_b, targets_b = states_b.to(device), targets_b.to(device)\n"
    "                correct += (model(states_b).argmax(1) == targets_b).sum().item()\n"
    "                total   += len(targets_b)\n"
    "        val_acc = correct / total\n"
    "        history.append({\"epoch\": epoch, \"val_acc\": val_acc})\n"
    "\n"
    "        if epoch % 10 == 0 or epoch == epochs:\n"
    "            print(f\"    epoch {epoch:3d}  val_acc={val_acc*100:.1f}%\")\n"
    "\n"
    "    out.write_text(json.dumps(history))\n"
    "    return history\n"
    "\n"
    "\n"
    'print("=== Multi-seed training ===")\n'
    "results_multiseed = {}\n"
    "for opt_name, cfg in optimizer_ms_configs.items():\n"
    "    for seed in SEEDS:\n"
    "        hist = run_multiseed_one(opt_name, seed, **cfg)\n"
    "        results_multiseed[(opt_name, seed)] = hist\n"
    "\n"
    'print("Done. results_multiseed keys:", list(results_multiseed.keys()))\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 16 — Training dynamics figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'COLORS_MS    = {"adamw": "#1f77b4", "adam": "#ff7f0e", "sgd": "#2ca02c"}\n'
    'LABELS_MS    = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD + momentum"}\n'
    'LINESTYLE_MS = {"adamw": "-", "adam": "--", "sgd": ":"}\n'
    "\n"
    "fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))\n"
    "\n"
    "epochs_arr = np.arange(1, 31)\n"
    'for opt_name in ["adamw", "adam", "sgd"]:\n'
    "    accs = []\n"
    "    for seed in SEEDS:\n"
    "        hist = results_multiseed[(opt_name, seed)]\n"
    '        accs.append([h["val_acc"] * 100 for h in hist])\n'
    "    accs = np.array(accs)  # shape (3, 30)\n"
    "    mean = accs.mean(axis=0)\n"
    "    std  = accs.std(axis=0)\n"
    "    c = COLORS_MS[opt_name]\n"
    "    ax.plot(epochs_arr, mean, color=c, label=LABELS_MS[opt_name],\n"
    "            linestyle=LINESTYLE_MS[opt_name], linewidth=1.8)\n"
    "    ax.fill_between(epochs_arr, mean - std, mean + std, color=c, alpha=0.15)\n"
    "\n"
    'ax.set_xlabel("Epoch")\n'
    'ax.set_ylabel("Validation accuracy (%)")\n'
    "ax.set_xlim(1, 30)\n"
    "ax.set_ylim(50, 83)\n"
    "ax.set_xticks([1, 5, 10, 15, 20, 25, 30])\n"
    'ax.legend(loc="lower right")\n'
    "ax.grid(True, linewidth=0.4, alpha=0.5)\n"
    'ax.set_title("Per-epoch validation accuracy (mean +/- std, 3 seeds)")\n'
    "\n"
    'ax.annotate("smooth, monotone\\nconvergence - no phase transition",\n'
    "            xy=(20, 76), xytext=(14, 60),\n"
    '            fontsize=7.5, color="#444444",\n'
    '            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))\n'
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "training_dynamics.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved training_dynamics.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 17 — Experiment 3: Hyperparameter sensitivity (run)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'print("=== 3. Hyperparameter Sensitivity ===")\n'
    "\n"
    "# Width sweep\n"
    "d_models = [64, 128, 256]\n"
    "results_dmodel = {}\n"
    "for d in d_models:\n"
    "    results_dmodel[d] = run_experiment(\n"
    '        f"dmodel_{d}", optimizer_type="adamw", scheduler_type="cosine",\n'
    "        d_model=d, lr=1e-3, wd=1e-4,\n"
    "    )\n"
    "\n"
    "# Weight decay sweep\n"
    "wd_values = [0.0, 1e-4, 1e-3, 1e-2]\n"
    "results_wd = {}\n"
    "for wd in wd_values:\n"
    '    name = f"wd_{wd:.0e}" if wd > 0 else "wd_0"\n'
    "    results_wd[wd] = run_experiment(\n"
    '        name, optimizer_type="adamw", scheduler_type="cosine",\n'
    "        d_model=128, lr=1e-3, wd=wd,\n"
    "    )\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 18 — Experiment 3: figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))\n"
    "\n"
    "# Width sweep: convergence curves\n"
    "cmap_d = plt.cm.Blues\n"
    "for i, (d, hist) in enumerate(sorted(results_dmodel.items())):\n"
    "    c = cmap_d(0.4 + 0.3 * i)\n"
    '    epochs_x = [h["epoch"]       for h in hist]\n'
    '    val_acc  = [h["val_acc"]*100 for h in hist]\n'
    "    axes[0].plot(epochs_x, val_acc, color=c, label=f\"d={d}\", linewidth=1.8)\n"
    'axes[0].set_xlabel("Epoch")\n'
    'axes[0].set_ylabel("Val Accuracy (%)")\n'
    'axes[0].set_title("Width Sensitivity")\n'
    "axes[0].legend()\n"
    "\n"
    "# Weight decay sweep: convergence curves\n"
    "cmap_w = plt.cm.Reds\n"
    'wd_labels = {0.0: "0 (none)", 1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2"}\n'
    "for i, (wd, hist) in enumerate(sorted(results_wd.items())):\n"
    "    c = cmap_w(0.3 + 0.18 * i)\n"
    '    epochs_x = [h["epoch"]       for h in hist]\n'
    '    val_acc  = [h["val_acc"]*100 for h in hist]\n'
    "    axes[1].plot(epochs_x, val_acc, color=c, label=f\"lambda={wd_labels[wd]}\", linewidth=1.8)\n"
    'axes[1].set_xlabel("Epoch")\n'
    'axes[1].set_ylabel("Val Accuracy (%)")\n'
    'axes[1].set_title("Weight Decay Sensitivity")\n'
    "axes[1].legend(fontsize=8)\n"
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "hyperparameter_sensitivity.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved hyperparameter_sensitivity.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 19 — Experiment 4: Heuristic quality (retrain best model)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    'print("=== 4. Learned Heuristic Quality ===")\n'
    "\n"
    "# Retrain the best model with fixed seed=0 for determinism\n"
    "torch.manual_seed(0)\n"
    "best_model = CubeTransformer(d_model=128, n_layers=4, n_heads=4, n_classes=12).to(device)\n"
    "best_opt   = torch.optim.AdamW(best_model.parameters(), lr=1e-3, weight_decay=1e-4)\n"
    "best_sched = torch.optim.lr_scheduler.CosineAnnealingLR(best_opt, T_max=30, eta_min=1e-3 / 20)\n"
    "cw = class_weights.to(device)\n"
    "\n"
    "for epoch in range(1, 31):\n"
    "    best_model.train()\n"
    "    for states_b, targets_b in train_loader:\n"
    "        states_b, targets_b = states_b.to(device), targets_b.to(device)\n"
    "        best_opt.zero_grad(set_to_none=True)\n"
    "        F.cross_entropy(best_model(states_b), targets_b, weight=cw).backward()\n"
    "        best_opt.step()\n"
    "    best_sched.step()\n"
    "    if epoch % 10 == 0:\n"
    "        _, acc, _, _, _ = evaluate(best_model, val_loader, device, class_weights)\n"
    "        print(f\"  epoch {epoch}  val_acc={acc*100:.1f}%\")\n"
    "\n"
    "_, final_acc, depth_acc, depth_mae, depth_errors = evaluate(\n"
    "    best_model, val_loader, device, class_weights\n"
    ")\n"
    "print(f\"\\nFinal val accuracy: {final_acc*100:.1f}%\")\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 20 — Heuristic quality figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "distances_list = sorted(depth_acc.keys())\n"
    "\n"
    "all_errors = [v for errs in depth_errors.values() for v in errs]\n"
    "overestimate_pct = 100 * np.mean(np.array(all_errors) > 0)\n"
    "print(f\"Overall overestimate rate: {overestimate_pct:.1f}%  ({'NOT ' if overestimate_pct > 0 else ''}admissible)\")\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))\n"
    "\n"
    "# Per-distance accuracy\n"
    "axes[0].bar(distances_list, [depth_acc[d]*100 for d in distances_list], color=\"#1f77b4\", alpha=0.85)\n"
    'axes[0].axhline(100/12, color="gray", linestyle="--", linewidth=1, label="Random (8.3%)")\n'
    'axes[0].set_xlabel("True Distance")\n'
    'axes[0].set_ylabel("Accuracy (%)")\n'
    'axes[0].set_title("Per-Distance Accuracy")\n'
    "axes[0].legend(fontsize=8)\n"
    "\n"
    "# Per-distance MAE\n"
    "axes[1].bar(distances_list, [depth_mae[d] for d in distances_list], color=\"#ff7f0e\", alpha=0.85)\n"
    'axes[1].set_xlabel("True Distance")\n'
    'axes[1].set_ylabel("Mean Absolute Error")\n'
    'axes[1].set_title("Heuristic MAE by Distance")\n'
    "\n"
    "# Error distribution\n"
    "error_vals = np.array(all_errors)\n"
    "bins = np.arange(error_vals.min() - 0.5, error_vals.max() + 1.5)\n"
    "axes[2].hist(error_vals, bins=bins, color=\"#2ca02c\", alpha=0.8, edgecolor=\"white\")\n"
    'axes[2].axvline(0, color="black", linewidth=1.2, linestyle="--")\n'
    'axes[2].set_xlabel("Prediction Error (h_hat - d*)")\n'
    'axes[2].set_ylabel("Count")\n'
    'axes[2].set_title("Error Distribution")\n'
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "heuristic_quality.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved heuristic_quality.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 21 — Per-distance overestimation figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "# Use depth_errors from best_model evaluation (cell 19) directly\n"
    "dist_range = list(range(11))\n"
    "n_over_list    = []\n"
    "n_correct_list = []\n"
    "n_under_list   = []\n"
    "totals_list    = []\n"
    "\n"
    "for d in dist_range:\n"
    "    errs = np.array(depth_errors.get(d, []))\n"
    "    n = len(errs)\n"
    "    totals_list.append(n)\n"
    "    if n == 0:\n"
    "        n_over_list.append(0); n_correct_list.append(0); n_under_list.append(0)\n"
    "    else:\n"
    "        n_over_list.append(int((errs > 0).sum()))\n"
    "        n_correct_list.append(int((errs == 0).sum()))\n"
    "        n_under_list.append(int((errs < 0).sum()))\n"
    "\n"
    "pct_over    = [100 * o / t if t else 0 for o, t in zip(n_over_list,    totals_list)]\n"
    "pct_correct = [100 * c / t if t else 0 for c, t in zip(n_correct_list, totals_list)]\n"
    "pct_under   = [100 * u / t if t else 0 for u, t in zip(n_under_list,   totals_list)]\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))\n"
    "\n"
    "# Left: stacked bar (admissibility breakdown)\n"
    "ax = axes[0]\n"
    "x = np.array(dist_range)\n"
    'ax.bar(x, pct_correct, color="#4ade80", label="Correct")\n'
    'ax.bar(x, pct_over,    bottom=pct_correct, color="#f87171", label="Overestimate (inadmissible)")\n'
    "ax.bar(x, pct_under,\n"
    "       bottom=[c + o for c, o in zip(pct_correct, pct_over)],\n"
    '       color="#93c5fd", label="Underestimate (admissible)")\n'
    "\n"
    'ax.set_xlabel("True distance d*")\n'
    'ax.set_ylabel("Fraction of validation examples (%)")\n'
    "ax.set_xticks(dist_range)\n"
    "ax.set_ylim(0, 105)\n"
    'ax.legend(loc="upper right", fontsize=7.5)\n'
    "ax.grid(axis=\"y\", linewidth=0.4, alpha=0.5)\n"
    'ax.set_title("Admissibility breakdown by distance class")\n'
    "\n"
    "if len(pct_over) > 5 and pct_over[5] > 0:\n"
    '    ax.annotate("inadmissible\\nzone (d=4-7)",\n'
    "                xy=(5, pct_over[5] / 2 + pct_correct[5]),\n"
    "                xytext=(7.2, 70),\n"
    '                fontsize=7.5, color="#b91c1c",\n'
    '                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9))\n'
    "\n"
    "# Right: overestimation rate bar chart\n"
    "ax2 = axes[1]\n"
    'ax2.bar(x, pct_over, color="#f87171", alpha=0.85, label="Overestimation rate")\n'
    "overall_rate = 100 * sum(n_over_list) / sum(totals_list) if sum(totals_list) > 0 else 0\n"
    'ax2.axhline(overall_rate, color="#991b1b", linestyle="--", linewidth=1.2,\n'
    '            label=f"Overall rate ({overall_rate:.1f}%)")\n'
    'ax2.set_xlabel("True distance d*")\n'
    'ax2.set_ylabel("Overestimation rate (%)")\n'
    "ax2.set_xticks(dist_range)\n"
    "ax2.set_ylim(0, 65)\n"
    "ax2.legend(fontsize=7.5)\n"
    'ax2.grid(axis="y", linewidth=0.4, alpha=0.5)\n'
    'ax2.set_title("Overestimation rate by distance (inadmissibility profile)")\n'
    "\n"
    'ax2.annotate("admissible\\n(d=8-10)", xy=(9, 2), xytext=(7.5, 20),\n'
    '             fontsize=7.5, color="#166534",\n'
    '             arrowprops=dict(arrowstyle="->", color="#166534", lw=0.9))\n'
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "overestimation.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved overestimation.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 22 — Loss landscape computation
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "def filter_normalize(direction, reference):\n"
    '    "Scale each filter in direction to match the Frobenius norm of reference."\n'
    "    out = []\n"
    "    for d, w in zip(direction, reference):\n"
    "        if w.dim() > 1:\n"
    "            w_norms = w.view(w.shape[0], -1).norm(dim=1, keepdim=True)\n"
    "            d_norms = d.view(d.shape[0], -1).norm(dim=1, keepdim=True) + 1e-10\n"
    "            scale = (w_norms / d_norms).view(-1, *([1] * (w.dim() - 1)))\n"
    "            out.append(d * scale)\n"
    "        else:\n"
    "            out.append(d * (w.norm() / (d.norm() + 1e-10)))\n"
    "    return out\n"
    "\n"
    "\n"
    "def compute_loss_landscape(model, loader, device, class_weights,\n"
    "                           grid_size=21, alpha_range=1.0, seed=42):\n"
    '    "Evaluate loss on a 2D perturbation grid around model weights."\n'
    "    weights = [p.data.clone() for p in model.parameters()]\n"
    "    cw = class_weights.to(device)\n"
    "\n"
    "    torch.manual_seed(seed)\n"
    "    dir1 = filter_normalize([torch.randn_like(w) for w in weights], weights)\n"
    "    dir2 = filter_normalize([torch.randn_like(w) for w in weights], weights)\n"
    "\n"
    "    alphas = np.linspace(-alpha_range, alpha_range, grid_size)\n"
    "    betas  = np.linspace(-alpha_range, alpha_range, grid_size)\n"
    "    loss_grid = np.zeros((grid_size, grid_size))\n"
    "\n"
    "    for i, alpha in enumerate(alphas):\n"
    "        for j, beta in enumerate(betas):\n"
    "            for p, w, d1, d2 in zip(model.parameters(), weights, dir1, dir2):\n"
    "                p.data.copy_(w + alpha * d1 + beta * d2)\n"
    "            total_loss = total = 0\n"
    "            with torch.no_grad():\n"
    "                for states_b, targets_b in loader:\n"
    "                    states_b, targets_b = states_b.to(device), targets_b.to(device)\n"
    "                    loss = F.cross_entropy(model(states_b), targets_b, weight=cw, reduction=\"sum\")\n"
    "                    total_loss += loss.item()\n"
    "                    total      += len(targets_b)\n"
    "            loss_grid[i, j] = total_loss / total\n"
    "        if (i + 1) % 5 == 0:\n"
    "            print(f\"  Row {i+1}/{grid_size} done\")\n"
    "\n"
    "    # Restore original weights\n"
    "    for p, w in zip(model.parameters(), weights):\n"
    "        p.data.copy_(w)\n"
    "\n"
    "    return alphas, betas, loss_grid\n"
    "\n"
    "\n"
    'print("Computing loss landscape (this takes ~5-10 min)...")\n'
    "alphas, betas, loss_grid = compute_loss_landscape(\n"
    "    best_model, val_loader, device, class_weights,\n"
    "    grid_size=21, alpha_range=1.0,\n"
    ")\n"
    'np.savez(RESULTS_DIR / "loss_landscape.npz",\n'
    "         alphas=alphas, betas=betas, loss_grid=loss_grid)\n"
    'print("Done. Saved loss_landscape.npz")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 23 — Loss landscape figure (use in-memory variables)
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "# Use alphas, betas, loss_grid from cell 22 (no file reload needed)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
    "\n"
    "# Contour plot\n"
    'A, B = np.meshgrid(alphas, betas, indexing="ij")\n'
    "center_loss = loss_grid[len(alphas)//2, len(betas)//2]\n"
    "levels = np.linspace(loss_grid.min(), min(loss_grid.max(), center_loss * 5), 20)\n"
    'cf = axes[0].contourf(A, B, loss_grid, levels=levels, cmap="viridis")\n'
    'axes[0].contour(A, B, loss_grid, levels=levels, colors="white", linewidths=0.4, alpha=0.4)\n'
    'axes[0].scatter([0], [0], color="red", s=60, zorder=5, label="theta*")\n'
    'fig.colorbar(cf, ax=axes[0], label="Val Loss")\n'
    'axes[0].set_xlabel("alpha (direction 1)")\n'
    'axes[0].set_ylabel("beta (direction 2)")\n'
    'axes[0].set_title("Loss Landscape (contour)")\n'
    "axes[0].legend(fontsize=9)\n"
    "\n"
    "# 1D slices through center\n"
    "mid = len(alphas) // 2\n"
    'axes[1].plot(alphas, loss_grid[:, mid], color="#1f77b4", label="direction 1 (beta=0)", linewidth=1.8)\n'
    'axes[1].plot(betas,  loss_grid[mid, :], color="#ff7f0e", label="direction 2 (alpha=0)", linewidth=1.8)\n'
    'axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)\n'
    'axes[1].set_xlabel("Perturbation magnitude")\n'
    'axes[1].set_ylabel("Val Loss")\n'
    'axes[1].set_title("1D Slices Through theta*")\n'
    "axes[1].legend()\n"
    "\n"
    'fig.suptitle("Filter-Normalized Loss Landscape", fontsize=12, y=1.02)\n'
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "loss_landscape.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved loss_landscape.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 24 — A* Search markdown
# ─────────────────────────────────────────────────────────────────
cells.append(new_markdown_cell(
    "## A* Search with Learned Heuristic\n\n"
    "Deploy the trained model as a search heuristic: f(s) = g(s) + h(s) where\n"
    "h(s) = model-predicted distance. Compare against Dijkstra (h=0, i.e. BFS)\n"
    "as a baseline.\n\n"
    "Metrics: nodes expanded (search efficiency) and solution optimality\n"
    "(solution length == BFS optimal distance).\n\n"
    "Node limit: 10,000 per search.\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 25 — A* search setup + run
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "MAX_NODES = 10_000\n"
    "N_PER_DIST = 20   # test cases per distance\n"
    "\n"
    "\n"
    "def _state_1hot_to_int8(state_1hot):\n"
    '    "Convert (144,) float32 one-hot -> (24,) int8 color indices."\n'
    "    return np.argmax(state_1hot.reshape(24, 6), axis=1).astype(np.int8)\n"
    "\n"
    "\n"
    "def _encode_int8_to_1hot(state_int8):\n"
    '    "Convert (24,) int8 -> (144,) float32 one-hot."\n'
    "    out = np.zeros(144, dtype=np.float32)\n"
    "    for i, c in enumerate(state_int8):\n"
    "        out[i * 6 + int(c)] = 1.0\n"
    "    return out\n"
    "\n"
    "\n"
    "@torch.no_grad()\n"
    "def _batch_h(states_list, model, device):\n"
    '    "Predict distance for a list of (24,) int8 states. Returns list[int]."\n'
    "    vecs = np.stack([_encode_int8_to_1hot(s) for s in states_list])\n"
    "    t = torch.from_numpy(vecs).to(device)\n"
    "    return model(t).argmax(dim=-1).cpu().numpy().tolist()\n"
    "\n"
    "\n"
    "def astar_search(start_int8, model, device, max_nodes=MAX_NODES, use_heuristic=True):\n"
    '    "A* (or Dijkstra if use_heuristic=False) from start_int8 to SOLVED_STATE."\n'
    "    if np.array_equal(start_int8, SOLVED_STATE):\n"
    "        return 0, 0\n"
    "\n"
    "    h0 = _batch_h([start_int8], model, device)[0] if use_heuristic else 0\n"
    "    counter = 0\n"
    "    heap = [(h0, 0, counter, start_int8.tobytes())]\n"
    "    best_g = {pack_state(start_int8): 0}\n"
    "    nodes_expanded = 0\n"
    "\n"
    "    while heap and nodes_expanded < max_nodes:\n"
    "        _, g, _, state_bytes = heapq.heappop(heap)\n"
    "        state = np.frombuffer(state_bytes, dtype=np.int8).copy()\n"
    "        key = pack_state(state)\n"
    "\n"
    "        if g > best_g.get(key, float(\"inf\")):\n"
    "            continue\n"
    "        nodes_expanded += 1\n"
    "\n"
    "        succs, keys, gs = [], [], []\n"
    "        for m in range(NUM_MOVES):\n"
    "            ns = Cube(state.copy()).apply_move(m).state\n"
    "            ng = g + 1\n"
    "            nk = pack_state(ns)\n"
    "            if ng < best_g.get(nk, float(\"inf\")):\n"
    "                best_g[nk] = ng\n"
    "                if np.array_equal(ns, SOLVED_STATE):\n"
    "                    return ng, nodes_expanded\n"
    "                succs.append(ns); keys.append(nk); gs.append(ng)\n"
    "\n"
    "        if not succs:\n"
    "            continue\n"
    "\n"
    "        h_vals = _batch_h(succs, model, device) if use_heuristic else [0] * len(succs)\n"
    "        for ns, nk, ng, h in zip(succs, keys, gs, h_vals):\n"
    "            counter += 1\n"
    "            heapq.heappush(heap, (ng + h, ng, counter, ns.tobytes()))\n"
    "\n"
    "    return None, nodes_expanded\n"
    "\n"
    "\n"
    'astar_out = RESULTS_DIR / "astar_results.json"\n'
    "\n"
    "if astar_out.exists():\n"
    '    print("[cached] astar_results.json")\n'
    "    astar_data = json.loads(astar_out.read_text())\n"
    "else:\n"
    '    val_states = val_data["states"]\n'
    '    val_dists  = val_data["optimal_distance"].astype(int)\n'
    "\n"
    "    astar_data = {}\n"
    "    for d in range(1, 11):\n"
    "        idxs = np.where(val_dists == d)[0]\n"
    "        if len(idxs) == 0:\n"
    "            continue\n"
    "        rng = np.random.default_rng(seed=d)\n"
    "        sampled = rng.choice(idxs, size=min(N_PER_DIST, len(idxs)), replace=False)\n"
    "\n"
    "        a_res, dijk_res = [], []\n"
    "        for idx in sampled:\n"
    "            s = _state_1hot_to_int8(val_states[idx])\n"
    "\n"
    "            sl, ne = astar_search(s, best_model, device, use_heuristic=True)\n"
    "            a_res.append({\"sol_len\": sl, \"nodes\": ne,\n"
    "                          \"optimal\": bool(sl == d) if sl is not None else None,\n"
    "                          \"success\": sl is not None})\n"
    "\n"
    "            sl2, ne2 = astar_search(s, best_model, device, use_heuristic=False)\n"
    "            dijk_res.append({\"sol_len\": sl2, \"nodes\": ne2,\n"
    "                             \"optimal\": bool(sl2 == d) if sl2 is not None else None,\n"
    "                             \"success\": sl2 is not None})\n"
    "\n"
    "        a_succ = sum(r[\"success\"] for r in a_res)\n"
    "        d_succ = sum(r[\"success\"] for r in dijk_res)\n"
    "        a_med  = int(np.median([r[\"nodes\"] for r in a_res]))\n"
    "        print(f\"  d={d:2d}: A* {a_succ}/{len(sampled)} success, median nodes={a_med:5d} | \"\n"
    "              f\"Dijkstra {d_succ}/{len(sampled)} success\")\n"
    "\n"
    "        astar_data[str(d)] = {\"d\": d, \"n\": len(sampled),\n"
    "                               \"astar\": a_res, \"dijkstra\": dijk_res}\n"
    "\n"
    "    astar_out.write_text(json.dumps(astar_data, indent=2))\n"
    "    print(f\"Saved {astar_out}\")\n"
    "\n"
    "print(f\"A* done. MAX_NODES={MAX_NODES:,}  N_PER_DIST={N_PER_DIST}\")\n"
))

# ─────────────────────────────────────────────────────────────────
# Cell 26 — A* figure
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "distances_astar = sorted(int(d) for d in astar_data.keys())\n"
    "\n"
    "a_nodes_med, a_nodes_q1, a_nodes_q3 = [], [], []\n"
    "d_nodes_med = []\n"
    "a_success, d_success, a_optimal = [], [], []\n"
    "\n"
    "for d in distances_astar:\n"
    "    r = astar_data[str(d)]\n"
    '    a_n = [x["nodes"] for x in r["astar"]]\n'
    '    d_n = [x["nodes"] for x in r["dijkstra"]]\n'
    "\n"
    "    a_nodes_med.append(float(np.median(a_n)))\n"
    "    a_nodes_q1.append(float(np.percentile(a_n, 25)))\n"
    "    a_nodes_q3.append(float(np.percentile(a_n, 75)))\n"
    "    d_nodes_med.append(float(np.median(d_n)))\n"
    "\n"
    '    a_success.append(100 * np.mean([x["success"] for x in r["astar"]]))\n'
    '    d_success.append(100 * np.mean([x["success"] for x in r["dijkstra"]]))\n'
    '    opt = [x["optimal"] for x in r["astar"] if x["success"]]\n'
    "    a_optimal.append(100 * np.mean(opt) if opt else 0.0)\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
    "\n"
    "ax = axes[0]\n"
    'ax.semilogy(distances_astar, a_nodes_med, "o-",  color="#1f77b4", linewidth=1.8, label="A* (learned h)")\n'
    "ax.fill_between(distances_astar, a_nodes_q1, a_nodes_q3, color=\"#1f77b4\", alpha=0.18)\n"
    'ax.semilogy(distances_astar, d_nodes_med,   "s--", color="#d62728", linewidth=1.8, label="Dijkstra (h=0)")\n'
    "ax.axhline(MAX_NODES, color=\"gray\", linestyle=\":\", linewidth=1.2,\n"
    "           label=f\"Node limit ({MAX_NODES:,})\")\n"
    'ax.set_xlabel("True Optimal Distance")\n'
    'ax.set_ylabel("Nodes Expanded (median +/- IQR, log)")\n'
    'ax.set_title("Search Efficiency")\n'
    "ax.set_xticks(distances_astar)\n"
    "ax.legend(fontsize=8)\n"
    "\n"
    "ax2 = axes[1]\n"
    "w = 0.28\n"
    "xs = np.array(distances_astar)\n"
    'ax2.bar(xs - w, a_optimal,  width=w, color="#2ca02c", alpha=0.85, label="Optimal rate (A*)")\n'
    'ax2.bar(xs,     a_success,  width=w, color="#1f77b4", alpha=0.75, label="Success rate (A*)")\n'
    'ax2.bar(xs + w, d_success,  width=w, color="#d62728", alpha=0.65, label="Success rate (Dijkstra)")\n'
    'ax2.set_xlabel("True Optimal Distance")\n'
    'ax2.set_ylabel("Rate (%)")\n'
    'ax2.set_title("Solution Quality and Search Success")\n'
    "ax2.set_xticks(distances_astar)\n"
    "ax2.set_ylim(0, 108)\n"
    "ax2.legend(fontsize=8)\n"
    "\n"
    "fig.tight_layout()\n"
    'fig.savefig(FIGURES_DIR / "astar_results.pdf", bbox_inches="tight")\n'
    "plt.show()\n"
    'print("Saved astar_results.pdf")\n'
))

# ─────────────────────────────────────────────────────────────────
# Cell 27 — Summary table
# ─────────────────────────────────────────────────────────────────
cells.append(new_code_cell(
    "print(f\"{'Experiment':<35} {'Final Val Acc':>14} {'Best Val Acc':>13}\")\n"
    'print("-" * 64)\n'
    "\n"
    "all_results = {\n"
    '    "AdamW (cosine)":    results_opt["adamw"],\n'
    '    "Adam (cosine)":     results_opt["adam"],\n'
    '    "SGD+mom (cosine)":  results_opt["sgd"],\n'
    '    "AdamW (step)":      results_sched["step"],\n'
    '    "AdamW (constant)":  results_sched["constant"],\n'
    "    **{f\"d_model={d}\": h for d, h in results_dmodel.items() if d != 128},\n"
    "    **{f\"wd={wd:.0e}\": h for wd, h in results_wd.items()},\n"
    "}\n"
    "\n"
    "for name, hist in all_results.items():\n"
    '    final = hist[-1]["val_acc"] * 100\n'
    '    best  = max(h["val_acc"] for h in hist) * 100\n'
    '    print(f"{name:<35} {final:>13.1f}%  {best:>12.1f}%")\n'
))

# ─────────────────────────────────────────────────────────────────
# Assemble and write notebook
# ─────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}

from pathlib import Path
out_path = Path(__file__).parent / "optimization_experiments.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

n_code = sum(1 for c in cells if c.cell_type == "code")
n_md   = sum(1 for c in cells if c.cell_type == "markdown")
print(f"Wrote {out_path}")
print(f"  {len(cells)} cells ({n_code} code, {n_md} markdown)")
