"""
2x2x2 Rubik's Cube simulator with one-hot state encoding.

Sticker layout — 24 stickers, indices 0..23:
  Face order (internal): U(top), D(bottom), F(front), B(back), L(left), R(right)
  Within each face, stickers are numbered in reading order (top-left → top-right → bottom-left → bottom-right):

        U face
       [0][1]
       [2][3]

  L face  F face  B face  R face
 [16][17] [8][9] [12][13] [20][21]
 [18][19][10][11][14][15] [22][23]

        D face
       [4][5]
       [6][7]

  Internal face→sticker assignment: U=0-3, D=4-7, F=8-11, B=12-15, L=16-19, R=20-23
  (Note: the unfolded net visualizer uses a different left-to-right order for the
   middle row; the internal ordering above is what the code uses.)

Color-to-index mapping:
  W=0 (white/U), Y=1 (yellow/D), G=4 (green/F),
  B=5 (blue/B),  O=2 (orange/L), R=3 (red/R)

Move vocabulary (18 moves = 6 faces × 3 quarter turns):
  U, U', U2, D, D', D2, F, F', F2, B, B', B2, L, L', L2, R, R', R2
  Index: U=0, U'=1, U2=2, D=3, D'=4, D2=5, F=6, F'=7, F2=8,
         B=9, B'=10, B2=11, L=12, L'=13, L2=14, R=15, R'=16, R2=17
"""

from __future__ import annotations

import numpy as np # pyright: ignore[reportMissingImports]
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_STICKERS = 24
NUM_COLORS   = 6
STATE_DIM    = NUM_STICKERS * NUM_COLORS  # 144

COLOR_TO_IDX = {"W": 0, "Y": 1, "O": 2, "R": 3, "G": 4, "B": 5}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

MOVE_NAMES = ["U", "U'", "U2", "D", "D'", "D2",
              "F", "F'", "F2", "B", "B'", "B2",
              "L", "L'", "L2", "R", "R'", "R2"]
NUM_MOVES      = len(MOVE_NAMES)  # 18
MOVE_NAME_TO_IDX = {name: idx for idx, name in enumerate(MOVE_NAMES)}

# Solved state: sticker i has color SOLVED_STATE[i]
# U=W(0), D=Y(1), F=O(2), B=R(3), L=G(4), R=B(5)
SOLVED_STATE = np.array(
    [0]*4 +   # U face: stickers 0-3  → White
    [1]*4 +   # D face: stickers 4-7  → Yellow
    [4]*4 +   # F face: stickers 8-11 → Green
    [5]*4 +   # B face: stickers 12-15 → Blue
    [2]*4 +   # L face: stickers 16-19 → Orange
    [3]*4,    # R face: stickers 20-23 → Red
    dtype=np.int8
)

# ---------------------------------------------------------------------------
# Move permutations
# ---------------------------------------------------------------------------

FACE_ORDER = ["U", "D", "F", "B", "L", "R"]
FACE_AXES = {
    "U": {"normal": (0, 1, 0), "right": (1, 0, 0), "down": (0, 0, 1)},
    "D": {"normal": (0, -1, 0), "right": (1, 0, 0), "down": (0, 0, -1)},
    "F": {"normal": (0, 0, 1), "right": (1, 0, 0), "down": (0, -1, 0)},
    "B": {"normal": (0, 0, -1), "right": (-1, 0, 0), "down": (0, -1, 0)},
    "L": {"normal": (-1, 0, 0), "right": (0, 0, 1), "down": (0, -1, 0)},
    "R": {"normal": (1, 0, 0), "right": (0, 0, -1), "down": (0, -1, 0)},
}


def _v_add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_scale(v: tuple[int, int, int], k: int) -> tuple[int, int, int]:
    return (v[0] * k, v[1] * k, v[2] * k)


def _v_neg(v: tuple[int, int, int]) -> tuple[int, int, int]:
    return (-v[0], -v[1], -v[2])


def _rotate_positive_axis(
    vec: tuple[int, int, int],
    axis: int,
    quarter_turns: int,
) -> tuple[int, int, int]:
    quarter_turns %= 4
    if quarter_turns == 0:
        return vec

    x, y, z = vec
    if axis == 0:
        rotations = {
            1: (x, -z, y),
            2: (x, -y, -z),
            3: (x, z, -y),
        }
    elif axis == 1:
        rotations = {
            1: (z, y, -x),
            2: (-x, y, -z),
            3: (-z, y, x),
        }
    else:
        rotations = {
            1: (-y, x, z),
            2: (-x, -y, z),
            3: (y, -x, z),
        }
    return rotations[quarter_turns]


def _rotate_about_normal(
    vec: tuple[int, int, int],
    normal: tuple[int, int, int],
    quarter_turns: int,
) -> tuple[int, int, int]:
    axis = next(i for i, value in enumerate(normal) if value != 0)
    sign = normal[axis]
    return _rotate_positive_axis(vec, axis, quarter_turns * sign)


def _cw_turn_for_face(face_name: str) -> int:
    spec = FACE_AXES[face_name]
    up = _v_neg(spec["down"])
    if _rotate_about_normal(up, spec["normal"], 1) == spec["right"]:
        return 1
    return -1


def _build_sticker_descriptors() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    descriptors = []
    for face_name in FACE_ORDER:
        spec = FACE_AXES[face_name]
        normal = spec["normal"]
        right = spec["right"]
        down = spec["down"]
        for row in range(2):
            for col in range(2):
                row_sign = -1 if row == 0 else 1
                col_sign = -1 if col == 0 else 1
                position = _v_add(
                    normal,
                    _v_add(_v_scale(right, col_sign), _v_scale(down, row_sign)),
                )
                descriptors.append((position, normal))
    return descriptors


def _build_cw_permutation(face_name: str) -> np.ndarray:
    spec = FACE_AXES[face_name]
    normal = spec["normal"]
    axis = next(i for i, value in enumerate(normal) if value != 0)
    layer_value = normal[axis]
    quarter_turn = _cw_turn_for_face(face_name)

    permutation = np.arange(NUM_STICKERS, dtype=np.int8)
    for old_idx, (position, sticker_normal) in enumerate(STICKER_DESCRIPTORS):
        if position[axis] != layer_value:
            continue
        new_position = _rotate_about_normal(position, normal, quarter_turn)
        new_normal = _rotate_about_normal(sticker_normal, normal, quarter_turn)
        new_idx = STICKER_INDEX[(new_position, new_normal)]
        permutation[new_idx] = old_idx
    return permutation


def _build_global_rotation_permutation(axis: int, quarter_turns: int) -> np.ndarray:
    permutation = np.arange(NUM_STICKERS, dtype=np.int8)
    for old_idx, (position, sticker_normal) in enumerate(STICKER_DESCRIPTORS):
        new_position = _rotate_positive_axis(position, axis, quarter_turns)
        new_normal = _rotate_positive_axis(sticker_normal, axis, quarter_turns)
        new_idx = STICKER_INDEX[(new_position, new_normal)]
        permutation[new_idx] = old_idx
    return permutation


def _build_cube_rotation_permutations() -> list[np.ndarray]:
    identity = np.arange(NUM_STICKERS, dtype=np.int8)
    generators = [
        _build_global_rotation_permutation(axis=0, quarter_turns=1),
        _build_global_rotation_permutation(axis=1, quarter_turns=1),
        _build_global_rotation_permutation(axis=2, quarter_turns=1),
    ]

    permutations = [identity]
    seen = {tuple(int(idx) for idx in identity)}
    queue = [identity]

    while queue:
        current = queue.pop()
        for generator in generators:
            composed = current[generator]
            key = tuple(int(idx) for idx in composed)
            if key in seen:
                continue
            seen.add(key)
            permutations.append(composed)
            queue.append(composed)

    return permutations


STICKER_DESCRIPTORS = _build_sticker_descriptors()
STICKER_INDEX = {descriptor: idx for idx, descriptor in enumerate(STICKER_DESCRIPTORS)}
BASE_PERMUTATIONS = [_build_cw_permutation(face_name) for face_name in FACE_ORDER]
INV_PERMUTATIONS = [np.argsort(p).astype(np.int8) for p in BASE_PERMUTATIONS]
CUBE_ROTATION_PERMUTATIONS = _build_cube_rotation_permutations()
SOLVED_SYMMETRY_STATES = np.stack(
    [SOLVED_STATE[permutation] for permutation in CUBE_ROTATION_PERMUTATIONS]
)

# Base-6 packing: a (24,) state in [0..5] fits in uint64 since 6**24 ≈ 4.74e18 < 2**64 ≈ 1.84e19.
_POWERS_OF_6 = (6 ** np.arange(NUM_STICKERS, dtype=np.uint64)).astype(np.uint64)


def pack_state(state: np.ndarray) -> int:
    """Pack a (24,) int8 state with values in [0..5] to a unique uint64."""
    return int((np.asarray(state, dtype=np.uint64) * _POWERS_OF_6).sum())


def _pack_states_batch(states: np.ndarray) -> np.ndarray:
    """Vectorized pack_state over a (N, 24) array."""
    return (states.astype(np.uint64) * _POWERS_OF_6).sum(axis=1)


class DistanceTable:
    """Dict-like wrapper over sorted uint64 keys + int8 distances.

    Replaces the old dict[bytes, int] cache. Backed by two numpy arrays that
    can be memory-mapped from .npz, so load is effectively instant even for
    88M entries (vs ~30–60s for the old pickle.load path).

    Accepts bytes, numpy arrays, or already-packed ints as keys, so call sites
    doing `distances[state.tobytes()]` or `distances.get(state.tobytes())`
    keep working.
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        self._keys = np.asarray(keys, dtype=np.uint64)
        self._values = np.asarray(values, dtype=np.int8)

    def _as_packed(self, key) -> int:
        if isinstance(key, (bytes, bytearray, memoryview)):
            return pack_state(np.frombuffer(bytes(key), dtype=np.int8))
        if isinstance(key, np.ndarray):
            return pack_state(key)
        return int(key)

    def __getitem__(self, key) -> int:
        k = self._as_packed(key)
        idx = int(np.searchsorted(self._keys, np.uint64(k)))
        if idx < len(self._keys) and int(self._keys[idx]) == k:
            return int(self._values[idx])
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return self.get(key, None) is not None

    def __len__(self) -> int:
        return int(len(self._keys))

    def values(self):
        return (int(v) for v in self._values)

    @classmethod
    def load(cls, path) -> "DistanceTable":
        """Memory-map an .npz file saved by `save()`."""
        data = np.load(path, mmap_mode="r")
        return cls(data["keys"], data["values"])

    def save(self, path) -> None:
        """Write to .npz. Callers should use a `.npz` extension."""
        np.savez(path, keys=self._keys, values=self._values)

CORNER_STICKERS = [
    (3, 9, 20),   # UFR
    (2, 8, 17),   # UFL
    (0, 13, 16),  # UBL
    (1, 12, 21),  # UBR
    (5, 11, 22),  # DFR
    (4, 10, 19),  # DFL
    (6, 15, 18),  # DBL
    (7, 14, 23),  # DBR
]
VALID_CORNER_COLOR_SETS = {
    tuple(sorted(int(SOLVED_STATE[idx]) for idx in stickers))
    for stickers in CORNER_STICKERS
}

# ---------------------------------------------------------------------------
# Cube class
# ---------------------------------------------------------------------------

class Cube:
    """
    2x2x2 Rubik's Cube.

    State is a (24,) int8 array of color indices.
    Encoding is a (144,) float32 one-hot vector for transformer input.
    """

    def __init__(self, state: Optional[np.ndarray] = None):
        if state is not None:
            if state.shape != (NUM_STICKERS,):
                raise ValueError(f"State must have shape ({NUM_STICKERS},), got {state.shape}")
            self.state = state.astype(np.int8)
        else:
            self.state = SOLVED_STATE.copy()

    # ------------------------------------------------------------------
    # Core move application
    # ------------------------------------------------------------------

    def apply_move(self, move_idx: int) -> "Cube":
        """
        Apply one of the 18 moves (in-place). Returns self for chaining.
        move_idx: 0..17 → [U, U', U2, D, D', D2, F, F', F2, B, B', B2, L, L', L2, R, R', R2]
        """
        if not 0 <= move_idx < NUM_MOVES:
            raise ValueError(f"Invalid move index {move_idx}")
        face_idx = move_idx // 3          # 0=U,1=D,2=F,3=B,4=L,5=R
        turn_type = move_idx % 3          # 0=CW, 1=CCW, 2=180

        if turn_type == 0:                # CW (quarter turn)
            self.state = self.state[BASE_PERMUTATIONS[face_idx]]
        elif turn_type == 1:              # CCW (inverse of CW)
            self.state = self.state[INV_PERMUTATIONS[face_idx]]
        else:                             # 180° = 2× CW
            p = BASE_PERMUTATIONS[face_idx]
            self.state = self.state[p][p]

        return self

    def apply_move_name(self, name: str) -> "Cube":
        """Apply a move by name, e.g. 'U', \"U'\", 'U2'."""
        if name not in MOVE_NAME_TO_IDX:
            raise ValueError(f"Unknown move name {name!r}")
        return self.apply_move(MOVE_NAME_TO_IDX[name])

    # ------------------------------------------------------------------
    # Scrambling
    # ------------------------------------------------------------------

    def scramble(self, n: int, rng: Optional[np.random.Generator] = None) -> list[int]:
        """
        Apply n random moves. Returns the list of move indices applied.
        Avoids applying the same face twice in a row (reduces redundant sequences).
        """
        if rng is None:
            rng = np.random.default_rng()

        moves = []
        last_face = -1
        for _ in range(n):
            # Exclude moves on the same face as the last move
            candidates = [m for m in range(NUM_MOVES) if m // 3 != last_face]
            move = int(rng.choice(candidates))
            self.apply_move(move)
            moves.append(move)
            last_face = move // 3
        return moves

    @classmethod
    def from_scramble(cls, n: int, rng=None) -> tuple["Cube", list[int]]:
        """Return (scrambled_cube, move_sequence) starting from solved."""
        cube = cls()
        moves = cube.scramble(n, rng)
        return cube, moves

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self) -> np.ndarray:
        """
        One-hot encode the state.
        Returns float32 array of shape (144,) — the transformer input token.
        """
        return np.eye(NUM_COLORS, dtype=np.float32)[self.state].flatten()

    @staticmethod
    def decode(encoded: np.ndarray) -> "Cube":
        """Reconstruct a Cube from a (144,) one-hot encoded vector."""
        if encoded.shape != (STATE_DIM,):
            raise ValueError(f"Encoded state must have shape ({STATE_DIM},), got {encoded.shape}")
        oh = encoded.reshape(NUM_STICKERS, NUM_COLORS)
        state = oh.argmax(axis=1).astype(np.int8)
        return Cube(state)

    # ------------------------------------------------------------------
    # Probe label extraction
    # ------------------------------------------------------------------

    def is_solved(self) -> bool:
        return bool(np.any(np.all(SOLVED_SYMMETRY_STATES == self.state, axis=1)))

    def face_solved(self) -> np.ndarray:
        """
        Returns bool array of shape (6,): whether each face is a single color.
        Order: U, D, F, B, L, R
        """
        solved = np.zeros(6, dtype=bool)
        for f in range(6):
            face_stickers = self.state[f*4 : f*4+4]
            solved[f] = bool((face_stickers == face_stickers[0]).all())
        return solved

    def corner_oriented(self) -> np.ndarray:
        """
        Returns bool array of shape (8,): whether each corner cubie has
        its U/D color facing up or down (orientation 0 = correct).

        Corner cubies and their sticker indices (U/D sticker, then the two side stickers):
          UFR: U=3,  F=9,  R=20
          UFL: U=2,  F=8,  L=17
          UBR: U=1,  B=12, R=21
          UBL: U=0,  B=13, L=16
          DFR: D=5,  F=11, R=22
          DFL: D=4,  F=10, L=19
          DBR: D=7,  B=14, R=23   (note: check against your permutations)
          DBL: D=6,  B=15, L=18
        """
        ud_colors = {COLOR_TO_IDX["W"], COLOR_TO_IDX["Y"]}
        oriented = np.zeros(8, dtype=bool)
        for i, (ud_idx, _, _) in enumerate(CORNER_STICKERS):
            oriented[i] = self.state[ud_idx] in ud_colors
        return oriented

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "Cube":
        return Cube(self.state.copy())

    def __repr__(self) -> str:
        colors = [IDX_TO_COLOR[int(c)] for c in self.state]
        faces = ["U", "D", "F", "B", "L", "R"]
        lines = []
        for f, name in enumerate(faces):
            row = colors[f*4 : f*4+4]
            lines.append(f"{name}: [{row[0]}{row[1]}|{row[2]}{row[3]}]")
        return "  ".join(lines)


# ---------------------------------------------------------------------------
# Optimal distance (BFS)
# ---------------------------------------------------------------------------

def compute_optimal_distances(verbose: bool = True) -> DistanceTable:
    """
    BFS over every state reachable by face moves from the solved cube (~88M
    states). Returns a `DistanceTable` (sorted uint64 packed keys + int8
    distances) that serialises to a compact .npz file and loads via mmap.

    Seeding from `SOLVED_STATE` alone is enough: face moves already connect
    every rotationally-equivalent solved state, so starting from all 24 would
    add the same entries a second time without growing the reachable set.

    Runtime: ~25–30 min depending on machine; peak memory during the BFS is
    ~15 GB (the intermediate dict). Call once and cache.
    """
    from collections import deque
    import time

    distances: dict[bytes, int] = {}
    queue: deque[bytes] = deque()

    key = SOLVED_STATE.tobytes()
    distances[key] = 0
    queue.append(key)

    t0 = time.perf_counter()
    while queue:
        key = queue.popleft()
        d = distances[key]
        state = np.frombuffer(key, dtype=np.int8)  # zero-copy view

        for move_idx in range(NUM_MOVES):
            face_idx = move_idx // 3
            turn_type = move_idx % 3
            if turn_type == 0:
                new_state = state[BASE_PERMUTATIONS[face_idx]]
            elif turn_type == 1:
                new_state = state[INV_PERMUTATIONS[face_idx]]
            else:
                p = BASE_PERMUTATIONS[face_idx]
                new_state = state[p][p]

            new_key = new_state.tobytes()
            if new_key not in distances:
                distances[new_key] = d + 1
                queue.append(new_key)

    bfs_seconds = time.perf_counter() - t0
    if verbose:
        from collections import Counter
        counts = Counter(distances.values())
        for dist in sorted(counts):
            print(f"  distance {dist:2d}: {counts[dist]:>12,} states")
        print(f"  total: {len(distances):,} states  ({bfs_seconds:.1f}s)")

    # Convert the dict to sorted (uint64, int8) arrays.
    # b''.join + frombuffer is all C-speed; this is much faster than a Python loop.
    if verbose:
        print("  packing and sorting for on-disk format...", end=" ", flush=True)
    t_pack = time.perf_counter()
    n = len(distances)
    state_array = np.frombuffer(b"".join(distances.keys()), dtype=np.int8).reshape(n, NUM_STICKERS)
    values = np.fromiter(distances.values(), dtype=np.int8, count=n)
    packed = _pack_states_batch(state_array)
    order = np.argsort(packed)
    table = DistanceTable(packed[order], values[order])
    if verbose:
        print(f"{time.perf_counter() - t_pack:.1f}s")
    return table


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sequences: int,
    max_scramble_depth: int = 11,
    rng: Optional[np.random.Generator] = None,
    distances: Optional["DistanceTable"] = None,
) -> dict:
    """
    Generate a dataset of (state, next_move) pairs from random scramble sequences.

    Pass `distances` (from compute_optimal_distances) to include an
    `optimal_distance` field — the true HTM distance to solved for each state.

    Returns a dict with:
      states:           float32 (N, 144)  — one-hot encoded cube states
      next_moves:       int64   (N,)      — move index applied to reach next state
      scramble_depth:   int32   (N,)      — step index in the scramble sequence
      face_solved:      bool    (N, 6)    — per-face solved status
      corner_oriented:  bool    (N, 8)    — per-corner orientation status
      optimal_distance: int8    (N,)      — HTM distance to solved (if distances given)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    all_states, all_moves, all_depths = [], [], []
    all_face_solved, all_corner_oriented = [], []
    all_optimal: list[int] = [] if distances is not None else None  # type: ignore[assignment]

    for _ in range(n_sequences):
        depth = int(rng.integers(1, max_scramble_depth + 1))
        cube = Cube()
        moves = cube.scramble(depth, rng)

        # Walk through the sequence and record (state_before_move, move)
        replay = Cube()
        for step, move in enumerate(moves):
            all_states.append(replay.encode())
            all_moves.append(move)
            all_depths.append(step)  # step index from solved in this scramble
            all_face_solved.append(replay.face_solved())
            all_corner_oriented.append(replay.corner_oriented())
            if distances is not None:
                all_optimal.append(distances[replay.state.tobytes()])
            replay.apply_move(move)

    result: dict = {
        "states":          np.stack(all_states).astype(np.float32),
        "next_moves":      np.array(all_moves, dtype=np.int64),
        "scramble_depth":  np.array(all_depths, dtype=np.int32),
        "face_solved":     np.stack(all_face_solved),
        "corner_oriented": np.stack(all_corner_oriented),
    }
    if distances is not None:
        result["optimal_distance"] = np.array(all_optimal, dtype=np.int8)
    return result


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(state: np.ndarray, distances: "DistanceTable") -> list[int]:
    """
    Return the optimal (HTM) solution sequence for a given cube state.

    Uses the precomputed BFS distance table to greedily pick whichever move
    reduces the distance by exactly 1 at each step.  Because the table encodes
    true shortest-path distances, this greedy choice is always optimal.

    Args:
        state:     (24,) int8 array of color indices (from Cube.state)
        distances: DistanceTable from compute_optimal_distances / DistanceTable.load

    Returns:
        List of move indices (0–17) that solve the cube.  Empty list if already solved.
    """
    state = np.asarray(state, dtype=np.int8)
    solution: list[int] = []

    for _ in range(20):  # God's number for 2x2 is 11; 20 is a safe upper bound
        key = state.tobytes()
        d = distances.get(key)
        if d is None:
            raise ValueError("State not found in distance table — was it computed correctly?")
        if d == 0:
            break

        for move_idx in range(NUM_MOVES):
            face_idx = move_idx // 3
            turn_type = move_idx % 3
            if turn_type == 0:
                next_state = state[BASE_PERMUTATIONS[face_idx]]
            elif turn_type == 1:
                next_state = state[INV_PERMUTATIONS[face_idx]]
            else:
                p = BASE_PERMUTATIONS[face_idx]
                next_state = state[p][p]

            if distances.get(next_state.tobytes()) == d - 1:
                solution.append(move_idx)
                state = next_state
                break

    return solution


def generate_scramble_solution_pairs(
    n_pairs: int,
    distances: "DistanceTable",
    max_scramble: int = 11,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate (scramble_moves, solution_moves) pairs.

    Each pair starts from the solved state, applies a random scramble of up to
    `max_scramble` moves, then finds the optimal solution using the BFS table.

    Returns:
        List of (scramble_moves, solution_moves) tuples, where each element is
        a list of move indices (0–17).  Use MOVE_NAMES[i] to convert to strings.
    """
    if rng is None:
        rng = np.random.default_rng()

    pairs: list[tuple[list[int], list[int]]] = []
    for _ in range(n_pairs):
        depth = int(rng.integers(1, max_scramble + 1))
        cube = Cube()
        scramble_moves = cube.scramble(depth, rng)
        solution_moves = solve(cube.state, distances)
        pairs.append((scramble_moves, solution_moves))
    return pairs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_tests():
    print("Running unit tests...\n")

    def assert_physical_corner_sets(cube: Cube, label: str):
        for stickers in CORNER_STICKERS:
            color_set = tuple(sorted(int(cube.state[idx]) for idx in stickers))
            assert color_set in VALID_CORNER_COLOR_SETS, (
                f"{label}: invalid corner color set {color_set} at stickers {stickers}"
            )

    # Test 1: solved cube encodes correctly
    cube = Cube()
    enc = cube.encode()
    assert enc.shape == (144,), f"Expected (144,), got {enc.shape}"
    assert enc.dtype == np.float32
    assert enc.sum() == 24.0, "Each of 24 stickers should contribute exactly one 1"
    print("PASS  encode shape and sum")

    # Test 2: round-trip encode/decode
    cube2 = Cube.decode(enc)
    assert np.array_equal(cube.state, cube2.state), "Decode should recover original state"
    print("PASS  encode → decode round-trip")

    # Test 3: solved cube is solved
    assert cube.is_solved(), "Fresh cube should be solved"
    print("PASS  is_solved() on fresh cube")

    # Test 4: all 24 whole-cube rotations of solved count as solved
    assert len(CUBE_ROTATION_PERMUTATIONS) == 24, "Cube should have 24 rotational symmetries"
    for permutation in CUBE_ROTATION_PERMUTATIONS:
        rotated = Cube(SOLVED_STATE[permutation])
        assert rotated.is_solved(), "Solved cube should remain solved under whole-cube rotation"
    print("PASS  is_solved() is rotation-invariant across 24 symmetries")

    # Test 5: rotated unsolved states are still unsolved
    unsolved = Cube()
    unsolved.apply_move_name("F")
    for permutation in CUBE_ROTATION_PERMUTATIONS:
        rotated_unsolved = Cube(unsolved.state[permutation])
        assert not rotated_unsolved.is_solved(), "Unsolved cube should stay unsolved under rotation"
    print("PASS  unsolved states stay unsolved under whole-cube rotation")

    # Test 6: apply move then inverse returns to start
    for move_idx in range(0, NUM_MOVES, 3):  # test each CW move
        c = Cube()
        c.apply_move(move_idx)          # CW
        c.apply_move(move_idx + 1)      # CCW (inverse)
        assert c.is_solved(), f"Move {MOVE_NAMES[move_idx]} + inverse should return to solved"
    print("PASS  move + inverse = identity (all 6 faces)")

    # Test 7: U4 = identity
    c = Cube()
    for _ in range(4):
        c.apply_move(0)  # U CW × 4
    assert c.is_solved(), "Four U moves should return to solved"
    print("PASS  U×4 = identity")

    # Test 8: 180° move applied twice = identity
    for face in range(6):
        c = Cube()
        move_180 = face * 3 + 2
        c.apply_move(move_180)
        c.apply_move(move_180)
        assert c.is_solved(), f"180° move ×2 should return to solved (face {face})"
    print("PASS  180° move ×2 = identity (all 6 faces)")

    # Test 9: solved cube has all faces solved
    cube = Cube()
    assert cube.face_solved().all(), "Solved cube: all faces should be solved"
    print("PASS  face_solved() on solved cube")

    # Test 10: solved cube has all corners oriented
    cube = Cube()
    assert cube.corner_oriented().all(), "Solved cube: all corners should be oriented"
    print("PASS  corner_oriented() on solved cube")

    # Test 11: scramble produces a different state
    cube = Cube()
    cube.scramble(5)
    assert not cube.is_solved(), "Scrambled cube should (almost certainly) not be solved"
    print("PASS  scramble changes state")

    # Test 12: every single move preserves physically valid corner cubies
    for move_name in MOVE_NAMES:
        c = Cube()
        c.apply_move_name(move_name)
        assert_physical_corner_sets(c, f"single move {move_name}")
    print("PASS  single moves preserve valid corner cubies")

    # Test 13: random sequences preserve physically valid corner cubies
    rng = np.random.default_rng(123)
    c = Cube()
    c.scramble(100, rng)
    assert_physical_corner_sets(c, "random scramble")
    print("PASS  random scrambles preserve valid corner cubies")

    # Test 14: dataset generation runs and has correct shapes
    data = generate_dataset(n_sequences=10, max_scramble_depth=5)
    N = data["states"].shape[0]
    assert data["states"].shape == (N, 144)
    assert data["next_moves"].shape == (N,)
    assert data["face_solved"].shape == (N, 6)
    assert data["corner_oriented"].shape == (N, 8)
    print(f"PASS  generate_dataset shapes correct (N={N})")

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_tests()

    print("\nExample cube state:")
    cube = Cube()
    cube.scramble(3)
    print(f"  {cube}")
    print(f"  encoded shape: {cube.encode().shape}")
    print(f"  face_solved:   {cube.face_solved()}")
    print(f"  corner_oriented: {cube.corner_oriented()}")
