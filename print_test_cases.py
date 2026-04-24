"""
print_test_cases.py

Generates scrambled cube states and prints each text representation
ready to copy-paste into a chat interface (Claude.ai, ChatGPT, etc.).
Also prints the known optimal solution for validation.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import Cube, DistanceTable, MOVE_NAMES, CORNER_STICKERS, IDX_TO_COLOR, SOLVED_STATE, solve

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "distances.npz")

FACE_LABELS = ["U", "D", "F", "B", "L", "R"]
POS_NAMES   = ["UFR", "UFL", "UBL", "UBR", "DFR", "DFL", "DBL", "DBR"]
CORNER_FACE_DIRS = [
    ("U","F","R"), ("U","F","L"), ("U","B","L"), ("U","B","R"),
    ("D","F","R"), ("D","F","L"), ("D","B","L"), ("D","B","R"),
]
UD_COLORS = {"W", "Y"}

def _c(state, i):
    return IDX_TO_COLOR[int(state[i])]


# ---------------------------------------------------------------------------
# Existing representations
# ---------------------------------------------------------------------------

def fmt_face_grid(cube):
    s = cube.state
    c = lambda i: _c(s, i)
    return "\n".join([
        "Standard orientation: White on top (U), Green in front (F)",
        "Each face shown as viewed from outside the cube.",
        "",
        "        U (top):",
        f"        {c(0)} {c(1)}",
        f"        {c(2)} {c(3)}",
        "",
        "L (left)    F (front)   R (right)   B (back)",
        f"{c(16)} {c(17)}          {c(8)} {c(9)}          {c(20)} {c(21)}         {c(12)} {c(13)}",
        f"{c(18)} {c(19)}         {c(10)} {c(11)}         {c(22)} {c(23)}         {c(14)} {c(15)}",
        "",
        "        D (bottom):",
        f"        {c(4)} {c(5)}",
        f"        {c(6)} {c(7)}",
    ])


def fmt_compact_string(cube):
    s = cube.state
    face_strs = ["".join(_c(s, f*4 + i) for i in range(4)) for f in range(6)]
    return "\n".join([
        "Standard orientation: White on top (U), Green in front (F)",
        "Format: 6 faces × 4 stickers, reading order (TL TR BL BR) as viewed from outside.",
        "Face order: U D F B L R",
        "",
        " ".join(f"{label}:{fs}" for label, fs in zip(FACE_LABELS, face_strs)),
    ])


def fmt_corner_cubies(cube):
    s = cube.state
    lines = [
        "Standard orientation: White on top (U), Green in front (F)",
        "Each corner shows which color is visible on each face at that position.",
        "",
        "Solved reference:",
        "  UFR:(U:W F:G R:R)  UFL:(U:W F:G L:O)  UBL:(U:W B:B L:O)  UBR:(U:W B:B R:R)",
        "  DFR:(D:Y F:G R:R)  DFL:(D:Y F:G L:O)  DBL:(D:Y B:B L:O)  DBR:(D:Y B:B R:R)",
        "",
        "Current state:",
    ]
    for i, (si0, si1, si2) in enumerate(CORNER_STICKERS):
        pos = POS_NAMES[i]
        d0, d1, d2 = CORNER_FACE_DIRS[i]
        lines.append(f"  {pos}:({d0}:{_c(s,si0)} {d1}:{_c(s,si1)} {d2}:{_c(s,si2)})")
    return "\n".join(lines)


def fmt_move_sequence(scramble_moves):
    return "\n".join([
        "Standard orientation: White on top (U), Green in front (F)",
        "Starting from solved, this scramble was applied:",
        "",
        "  " + " ".join(MOVE_NAMES[m] for m in scramble_moves),
    ])


# ---------------------------------------------------------------------------
# New: piece-identity representation
# ---------------------------------------------------------------------------

# Build piece table from solved state once at import time
_SOLVED_PIECES = []  # index = corner position idx → (piece_name, ud_color, frozenset_colors)
for _i, (_si0, _si1, _si2) in enumerate(CORNER_STICKERS):
    _c0 = IDX_TO_COLOR[int(SOLVED_STATE[_si0])]
    _c1 = IDX_TO_COLOR[int(SOLVED_STATE[_si1])]
    _c2 = IDX_TO_COLOR[int(SOLVED_STATE[_si2])]
    _name = f"{_c0}-{_c1}-{_c2}"
    _ud = next(c for c in [_c0, _c1, _c2] if c in UD_COLORS)
    _SOLVED_PIECES.append((_name, _ud, frozenset([_c0, _c1, _c2])))


def fmt_piece_identity(cube):
    """
    Describes the cube in terms of piece permutation and orientation:
    for each of the 8 corner pieces (named by their solved-state colors),
    says where it currently is and which face its W/Y sticker is facing.

    This is richer than corner_cubies because it makes the permutation
    explicit — the model knows directly which piece is displaced and where
    it needs to go, rather than having to infer piece identity from raw colors.
    """
    s = cube.state

    # Read current colors at each position
    current = []
    for si0, si1, si2 in CORNER_STICKERS:
        c0 = IDX_TO_COLOR[int(s[si0])]
        c1 = IDX_TO_COLOR[int(s[si1])]
        c2 = IDX_TO_COLOR[int(s[si2])]
        current.append((frozenset([c0, c1, c2]), c0, c1, c2))

    color_to_pos = {colors: i for i, (colors, _, _, _) in enumerate(current)}

    lines = [
        "Standard orientation: White on top (U), Green in front (F)",
        "8 corner pieces, each named [W/Y]-[F/B-color]-[L/R-color] as in the solved state.",
        "Goal: every piece at its home position with its W or Y sticker facing U or D.",
        "",
        "Solved reference:",
        "  W-G-R@UFR  W-G-O@UFL  W-B-O@UBL  W-B-R@UBR",
        "  Y-G-R@DFR  Y-G-O@DFL  Y-B-O@DBL  Y-B-R@DBR",
        "",
        "Current state (piece → current position, which face its W/Y sticker faces):",
    ]

    for home_idx, (piece_name, ud_color, piece_colors) in enumerate(_SOLVED_PIECES):
        home_pos = POS_NAMES[home_idx]
        cur_idx  = color_to_pos[piece_colors]
        cur_pos  = POS_NAMES[cur_idx]
        dirs     = CORNER_FACE_DIRS[cur_idx]
        _, c0, c1, c2 = current[cur_idx]

        # Which face is the W/Y sticker facing?
        if c0 == ud_color:
            ud_face = dirs[0]
        elif c1 == ud_color:
            ud_face = dirs[1]
        else:
            ud_face = dirs[2]

        at_home    = (cur_idx == home_idx)
        ud_correct = ud_face in ("U", "D")

        if at_home and ud_correct:
            desc = "✓ solved"
        elif at_home:
            desc = f"home ({home_pos}), but {ud_color} faces {ud_face} — needs a twist"
        elif ud_correct:
            desc = f"at {cur_pos}, {ud_color} faces {ud_face} ✓ — needs to move home ({home_pos})"
        else:
            desc = f"at {cur_pos}, {ud_color} faces {ud_face} — needs to move to {home_pos} + twist"

        lines.append(f"  {piece_name}: {desc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_DIRECT = """\
You are an expert at solving 2x2x2 Rubik's cubes.

Move notation:
- U/D/F/B/L/R = top/bottom/front/back/left/right face
- No suffix = 90° clockwise (when looking at that face head-on)
- ' suffix = 90° counterclockwise
- 2 suffix = 180°

Solved = each face is a single color.
Standard solved state: White top (U), Green front (F), Red right (R), Orange left (L), Blue back (B), Yellow bottom (D).

Respond with ONLY the solution moves separated by spaces. No explanation."""

SYSTEM_COT = """\
You are an expert at solving 2x2x2 Rubik's cubes.

Move notation:
- U/D/F/B/L/R = top/bottom/front/back/left/right face
- No suffix = 90° clockwise (when looking at that face head-on)
- ' suffix = 90° counterclockwise
- 2 suffix = 180°

Solved = each face is a single color.
Standard solved state: White top (U), Green front (F), Red right (R), Orange left (L), Blue back (B), Yellow bottom (D).

Think step by step: identify which pieces are displaced or misoriented, plan which moves will fix each one, and track how each move affects the other pieces.
End your response with a line that reads exactly: Solution: [moves]"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def divider(label=""):
    w = 70
    if label:
        pad = (w - len(label) - 2) // 2
        print("─" * pad + f" {label} " + "─" * (w - pad - len(label) - 2))
    else:
        print("─" * w)


def run():
    print("Loading BFS distance table...")
    distances = DistanceTable.load(CACHE_PATH)
    print(f"  {len(distances):,} states loaded\n")

    rng = np.random.default_rng(42)
    target_distances = [3, 5, 7]

    cases = []
    for target_d in target_distances:
        for _ in range(2000):
            cube = Cube()
            scramble = cube.scramble(target_d + 4, rng)
            if distances[cube.state.tobytes()] == target_d:
                cases.append((cube.state.copy(), scramble, target_d))
                break

    # ---- Section 1: direct prompt, all 5 representations ----
    print("=" * 70)
    print("SECTION 1 — DIRECT PROMPT (respond with moves only)")
    print("=" * 70)
    print("\nSYSTEM PROMPT:")
    print(SYSTEM_DIRECT)

    for i, (state, scramble_moves, opt_d) in enumerate(cases):
        cube = Cube(state.copy())
        opt_str = " ".join(MOVE_NAMES[m] for m in solve(state, distances))

        print(f"\n\n{'#' * 70}")
        print(f"# TEST CASE {i+1}  —  optimal distance: {opt_d} moves")
        print(f"# KNOWN OPTIMAL SOLUTION: {opt_str}")
        print(f"{'#' * 70}")

        reps = [
            ("1. face_grid",       fmt_face_grid(cube)),
            ("2. compact_string",  fmt_compact_string(cube)),
            ("3. corner_cubies",   fmt_corner_cubies(cube)),
            ("4. piece_identity",  fmt_piece_identity(cube)),
            ("5. move_sequence",   fmt_move_sequence(scramble_moves)),
        ]
        for rep_name, rep_text in reps:
            print()
            divider(rep_name)
            print(f"Solve this 2x2x2 cube:\n\n{rep_text}")
            divider()

    # ---- Section 2: chain-of-thought, piece_identity only ----
    print(f"\n\n{'=' * 70}")
    print("SECTION 2 — CHAIN-OF-THOUGHT PROMPT (piece_identity only)")
    print("=" * 70)
    print("\nSYSTEM PROMPT:")
    print(SYSTEM_COT)

    for i, (state, scramble_moves, opt_d) in enumerate(cases):
        cube = Cube(state.copy())
        opt_str = " ".join(MOVE_NAMES[m] for m in solve(state, distances))

        print(f"\n\n{'#' * 70}")
        print(f"# TEST CASE {i+1}  —  optimal distance: {opt_d} moves")
        print(f"# KNOWN OPTIMAL SOLUTION: {opt_str}")
        print(f"{'#' * 70}")

        print()
        divider("piece_identity + CoT")
        print(f"Solve this 2x2x2 cube:\n\n{fmt_piece_identity(cube)}")
        divider()


if __name__ == "__main__":
    run()
