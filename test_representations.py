"""
test_representations.py

Tests 4 text representations of 2x2x2 Rubik's cube states across multiple LLMs
to find which representations enable LLM-based solving.

Representations tested:
  1. face_grid       — 2x2 grid per face, compass layout
  2. compact_string  — 24-char flat string, faces in U/D/F/B/L/R order
  3. corner_cubies   — per-corner piece description (which color faces each direction)
  4. piece_identity  — per-piece name, current position, and W/Y face orientation
  5. move_sequence   — the scramble sequence itself (degenerate ceiling)

Supported providers (set the corresponding env var to activate):
  OPENAI_API_KEY    — gpt-*, o1-*, o3-*, o4-*
  GEMINI_API_KEY    — gemini-*
  ANTHROPIC_API_KEY — claude-*
  GROQ_API_KEY      — llama-*, mixtral-*, and other Groq-hosted models

Edit MODELS below to choose which models to run.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

# ---------------------------------------------------------------------------
# Config — edit these to choose models and test volume
# ---------------------------------------------------------------------------

MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "gpt-4o",
    "gemini-2.5-pro",
    "llama-3.3-70b-versatile",
]

N_PER_DISTANCE = 10  # test cases per distance level (50 total × 5 reps × n_models queries)

RESULTS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llm_eval_cache.json")

# Seconds to sleep between queries — Groq has stricter rate limits than the other providers
PROVIDER_SLEEP = {
    "openai":    4.0,
    "anthropic": 4.0,
    "gemini":    4.0,
    "groq":     10.0,
}

# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import (
    Cube, DistanceTable, MOVE_NAMES, MOVE_NAME_TO_IDX, CORNER_STICKERS,
    IDX_TO_COLOR, SOLVED_STATE, solve,
)

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "distances.npz")

# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

def provider_for(model: str) -> str:
    if model.startswith(("gpt-", "o1-", "o2-", "o3-", "o4-")):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    return "groq"

KEY_VARS = {
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}

def build_clients(models: list[str]) -> dict:
    needed = {provider_for(m) for m in models}
    clients = {}
    for provider in needed:
        key = os.environ.get(KEY_VARS[provider])
        if not key:
            print(f"  Warning: {KEY_VARS[provider]} not set — skipping {provider} models")
            continue
        if provider == "openai":
            from openai import OpenAI
            clients[provider] = OpenAI(api_key=key)
        elif provider == "gemini":
            from google import genai
            clients[provider] = genai.Client(api_key=key)
        elif provider == "anthropic":
            import anthropic
            clients[provider] = anthropic.Anthropic(api_key=key)
        elif provider == "groq":
            from groq import Groq
            clients[provider] = Groq(api_key=key)
    return clients

# ---------------------------------------------------------------------------
# Representation formatters
# ---------------------------------------------------------------------------

FACE_LABELS = ["U", "D", "F", "B", "L", "R"]

def _c(state, i):
    return IDX_TO_COLOR[int(state[i])]


def fmt_face_grid(cube: Cube) -> str:
    c = lambda i: _c(cube.state, i)
    lines = [
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
    ]
    return "\n".join(lines)


def fmt_compact_string(cube: Cube) -> str:
    s = cube.state
    face_strs = ["".join(_c(s, f*4 + i) for i in range(4)) for f in range(6)]
    return "\n".join([
        "Standard orientation: White on top (U), Green in front (F)",
        "Format: 6 faces × 4 stickers, reading order (TL TR BL BR) as viewed from outside.",
        "Face order: U D F B L R",
        "",
        " ".join(f"{label}:{fs}" for label, fs in zip(FACE_LABELS, face_strs)),
    ])


def fmt_corner_cubies(cube: Cube) -> str:
    s = cube.state
    CORNER_POSITIONS = [
        ("UFR", "U", "F", "R"), ("UFL", "U", "F", "L"),
        ("UBL", "U", "B", "L"), ("UBR", "U", "B", "R"),
        ("DFR", "D", "F", "R"), ("DFL", "D", "F", "L"),
        ("DBL", "D", "B", "L"), ("DBR", "D", "B", "R"),
    ]
    lines = [
        "Standard orientation: White on top (U), Green in front (F)",
        "There are 8 corner cubies. Each is described by the color visible on each face at that corner position.",
        "",
        "Solved state for reference:",
        "  UFR:(U:W F:G R:R)  UFL:(U:W F:G L:O)  UBL:(U:W B:B L:O)  UBR:(U:W B:B R:R)",
        "  DFR:(D:Y F:G R:R)  DFL:(D:Y F:G L:O)  DBL:(D:Y B:B L:O)  DBR:(D:Y B:B R:R)",
        "",
        "Current state:",
    ]
    for (pos, d1, d2, d3), (si1, si2, si3) in zip(CORNER_POSITIONS, CORNER_STICKERS):
        lines.append(f"  {pos}:({d1}:{_c(s,si1)} {d2}:{_c(s,si2)} {d3}:{_c(s,si3)})")
    return "\n".join(lines)


def fmt_move_sequence(scramble_moves: list[int]) -> str:
    move_str = " ".join(MOVE_NAMES[m] for m in scramble_moves)
    return "\n".join([
        "Standard orientation: White on top (U), Green in front (F)",
        "Starting from the solved state, the following scramble was applied:",
        "",
        f"  {move_str}",
    ])


POS_NAMES = ["UFR", "UFL", "UBL", "UBR", "DFR", "DFL", "DBL", "DBR"]
CORNER_FACE_DIRS = [
    ("U", "F", "R"), ("U", "F", "L"), ("U", "B", "L"), ("U", "B", "R"),
    ("D", "F", "R"), ("D", "F", "L"), ("D", "B", "L"), ("D", "B", "R"),
]
UD_COLORS = {"W", "Y"}

_SOLVED_PIECES = []
for _i, (_si0, _si1, _si2) in enumerate(CORNER_STICKERS):
    _c0 = IDX_TO_COLOR[int(SOLVED_STATE[_si0])]
    _c1 = IDX_TO_COLOR[int(SOLVED_STATE[_si1])]
    _c2 = IDX_TO_COLOR[int(SOLVED_STATE[_si2])]
    _SOLVED_PIECES.append((
        f"{_c0}-{_c1}-{_c2}",
        next(c for c in [_c0, _c1, _c2] if c in UD_COLORS),
        frozenset([_c0, _c1, _c2]),
    ))


def fmt_piece_identity(cube: Cube) -> str:
    s = cube.state
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

        if c0 == ud_color:
            ud_face = dirs[0]
        elif c1 == ud_color:
            ud_face = dirs[1]
        else:
            ud_face = dirs[2]

        at_home    = (cur_idx == home_idx)
        ud_correct = ud_face in ("U", "D")

        if at_home and ud_correct:
            desc = "solved"
        elif at_home:
            desc = f"home ({home_pos}), but {ud_color} faces {ud_face} — needs a twist"
        elif ud_correct:
            desc = f"at {cur_pos}, {ud_color} faces {ud_face} — needs to move home ({home_pos})"
        else:
            desc = f"at {cur_pos}, {ud_color} faces {ud_face} — needs to move to {home_pos} + twist"

        lines.append(f"  {piece_name}: {desc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at solving 2x2x2 Rubik's cubes.

Move notation:
- U/D/F/B/L/R = top/bottom/front/back/left/right face
- No suffix = 90° clockwise (when looking at that face head-on)
- ' suffix = 90° counterclockwise
- 2 suffix = 180°

The cube is solved when every face shows a single color.
Standard solved orientation: White on top (U), Green in front (F), Red on right (R),
Orange on left (L), Blue on back (B), Yellow on bottom (D).

Do NOT write code, pseudocode, or algorithms of any kind.
Do NOT explain your reasoning or describe your approach.
Respond with ONLY the solution move sequence, moves separated by spaces.
No explanation. No extra text. Example: U R' F2 D B2 L\
"""

# ---------------------------------------------------------------------------
# Per-provider query implementations
# ---------------------------------------------------------------------------

def _query_openai(prompt: str, model: str, client) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_completion_tokens=512,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    if not text:
        raise ValueError(f"Empty response: finish_reason={response.choices[0].finish_reason}")
    return text.strip()


def _query_gemini(prompt: str, model: str, client) -> str:
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2048,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        ),
    )
    text = response.text
    if text is None and response.candidates:
        cand = response.candidates[0]
        raw_parts = (cand.content.parts if cand.content else None) or []
        texts = [p.text for p in raw_parts
                 if getattr(p, "text", None) and not getattr(p, "thought", False)]
        text = texts[-1] if texts else None
    if text is None:
        fr = response.candidates[0].finish_reason if response.candidates else "unknown"
        raise ValueError(f"Empty response: finish_reason={fr}")
    return text.strip()


def _query_anthropic(prompt: str, model: str, client) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else None
    if not text:
        raise ValueError(f"Empty response: stop_reason={response.stop_reason}")
    return text.strip()


def _query_groq(prompt: str, model: str, client) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    text = response.choices[0].message.content
    if not text:
        raise ValueError("Empty response")
    return text.strip()


_QUERY_FNS = {
    "openai":    _query_openai,
    "gemini":    _query_gemini,
    "anthropic": _query_anthropic,
    "groq":      _query_groq,
}


def query_model(prompt: str, model: str, client, retries: int = 6) -> str:
    provider = provider_for(model)
    delay = 15
    for attempt in range(retries + 1):
        try:
            return _QUERY_FNS[provider](prompt, model, client)
        except Exception as e:
            if attempt < retries and "429" in str(e):
                import re as _re
                m = _re.search(r"try again in (?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", str(e))
                wait = int((int(m.group(1) or 0) * 60) + float(m.group(2))) + 2 if m else delay
                print(f"    rate-limited, retrying in {wait}s…")
                time.sleep(wait)
                delay = min(delay * 2, 120)
            else:
                raise

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_moves(response: str) -> list[int] | None:
    tokens = response.strip().split()
    moves = []
    for token in tokens:
        token = token.strip(".,;:()")
        if token in MOVE_NAME_TO_IDX:
            moves.append(MOVE_NAME_TO_IDX[token])
        else:
            return None
    return moves if moves else None


def validate(state: np.ndarray, moves: list[int]) -> bool:
    cube = Cube(state.copy())
    for m in moves:
        cube.apply_move(m)
    return cube.is_solved()

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def generate_test_cases(distances, n_per_distance, target_distances):
    rng = np.random.default_rng(42)
    cases = []
    for target_d in target_distances:
        found = 0
        for _ in range(200000):
            if found >= n_per_distance:
                break
            cube = Cube()
            scramble = cube.scramble(target_d + 4, rng)
            opt_d = distances[cube.state.tobytes()]
            if opt_d == target_d:
                cases.append((cube.state.copy(), scramble, opt_d))
                found += 1
        if found < n_per_distance:
            print(f"  Warning: only found {found}/{n_per_distance} cases at distance {target_d}")
    return cases


def print_summary(model: str, results: dict, target_distances: list, rep_configs: list):
    print("=" * 72)
    print(f"RESULTS — {model}")
    print("=" * 72)
    header = f"{'Representation':<20}" + "".join(f"  d={d}" for d in target_distances) + "  Total"
    print(header)
    print("-" * 72)
    for rep_name, _, _ in rep_configs:
        res = results[rep_name]
        by_dist: dict[int, list[bool]] = {}
        for d, solved, _ in res:
            by_dist.setdefault(d, []).append(solved)
        row = f"{rep_name:<20}"
        for d in target_distances:
            if d in by_dist:
                k, n = sum(by_dist[d]), len(by_dist[d])
                row += f"  {k}/{n}"
            else:
                row += "   -"
        total_solved = sum(s for _, s, _ in res)
        row += f"   {total_solved}/{len(res)}"
        print(row)
    print("=" * 72)
    print("\nMean solution length vs optimal (solved cases only):")
    for rep_name, _, _ in rep_configs:
        solved_cases = [(sl, od) for od, s, sl in results[rep_name] if s and sl is not None]
        if solved_cases:
            gaps = [sl - od for sl, od in solved_cases]
            print(f"  {rep_name:<20}  mean_len={sum(s for s,_ in solved_cases)/len(solved_cases):.1f}"
                  f"  mean_gap={sum(gaps)/len(gaps):.1f}  (n={len(solved_cases)})")
        else:
            print(f"  {rep_name:<20}  no solved cases")
    print()


def print_comparison(all_results: dict, active_models: list, target_distances: list, rep_configs: list):
    print("=" * 72)
    print("CROSS-MODEL COMPARISON  (move_sequence solve rate)")
    print("=" * 72)
    col_w = max(len(m) for m in active_models) + 2
    header = f"{'Model':<{col_w}}" + "".join(f"  d={d}" for d in target_distances) + "  Total"
    print(header)
    print("-" * 72)
    for model in active_models:
        res = all_results[model]["move_sequence"]
        by_dist: dict[int, list[bool]] = {}
        for d, solved, _ in res:
            by_dist.setdefault(d, []).append(solved)
        row = f"{model:<{col_w}}"
        for d in target_distances:
            if d in by_dist:
                k, n = sum(by_dist[d]), len(by_dist[d])
                row += f"  {k}/{n}"
            else:
                row += "   -"
        total = sum(s for _, s, _ in res)
        row += f"   {total}/{len(res)}"
        print(row)
    print("=" * 72)


_print_lock = threading.Lock()


def _log(model: str, msg: str) -> None:
    with _print_lock:
        prefix = f"[{model}] "
        print("\n".join(prefix + line for line in msg.splitlines()))


def run_model(
    model: str,
    client,
    cases: list,
    rep_configs: list,
    ex_scramble,
    ex_cube,
    ex_sol_str: str,
    target_distances: list,
) -> tuple[str, dict]:
    """Evaluate one model against all cases. Returns (model, results)."""
    results = {name: [] for name, _, _ in rep_configs}

    _log(model, f"{'─'*60}\nStarting {len(cases)} cases\n{'─'*60}")

    for case_idx, (state, scramble_moves, opt_d) in enumerate(cases):
        cube = Cube(state.copy())
        _log(model, f"Case {case_idx+1}/{len(cases)}  d={opt_d}  {cube}")

        for rep_name, formatter, needs_scramble in rep_configs:
            rep_text = formatter(scramble_moves if needs_scramble else cube)
            ex_text  = formatter(ex_scramble    if needs_scramble else ex_cube)
            prompt = (
                "Example:\n\n"
                f"{ex_text}\n\n"
                f"Solution: {ex_sol_str}\n\n"
                "---\n\n"
                "Now solve this cube:\n\n"
                f"{rep_text}"
            )
            try:
                raw = query_model(prompt, model, client)
                moves = parse_moves(raw)
                if moves is None:
                    solved, sol_len = False, None
                    tag = f"PARSE_FAIL  raw={raw[:50]!r}"
                else:
                    solved = validate(state, moves)
                    sol_len = len(moves)
                    tag = f"{'✓ SOLVED' if solved else '✗ WRONG '}  len={sol_len}  optimal={opt_d}"
                results[rep_name].append((opt_d, solved, sol_len))
                _log(model, f"  [{rep_name:<16}] {tag}")
            except Exception as e:
                _log(model, f"  [{rep_name:<16}] ERROR: {e}")
                results[rep_name].append((opt_d, False, None))

            time.sleep(PROVIDER_SLEEP.get(provider_for(model), 4.0))

    _log(model, "Done.")
    return model, results


def load_cache() -> dict:
    if os.path.exists(RESULTS_CACHE):
        import json
        with open(RESULTS_CACHE, encoding="utf-8") as f:
            raw = json.load(f)
        # raw[model][rep_name] = list of [opt_d, solved, sol_len]
        return {
            model: {
                rep: [tuple(r) for r in rows]
                for rep, rows in reps.items()
            }
            for model, reps in raw.items()
        }
    return {}


def save_cache(cache: dict) -> None:
    import json
    os.makedirs(os.path.dirname(RESULTS_CACHE), exist_ok=True)
    with open(RESULTS_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def run(argv=None):
    import argparse as _ap
    parser = _ap.ArgumentParser(description="LLM cube-solving evaluation.")
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated list of models to run (default: all with available API keys).",
    )
    parser.add_argument(
        "--regen-html", action="store_true",
        help="Regenerate the HTML from cached results without running any new queries.",
    )
    args = parser.parse_args(argv)

    model_list = [m.strip() for m in args.models.split(",")] if args.models else MODELS

    # Load previously saved results
    cache = load_cache()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "llm_eval_results.html")
    rep_configs = [
        ("face_grid",       fmt_face_grid,       False),
        ("compact_string",  fmt_compact_string,  False),
        ("corner_cubies",   fmt_corner_cubies,   False),
        ("piece_identity",  fmt_piece_identity,  False),
        ("move_sequence",   fmt_move_sequence,   True),
    ]
    target_distances = [3, 5, 7, 9, 11]

    if args.regen_html:
        if not cache:
            print("No cached results found. Run without --regen-html first.")
            return
        all_models = list(cache.keys())
        write_html_results(cache, all_models, target_distances, rep_configs, out_path)
        print(f"Wrote {out_path}")
        return

    print("Loading BFS distance table...")
    distances = DistanceTable.load(CACHE_PATH)
    print(f"  {len(distances):,} states loaded\n")

    clients = build_clients(model_list)
    active_models = [m for m in model_list if provider_for(m) in clients]
    if not active_models:
        print("No API keys found for any configured model. Set at least one of:")
        for v in KEY_VARS.values():
            print(f"  {v}")
        if cache:
            print("\nExisting cached results for:", ", ".join(cache.keys()))
            all_models = list(cache.keys())
            write_html_results(cache, all_models, target_distances, rep_configs, out_path)
            print(f"Wrote {out_path} from cache")
        return
    print(f"Active models: {', '.join(active_models)}\n")

    # Fixed few-shot example — same for all models
    ex_cube = Cube()
    ex_scramble = ex_cube.scramble(4, np.random.default_rng(7))
    ex_state = ex_cube.state.copy()
    ex_sol_moves = solve(ex_state, distances)
    ex_sol_str = " ".join(MOVE_NAMES[m] for m in ex_sol_moves)
    print(f"Few-shot example:")
    print(f"  Scramble : {' '.join(MOVE_NAMES[m] for m in ex_scramble)}")
    print(f"  Solution : {ex_sol_str}\n")

    print(f"Generating test cases ({N_PER_DISTANCE} per distance level)...")
    cases = generate_test_cases(distances, N_PER_DISTANCE, target_distances)
    print(f"  {len(cases)} cases ready\n")

    new_results: dict[str, dict] = {}

    print(f"Running {len(active_models)} model(s) in parallel...\n")
    with ThreadPoolExecutor(max_workers=len(active_models)) as pool:
        futures = {
            pool.submit(
                run_model,
                model,
                clients[provider_for(model)],
                cases,
                rep_configs,
                ex_scramble,
                ex_cube,
                ex_sol_str,
                target_distances,
            ): model
            for model in active_models
        }
        for future in as_completed(futures):
            model, results = future.result()
            new_results[model] = results
            print_summary(model, results, target_distances, rep_configs)

    if len(active_models) > 1:
        print_comparison(new_results, active_models, target_distances, rep_configs)

    # Merge new results into cache and save
    cache.update(new_results)
    save_cache(cache)
    print(f"Saved results to {RESULTS_CACHE}")

    all_models = list(cache.keys())
    write_html_results(cache, all_models, target_distances, rep_configs, out_path)
    print(f"Wrote {out_path}")


def write_html_results(
    all_results: dict,
    active_models: list,
    target_distances: list,
    rep_configs: list,
    out_path: str,
) -> None:
    import plotly.graph_objects as go

    _DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#1e293b", font=dict(color="#e2e8f0"))
    _GRID = dict(gridcolor="#334155")
    COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

    rep_names = [name for name, _, _ in rep_configs]

    def solve_rates(results):
        """Return (totals[], heatmap_z[], heatmap_text[]) for one model's results."""
        z, text, totals, total_texts = [], [], [], []
        for rep_name in rep_names:
            res = results[rep_name]
            by_dist: dict[int, list] = {}
            for d, solved, _ in res:
                by_dist.setdefault(d, []).append(solved)
            row_z, row_t = [], []
            for d in target_distances:
                if d in by_dist:
                    k, n = sum(by_dist[d]), len(by_dist[d])
                    row_z.append(100.0 * k / n)
                    row_t.append(f"{k}/{n}")
                else:
                    row_z.append(0.0)
                    row_t.append("-")
            z.append(row_z)
            text.append(row_t)
            n_solved = sum(s for _, s, _ in res)
            totals.append(100.0 * n_solved / len(res))
            total_texts.append(f"{n_solved}/{len(res)}")
        return totals, total_texts, z, text

    plotly_loaded = False
    sections = []   # list of (model, html_string)

    # Cross-model comparison bar chart (only if >1 model)
    if len(active_models) > 1:
        fig_cmp = go.Figure()
        for ci, model in enumerate(active_models):
            totals, total_texts, _, _ = solve_rates(all_results[model])
            fig_cmp.add_trace(go.Bar(
                name=model,
                x=rep_names, y=totals,
                text=total_texts, textposition="outside",
                marker_color=COLORS[ci % len(COLORS)],
                hovertemplate=f"{model}<br>%{{x}}: %{{text}} (%{{y:.1f}}%)<extra></extra>",
            ))
        fig_cmp.update_layout(
            title="Cross-model comparison — total solve rate by representation",
            xaxis=dict(title="Representation", **_GRID),
            yaxis=dict(title="Solve rate (%)", range=[0, 115], **_GRID),
            barmode="group", height=420, **_DARK,
        )
        cmp_html = fig_cmp.to_html(full_html=False, include_plotlyjs="cdn")
        plotly_loaded = True
        sections.append(("__comparison__", cmp_html))

    # Per-model sections
    for model in active_models:
        results = all_results[model]
        totals, total_texts, z, text = solve_rates(results)

        fig_heat = go.Figure(go.Heatmap(
            z=z,
            x=[f"d={d}" for d in target_distances],
            y=rep_names,
            colorscale="Blues", zmin=0, zmax=100,
            text=text, texttemplate="%{text}",
            colorbar=dict(title="Solve %"),
            hovertemplate="Rep: %{y}<br>Distance: %{x}<br>Solve rate: %{z:.1f}%<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"{model} — Solve rate (%) by representation × distance",
            xaxis_title="Optimal distance", yaxis_title="Representation",
            height=360, **_DARK,
        )

        fig_bar = go.Figure(go.Bar(
            x=rep_names, y=totals,
            text=total_texts, textposition="outside",
            marker_color="#3b82f6",
            hovertemplate="%{x}<br>%{text} solved (%{y:.1f}%)<extra></extra>",
        ))
        fig_bar.update_layout(
            title=f"{model} — Total solve rate by representation",
            xaxis=dict(title="Representation", **_GRID),
            yaxis=dict(title="Solve rate (%)", range=[0, 115], **_GRID),
            height=360, **_DARK,
        )

        inc = "cdn" if not plotly_loaded else False
        plotly_loaded = True
        model_html = (
            fig_heat.to_html(full_html=False, include_plotlyjs=inc)
            + "\n"
            + fig_bar.to_html(full_html=False, include_plotlyjs=False)
        )
        sections.append((model, model_html))

    # Build tab UI
    model_sections = [(m, h) for m, h in sections if m != "__comparison__"]
    comparison_html = next((h for m, h in sections if m == "__comparison__"), "")

    tab_ids = [f"tab-{i}" for i in range(len(model_sections))]
    tab_buttons = "\n".join(
        f'  <button class="tab-btn{" active" if i==0 else ""}" '
        f'onclick="showTab({i})" id="btn-{i}">{m}</button>'
        for i, (m, _) in enumerate(model_sections)
    )
    tab_panels = "\n".join(
        f'<div class="tab-panel" id="{tid}" style="display:{"block" if i==0 else "none"}">{html}</div>'
        for i, ((m, html), tid) in enumerate(zip(model_sections, tab_ids))
    )

    tab_css = """\
.tab-bar { display: flex; gap: 6px; flex-wrap: wrap; padding: 12px 20px 0; }
.tab-btn { background: #1e293b; border: 1px solid #334155; color: #94a3b8; font-size: 12px;
  padding: 5px 14px; border-radius: 5px; cursor: pointer; font-family: inherit;
  transition: background .15s, color .15s; }
.tab-btn:hover { background: #334155; color: #e2e8f0; }
.tab-btn.active { background: #334155; border-color: #475569; color: #f1f5f9; font-weight: 600; }"""

    tab_js = """\
<script>
function showTab(i) {
  document.querySelectorAll('.tab-panel').forEach((p, j) => p.style.display = j===i?'block':'none');
  document.querySelectorAll('.tab-btn').forEach((b, j) => b.classList.toggle('active', j===i));
}
</script>"""

    nav_css = """\
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
header { background: #1e293b; border-bottom: 1px solid #334155; padding: 10px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
header h1 { font-size: 14px; font-weight: 700; color: #f1f5f9; white-space: nowrap; }
nav { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
nav a { color: #94a3b8; text-decoration: none; font-size: 11px; padding: 4px 10px; border-radius: 5px; background: #0f172a; border: 1px solid #334155; transition: color .15s, background .15s; white-space: nowrap; }
nav a:hover { color: #e2e8f0; background: #334155; }
nav a[aria-current="page"] { color: #f1f5f9; background: #334155; border-color: #475569; }
.nav-group { position: relative; }
.nav-group-btn { color: #94a3b8; font-size: 11px; padding: 4px 10px; border-radius: 5px; background: #0f172a; border: 1px solid #334155; cursor: pointer; white-space: nowrap; font-family: inherit; transition: color .15s, background .15s; }
.nav-group-btn:hover, .nav-group:focus-within .nav-group-btn { color: #e2e8f0; background: #334155; }
.nav-group-btn.active { color: #f1f5f9; background: #334155; border-color: #475569; }
.nav-group-menu { display: none; position: absolute; top: calc(100% + 4px); left: 0; background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 4px; min-width: 220px; z-index: 100; flex-direction: column; gap: 2px; box-shadow: 0 8px 24px rgba(0,0,0,.5); }
.nav-group:hover .nav-group-menu, .nav-group:focus-within .nav-group-menu { display: flex; }
.nav-group-menu a { border-radius: 4px; }
.page-intro { max-width: 860px; padding: 18px 20px 4px; }
.page-intro h2 { font-size: 15px; font-weight: 700; color: #f1f5f9; margin-bottom: 10px; }
.page-intro p { font-size: 13px; color: #94a3b8; line-height: 1.65; margin-bottom: 6px; }
.page-intro strong { color: #cbd5e1; }
""" + tab_css + "\n</style>"

    nav_body = """\
<header>
  <h1>2×2 Cube Interpretability</h1>
  <nav>
    <a href="index.html">Visualizer</a>
    <div class="nav-group">
      <button class="nav-group-btn active">Analyses &#9662;</button>
      <div class="nav-group-menu">
        <a href="probe_results.html">Phase 4 — Probing</a>
        <a href="phase5a_results.html">Phase 5a — Patching</a>
        <a href="phase5b_results.html">Phase 5b — Lenses &amp; SAE</a>
        <a href="circuit_results.html">Phase 7 — Circuit</a>
        <a href="weights_results.html">Phase 8 — Weights</a>
        <a href="grokking_results.html">Phase 9 — Grokking</a>
        <a href="next_move_results.html">Phase 10 — Next Move</a>
        <a href="superposition_results.html">Phase 11 — Superposition</a>
        <a href="corner_model_results.html">Phase 12 — Corner Model</a>
        <a href="corner_attn_results.html">Phase 13 — Head Ablation</a>
        <a href="neuron_profile_results.html">Phase 14 — Neuron Profile</a>
        <a href="corner_circuit_results.html">Phase 15 — Corner Circuit</a>
        <a href="next_move_analysis_results.html">Phase 16 — Next-Move Analysis</a>
        <a href="llm_eval_results.html" aria-current="page">Phase 17 — LLM Eval</a>
      </div>
    </div>
  </nav>
</header>"""

    page_intro = """\
<section class="page-intro">
  <h2>Phase 17 — LLM Evaluation: Which Representation Best Enables Solving?</h2>
  <p><strong>Question:</strong> Which text representation of the cube state gives a general-purpose LLM the best chance of solving it?</p>
  <p><strong>Method:</strong> Five representations (face_grid, compact_string, corner_cubies, piece_identity, move_sequence) are tested at distances 3, 5, 7, 9, and 11 with a one-shot prompt. Solve rate and solution length are measured.</p>
  <p><strong>Baseline:</strong> move_sequence gives the model the scramble moves directly — any model that can reverse a sequence achieves a near-100% ceiling. The other representations require genuine spatial reasoning.</p>
  <p><strong>To add a model:</strong> set its API key env var and run <code style="background:#0f172a;padding:1px 5px;border-radius:3px">uv run python test_representations.py --models MODEL_NAME</code>. Results accumulate across runs.</p>
</section>"""

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        + nav_css + "\n"
        + "<title>Phase 17 — LLM Eval</title>\n"
        "</head>\n<body>\n"
        + nav_body + "\n"
        + page_intro + "\n"
        + comparison_html + "\n"
        + '<div class="tab-bar">\n' + tab_buttons + "\n</div>\n"
        + tab_panels + "\n"
        + tab_js + "\n"
        + "</body>\n</html>\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    run()
