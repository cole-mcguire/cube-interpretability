"""
test_representations.py

Tests 4 text representations of 2x2x2 Rubik's cube states across multiple LLMs
to find which representations enable LLM-based solving.

Representations tested:
  1. face_grid       — 2x2 grid per face, compass layout
  2. compact_string  — 24-char flat string, faces in U/D/F/B/L/R order
  3. corner_cubies   — per-corner piece description (which color faces each direction)
  4. move_sequence   — the scramble sequence itself (degenerate ceiling)

Supported providers (set the corresponding env var to activate):
  OPENAI_API_KEY    — gpt-*, o1-*, o3-*, o4-*
  GEMINI_API_KEY    — gemini-*
  ANTHROPIC_API_KEY — claude-*
  GROQ_API_KEY      — llama-*, mixtral-*, and other Groq-hosted models

Edit MODELS below to choose which models to run.
"""

import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

# ---------------------------------------------------------------------------
# Config — edit these to choose models and test volume
# ---------------------------------------------------------------------------

MODELS = [
    "gpt-5.4",
    # "gpt-5.2",
    # "gpt-4o",
    # "gemini-2.5-pro",
    # "claude-opus-4-7",
    # "llama-3.3-70b-versatile",
]

N_PER_DISTANCE = 10   # test cases per distance level (50 total × 4 reps × n_models queries)

# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube import (
    Cube, DistanceTable, MOVE_NAMES, MOVE_NAME_TO_IDX, CORNER_STICKERS,
    IDX_TO_COLOR, solve,
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
        for _ in range(50000):
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


def run():
    print("Loading BFS distance table...")
    distances = DistanceTable.load(CACHE_PATH)
    print(f"  {len(distances):,} states loaded\n")

    clients = build_clients(MODELS)
    active_models = [m for m in MODELS if provider_for(m) in clients]
    if not active_models:
        print("No API keys found for any configured model. Set at least one of:")
        for v in KEY_VARS.values():
            print(f"  {v}")
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

    target_distances = [3, 5, 7, 9, 11]
    print(f"Generating test cases ({N_PER_DISTANCE} per distance level)...")
    cases = generate_test_cases(distances, N_PER_DISTANCE, target_distances)
    print(f"  {len(cases)} cases ready\n")

    rep_configs = [
        ("face_grid",      fmt_face_grid,      False),
        ("compact_string", fmt_compact_string, False),
        ("corner_cubies",  fmt_corner_cubies,  False),
        ("move_sequence",  fmt_move_sequence,  True),
    ]

    all_results: dict[str, dict] = {}

    for model in active_models:
        print(f"{'─'*72}")
        print(f"Testing model: {model}")
        print(f"{'─'*72}\n")

        client = clients[provider_for(model)]
        results = {name: [] for name, _, _ in rep_configs}

        for case_idx, (state, scramble_moves, opt_d) in enumerate(cases):
            cube = Cube(state.copy())
            print(f"─── Case {case_idx+1}/{len(cases)}  optimal_distance={opt_d}  {cube}")

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
                    print(f"  [{rep_name:<16}] {tag}")
                except Exception as e:
                    print(f"  [{rep_name:<16}] ERROR: {e}")
                    results[rep_name].append((opt_d, False, None))

                time.sleep(4.0)

            print()

        print_summary(model, results, target_distances, rep_configs)
        all_results[model] = results

    if len(active_models) > 1:
        print_comparison(all_results, active_models, target_distances, rep_configs)


if __name__ == "__main__":
    run()
