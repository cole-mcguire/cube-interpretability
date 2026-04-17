"""
Dataset generation, persistence, and loading for the cube interpretability project.

Usage:
    uv run cube-dataset                  # generate with defaults → data/
    uv run cube-dataset --out my_data    # custom output directory
    uv run cube-dataset --help

Splits saved as:
    <out>/train.npz   (~300k samples, 50k sequences)
    <out>/val.npz     (~30k samples,  5k sequences)
    <out>/test.npz    (~30k samples,  5k sequences)

Each .npz contains:
    states:           float32 (N, 144)  — one-hot encoded cube state
    next_moves:       int64   (N,)      — move index to predict (0–17)
    scramble_depth:   int32   (N,)      — steps already applied from solved
    face_solved:      bool    (N, 6)    — per-face solved status
    corner_oriented:  bool    (N, 8)    — per-corner U/D orientation
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pickle

import numpy as np

from cube import MOVE_NAMES, NUM_MOVES, compute_optimal_distances, generate_dataset

_DISTANCES: dict | None = None
_DISTANCES_CACHE = Path("data/distances_cache.pkl")


def _get_distances() -> dict:
    """Return the BFS distance table, computing and caching it on first call."""
    global _DISTANCES
    if _DISTANCES is not None:
        return _DISTANCES
    if _DISTANCES_CACHE.exists():
        print(f"Loading cached distances ({_DISTANCES_CACHE})...", end=" ", flush=True)
        with open(_DISTANCES_CACHE, "rb") as f:
            _DISTANCES = pickle.load(f)
        print(f"{len(_DISTANCES):,} states")
    else:
        print("Computing optimal distances via BFS (~25 min, cached afterward)...")
        _DISTANCES = compute_optimal_distances(verbose=True)
        _DISTANCES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DISTANCES_CACHE, "wb") as f:
            pickle.dump(_DISTANCES, f)
        print(f"Cached to {_DISTANCES_CACHE}")
    return _DISTANCES

# ---------------------------------------------------------------------------
# Split sizes
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_SEQUENCES = 50_000
DEFAULT_VAL_SEQUENCES   =  5_000
DEFAULT_TEST_SEQUENCES  =  5_000
DEFAULT_MAX_DEPTH       = 11
DEFAULT_OUT_DIR         = "data"

SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def generate_split(
    n_sequences: int,
    max_scramble_depth: int = DEFAULT_MAX_DEPTH,
    seed: int = 0,
) -> dict:
    """Generate one dataset split with optimal distances included."""
    rng = np.random.default_rng(seed)
    return generate_dataset(
        n_sequences,
        max_scramble_depth=max_scramble_depth,
        rng=rng,
        distances=_get_distances(),
    )


def save_split(data: dict, path: str | Path) -> None:
    """Save a split dict to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_split(path: str | Path) -> dict:
    """Load a split from a .npz file. Returns a plain dict of arrays."""
    raw = np.load(path)
    return {key: raw[key] for key in raw.files}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(data: dict, name: str) -> None:
    n = data["states"].shape[0]
    depths = data["scramble_depth"]
    moves = data["next_moves"]
    face_solved = data["face_solved"]
    corner_oriented = data["corner_oriented"]

    print(f"\n── {name} ({'─' * (40 - len(name))})")
    print(f"  samples:          {n:>10,}")
    print(f"  depth   min/mean/max:  {depths.min()} / {depths.mean():.2f} / {depths.max()}")

    # Move class balance: min and max frequency, and whether it's roughly uniform
    counts = np.bincount(moves, minlength=NUM_MOVES)
    freq = counts / counts.sum()
    print(f"  move freq  min/max:  {freq.min():.3f} / {freq.max():.3f}  "
          f"(uniform = {1/NUM_MOVES:.3f})")

    # Most/least common moves
    most  = int(counts.argmax())
    least = int(counts.argmin())
    print(f"  most common move:   {MOVE_NAMES[most]:>3}  ({counts[most]:,})")
    print(f"  least common move:  {MOVE_NAMES[least]:>3}  ({counts[least]:,})")

    # Probe label prevalence
    print(f"  faces solved (mean per face):  "
          f"{face_solved.mean(axis=0).round(3).tolist()}")
    print(f"  corners oriented (mean):       "
          f"{corner_oriented.mean():.3f}")

    # Optimal distance distribution
    if "optimal_distance" in data:
        opt = data["optimal_distance"]
        counts = np.bincount(opt.astype(np.int64), minlength=12)
        dist_str = "  ".join(f"{d}:{counts[d]:,}" for d in range(12) if counts[d] > 0)
        print(f"  optimal distance dist:  {dist_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for the cube interpretability project."
    )
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_SEQUENCES,
                        metavar="N", help=f"Train sequences (default: {DEFAULT_TRAIN_SEQUENCES:,})")
    parser.add_argument("--val", type=int, default=DEFAULT_VAL_SEQUENCES,
                        metavar="N", help=f"Val sequences (default: {DEFAULT_VAL_SEQUENCES:,})")
    parser.add_argument("--test", type=int, default=DEFAULT_TEST_SEQUENCES,
                        metavar="N", help=f"Test sequences (default: {DEFAULT_TEST_SEQUENCES:,})")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                        metavar="D", help=f"Max scramble depth (default: {DEFAULT_MAX_DEPTH})")
    args = parser.parse_args()

    out = Path(args.out)
    splits = [
        ("train", args.train,  SPLIT_SEEDS["train"]),
        ("val",   args.val,    SPLIT_SEEDS["val"]),
        ("test",  args.test,   SPLIT_SEEDS["test"]),
    ]

    print(f"Generating dataset → {out}/")
    print(f"  train: {args.train:,} sequences  |  val: {args.val:,}  |  test: {args.test:,}")
    print(f"  max scramble depth: {args.max_depth}")

    total_start = time.perf_counter()
    for split_name, n_seq, seed in splits:
        t0 = time.perf_counter()
        print(f"\nGenerating {split_name}...", end=" ", flush=True)
        data = generate_split(n_seq, max_scramble_depth=args.max_depth, seed=seed)
        elapsed = time.perf_counter() - t0
        n = data["states"].shape[0]
        print(f"{n:,} samples  ({elapsed:.1f}s)")

        path = out / f"{split_name}.npz"
        save_split(data, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  saved → {path}  ({size_mb:.1f} MB)")
        print_stats(data, split_name)

    total = time.perf_counter() - total_start
    print(f"\nDone in {total:.1f}s. Load with: dataset.load_split('{out}/train.npz')")


if __name__ == "__main__":
    main()
