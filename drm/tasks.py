"""Synthetic 8x8 grid tasks with exact, deterministic targets.

Every task maps an input grid X (values 0..7) to a unique output grid Y
(values 0..7). Colour 0 is background. Generators are pure functions of a
numpy Generator, so a (task, split, data_seed) triple fully determines a split.
"""
from __future__ import annotations

from collections import deque

import numpy as np

H = W = 8
M = H * W
N_COLORS = 8          # output/input colours 0..7
MASK = N_COLORS       # extra label token used only in the (corrupted) target stream
TASKS = ("colormap", "shift", "symfill", "gravity", "flood")

# fixed task constants (part of the task definition, not tuned)
COLORMAP_PERM = np.array([0, 3, 5, 1, 6, 2, 4, 7])
SHIFT_DR, SHIFT_DC = 1, 2
OCCLUDER = 7


def _colormap(rng):
    x = rng.integers(0, 7, size=(H, W))
    return x, COLORMAP_PERM[x]


def _shift(rng):
    x = np.zeros((H, W), dtype=np.int64)
    for _ in range(rng.integers(1, 4)):
        h, w = rng.integers(1, 4, size=2)
        r, c = rng.integers(0, H), rng.integers(0, W)
        col = rng.integers(1, 7)
        for i in range(h):
            for j in range(w):
                x[(r + i) % H, (c + j) % W] = col
    y = np.roll(x, shift=(SHIFT_DR, SHIFT_DC), axis=(0, 1))
    return x, y


def _symfill(rng):
    left = rng.integers(1, 7, size=(H, W // 2))
    y = np.concatenate([left, left[:, ::-1]], axis=1)
    x = y.copy()
    h, w = rng.integers(2, 5), rng.integers(2, 5)
    r = rng.integers(0, H - h + 1)
    side = rng.integers(0, 2)  # occluder stays inside one half -> mirror always visible
    c0 = rng.integers(0, W // 2 - w + 1) + side * (W // 2)
    x[r:r + h, c0:c0 + w] = OCCLUDER
    return x, y


def _gravity(rng):
    x = np.where(rng.random((H, W)) < 0.3, rng.integers(1, 7, size=(H, W)), 0)
    y = np.zeros_like(x)
    for c in range(W):
        col = x[:, c][x[:, c] > 0]
        if len(col):
            y[H - len(col):, c] = col
    return x, y


def reachable(walls: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """4-connected BFS over non-wall cells from src."""
    seen = np.zeros_like(walls, dtype=bool)
    q = deque([src])
    seen[src] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not walls[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _flood(rng):
    walls = rng.random((H, W)) < 0.35
    free = np.argwhere(~walls)
    src = tuple(free[rng.integers(len(free))])
    reach = reachable(walls, src)
    x = np.where(walls, 1, 0)
    x[src] = 2
    y = np.where(walls, 1, np.where(reach, 3, 0))
    y[src] = 2
    return x, y


GENERATORS = {"colormap": _colormap, "shift": _shift, "symfill": _symfill,
              "gravity": _gravity, "flood": _flood}

# split -> data-seed offset; train/dev/eval example sets are generated from
# different streams and then de-duplicated against each other.
SPLIT_OFFSET = {"train": 0, "dev": 1_000_003, "eval": 2_000_029}


def make_split(task: str, split: str, n: int, data_seed: int = 0,
               exclude: set[bytes] | None = None):
    """Return (X, Y) int64 arrays of shape (n, M), unique and disjoint from `exclude`."""
    rng = np.random.default_rng([data_seed, SPLIT_OFFSET[split], TASKS.index(task)])
    gen = GENERATORS[task]
    xs, ys, seen = [], [], set(exclude or ())
    tries = 0
    while len(xs) < n:
        tries += 1
        if tries > 50 * n:
            raise RuntimeError(f"could not generate {n} unique {task} examples")
        x, y = gen(rng)
        key = x.astype(np.int8).tobytes()
        if key in seen:
            continue
        seen.add(key)
        xs.append(x.reshape(-1))
        ys.append(y.reshape(-1))
    return np.stack(xs).astype(np.int64), np.stack(ys).astype(np.int64)


def keys(X: np.ndarray) -> set[bytes]:
    return {row.astype(np.int8).tobytes() for row in X}


def make_task_data(task: str, n_train: int, n_dev: int, n_eval: int, data_seed: int = 0):
    """Disjoint train/dev/eval splits for one task."""
    tr = make_split(task, "train", n_train, data_seed)
    dv = make_split(task, "dev", n_dev, data_seed, exclude=keys(tr[0]))
    ev = make_split(task, "eval", n_eval, data_seed, exclude=keys(tr[0]) | keys(dv[0]))
    return {"train": tr, "dev": dv, "eval": ev}
