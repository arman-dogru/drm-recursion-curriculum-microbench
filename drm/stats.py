"""Seed-level statistics: paired exact sign-flip test, cluster bootstrap CI, Holm.

The unit of replication is the training seed. Methods trained with the same
seed share initial weights' RNG seed, batch order RNG and inference RNG, so
comparisons are paired by seed.
"""
from __future__ import annotations

import itertools

import numpy as np


def sign_flip_p(diffs: np.ndarray, max_exact: int = 16, n_mc: int = 200_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(diffs) != 0.

    Exact enumeration for n <= max_exact, Monte Carlo otherwise. Ties with the
    observed statistic count as at-least-as-extreme (conservative).
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    if n <= max_exact:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((1.0, -1.0), size=(n_mc, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float(np.mean(stats >= obs - 1e-12))


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean, resampling seeds (clusters) only."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone)."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def t_ci(values: np.ndarray, alpha: float = 0.05):
    """Student-t CI of the mean over seeds. Reported next to the percentile bootstrap,
    which under-covers at n = 12 (its interval is about sqrt((n-1)/n) * z/t as wide)."""
    from scipy import stats as st
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2 or np.all(v == v[0]):
        return m, m, m
    half = float(st.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - half, m + half
