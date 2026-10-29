import itertools

import numpy as np
import pytest

from drm.stats import bootstrap_ci, holm, sign_flip_p


def brute(d):
    obs = abs(np.mean(d))
    c = 0
    for s in itertools.product((1, -1), repeat=len(d)):
        c += abs(np.mean(np.array(s) * d)) >= obs - 1e-12
    return c / 2 ** len(d)


def test_sign_flip_exact_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.normal(0.1, 1, size=7)
        assert sign_flip_p(d) == pytest.approx(brute(d))


def test_sign_flip_all_positive_minimum():
    assert sign_flip_p(np.ones(5)) == pytest.approx(2 / 32)
    assert sign_flip_p(np.arange(1, 13) / 10) == pytest.approx(2 / 4096)


def test_sign_flip_zero_effect():
    assert sign_flip_p(np.zeros(6)) == 1.0


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.2})["x"] == pytest.approx(0.2)


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert m == 2.5 and lo < 2.5 < hi and lo >= 1 and hi <= 4
    m, lo, hi = bootstrap_ci(np.full(5, 0.3))
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)
