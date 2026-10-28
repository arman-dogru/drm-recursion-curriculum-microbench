import numpy as np
import pytest

from drm import tasks
from drm.tasks import H, W, M, TASKS, make_task_data, reachable


@pytest.mark.parametrize("task", TASKS)
def test_deterministic_and_shapes(task):
    a = make_task_data(task, 50, 20, 30, data_seed=0)
    b = make_task_data(task, 50, 20, 30, data_seed=0)
    for s in ("train", "dev", "eval"):
        assert np.array_equal(a[s][0], b[s][0]) and np.array_equal(a[s][1], b[s][1])
        assert a[s][0].shape[1] == M and a[s][1].shape[1] == M
        assert a[s][1].min() >= 0 and a[s][1].max() < tasks.N_COLORS


@pytest.mark.parametrize("task", TASKS)
def test_splits_disjoint_at_data_level(task):
    d = make_task_data(task, 200, 100, 100, data_seed=0)
    k = {s: tasks.keys(d[s][0]) for s in d}
    assert len(k["train"]) == 200 and len(k["dev"]) == 100 and len(k["eval"]) == 100
    assert not (k["train"] & k["dev"]) and not (k["train"] & k["eval"]) and not (k["dev"] & k["eval"])


def test_colormap_and_shift_exact():
    X, Y = make_task_data("colormap", 20, 5, 5)["train"]
    assert np.array_equal(tasks.COLORMAP_PERM[X], Y)
    X, Y = make_task_data("shift", 20, 5, 5)["train"]
    for x, y in zip(X, Y):
        assert np.array_equal(np.roll(x.reshape(H, W), (tasks.SHIFT_DR, tasks.SHIFT_DC), (0, 1)), y.reshape(H, W))


def test_symfill_symmetric_and_recoverable():
    X, Y = make_task_data("symfill", 100, 5, 5)["train"]
    for x, y in zip(X, Y):
        x, y = x.reshape(H, W), y.reshape(H, W)
        assert np.array_equal(y, y[:, ::-1])
        occ = x == tasks.OCCLUDER
        assert occ.any()
        # every occluded cell has a visible mirror
        assert not (occ & occ[:, ::-1]).any()
        assert np.array_equal(x[~occ], y[~occ])


def test_gravity_preserves_columns():
    X, Y = make_task_data("gravity", 100, 5, 5)["train"]
    for x, y in zip(X, Y):
        x, y = x.reshape(H, W), y.reshape(H, W)
        for c in range(W):
            assert list(x[:, c][x[:, c] > 0]) == list(y[:, c][y[:, c] > 0])
            nz = np.nonzero(y[:, c])[0]
            assert len(nz) == 0 or nz.min() == H - len(nz)


def test_flood_matches_bfs_and_needs_depth():
    X, Y = make_task_data("flood", 200, 5, 5)["train"]
    far = 0
    for x, y in zip(X, Y):
        x, y = x.reshape(H, W), y.reshape(H, W)
        src = tuple(np.argwhere(x == 2)[0])
        reach = reachable(x == 1, src)
        exp = np.where(x == 1, 1, np.where(reach, 3, 0))
        exp[src] = 2
        assert np.array_equal(exp, y)
        cells = np.argwhere(y == 3)
        if len(cells) and np.abs(cells - np.array(src)).max(1).max() > 2:
            far += 1
    assert far > 50   # many examples need information to travel > 2 cells


def test_reachable_simple():
    walls = np.zeros((H, W), dtype=bool)
    walls[:, 2] = True
    r = reachable(walls, (0, 0))
    assert r[:, :2].all() and not r[:, 2:].any()
