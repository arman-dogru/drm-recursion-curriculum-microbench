"""Aggregation wiring on a fabricated eval grid with known answers."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aggregate as A
import run_grid
import run_one
from drm.config import METHODS, inference_grid
from run_grid import MAIN_SEEDS as EVAL_SEEDS, MAIN_TASKS
from drm.tasks import TASKS
from tune import GridError

ROOT = Path(__file__).resolve().parents[1]
SCHED = {m: s for m, s in zip(METHODS, ([16, 1], [16, 1], [1, 16], [2, 8], [4, 4]))}
BASE = {"fwd": 0.30, "fwd_steps": 0.20, "bwd_k": 0.25, "bwd_tbptt": 0.40, "drm": 0.50}


def value(cfg, s, k):
    """Chosen schedule carries the signal; every other schedule is a decoy."""
    if [s, k] != SCHED[cfg.method]:
        return 0.99 if cfg.method != "drm" else 0.0
    v = BASE[cfg.method] + 0.01 * TASKS.index(cfg.task) + 0.001 * (cfg.seed - 100)
    if cfg.method == "bwd_tbptt" and cfg.task == "flood":
        v = 0.5 + 0.001 * (cfg.seed - 100) + (0.02 if cfg.seed % 2 else -0.02) + 0.01 * TASKS.index("flood")
    return v


def fake_run(cfg, code="c"):
    trace = [{"step": i, "cell_acc": 0.5, "exact": 0.1, "visible_frac": 0.1,
              "visible_wrong_frac": 0.2, "self_correct_rate": float("nan") if i == 1 else 0.3}
             for i in range(1, 5)]
    return {"meta": {"code_hash": code, "config_hash": cfg.config_hash(), "config": {"split": cfg.split},
                     "n_params": 123},
            "train": {"curve": [[1, 10, 0.1], [2, 20, 0.2]], "train_seconds": 1.0},
            "eval": {f"S{s}_k{k}": {"S": s, "k_in": k, "pass1": value(cfg, s, k)} for s, k in inference_grid()},
            "traces": {"S16_k1": {"own": trace, "oracle": trace}}}


def write(cfg, raw, d=None):
    js, npz = run_one.out_paths(cfg, raw)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(d or fake_run(cfg)))
    np.savez(npz, a=np.zeros(1))
    return js


@pytest.fixture
def grid(tmp_path, monkeypatch):
    tuning = {t: {m: {"lr": 1e-3, "schedule": SCHED[m]} for m in METHODS} for t in TASKS}
    by_compute = {t: {m: {"16": {"schedule": SCHED[m]}} for m in METHODS} for t in TASKS}
    tp = tmp_path / "tuning.json"
    tp.write_text(json.dumps({"meta": {"code_hash": "c"}, "choice": tuning, "by_compute": by_compute}))
    monkeypatch.setattr(A, "TUNING_PATH", tp)
    raw = tmp_path / "raw"
    jobs = run_grid.main_jobs(tuning) + run_grid.ksweep_jobs(tuning)
    for c in jobs:
        write(c, raw)
    return raw, tmp_path / "out", tuning, jobs


def test_primary_uses_dev_chosen_schedule_and_pools(grid):
    raw, out, _, _ = grid
    s = A.main(raw, out)
    for t in MAIN_TASKS:
        for m in METHODS:
            if m == "bwd_tbptt" and t == "flood":
                continue
            exp = BASE[m] + 0.01 * TASKS.index(t) + 0.001 * np.mean(np.array(EVAL_SEEDS) - 100)
            assert s["primary"][t][m]["mean"] == pytest.approx(exp)
    pooled = np.mean([s["primary"][t]["drm"]["mean"] for t in MAIN_TASKS])
    assert s["primary"]["pooled"]["drm"]["mean"] == pytest.approx(pooled)
    assert s["primary"]["pooled"]["drm"]["n_seeds"] == len(EVAL_SEEDS)
    assert (out / "tables" / "primary.csv").exists() and (out / "tables" / "tests.csv").exists()


def test_paired_tests_and_holm_family(grid):
    raw, out, _, _ = grid
    s = A.main(raw, out)
    T = s["tests"]
    assert s["meta"]["holm_family_size"] == len(T) == 4 * (len(MAIN_TASKS) + 1)
    r = T["gravity:drm-fwd"]
    assert r["diff"] == pytest.approx(0.2) and r["wins"] == len(EVAL_SEEDS)
    assert r["p"] == pytest.approx(2 / 2 ** len(EVAL_SEEDS)) and r["significant"]
    assert r["p_holm"] == pytest.approx(len(T) * r["p"])          # one Holm family over all tests
    # flood vs tbptt: paired diff alternates +-0.02 around 0 -> not significant
    f = T["flood:drm-bwd_tbptt"]
    assert f["diff"] == pytest.approx(0.0, abs=1e-9) and not f["significant"] and f["p_holm"] >= f["p"]


def test_ksweep_uses_native_schedule_and_ksweep_seeds(grid):
    raw, out, _, _ = grid
    s = A.main(raw, out)
    ks = s["ksweep"]["flood"]["drm"]
    assert set(ks) == {"1", "2", "4", "8"}
    assert all(ks[k]["native"]["n_seeds"] == len(run_grid.KSWEEP_SEEDS) for k in ks)
    assert ks["4"]["native"]["mean"] == pytest.approx(
        BASE["drm"] + 0.01 * TASKS.index("flood") + 0.001 * np.mean(np.array(run_grid.KSWEEP_SEEDS) - 100))
    assert ks["8"]["native"]["mean"] == 0.0           # S2_k8 is a decoy (0.0) for drm


def test_refuses_missing(grid):
    raw, out, _, jobs = grid
    run_one.out_paths(jobs[5], raw)[0].unlink()
    with pytest.raises(GridError, match="incomplete"):
        A.main(raw, out)
    s = A.main(raw, out, allow_partial=True)
    assert s["meta"]["partial"] and len(s["meta"]["missing"]) == 1


def test_refuses_extra_file(grid):
    raw, out, _, jobs = grid
    c = jobs[0]
    import dataclasses
    write(dataclasses.replace(c, seed=999), raw)
    with pytest.raises(GridError, match="unexpected"):
        A.main(raw, out)


def test_refuses_mixed_code_and_stale_config(grid):
    raw, out, _, jobs = grid
    write(jobs[3], raw, fake_run(jobs[3], code="other"))
    with pytest.raises(GridError, match="mixed"):
        A.main(raw, out)
    d = fake_run(jobs[3])
    d["meta"]["config_hash"] = "stale"
    write(jobs[3], raw, d)
    with pytest.raises(GridError, match="stale"):
        A.main(raw, out)


def test_refuses_dev_split_in_eval_grid(grid):
    raw, out, _, jobs = grid
    d = fake_run(jobs[2])
    d["meta"]["config"]["split"] = "dev"
    write(jobs[2], raw, d)
    with pytest.raises(GridError, match="non-eval"):
        A.main(raw, out)


def test_cli_allow_partial_must_not_overwrite_results():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate.py"), "--allow-partial"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "separate --out" in r.stderr


def test_compare_is_paired_by_seed():
    seeds = list(EVAL_SEEDS)
    base = {s: 0.05 * i for i, s in enumerate(seeds)}              # strong seed effect
    ps = {"g": {"drm": {s: base[s] + 0.01 for s in seeds}}}
    for b in A.BASELINES:
        ps["g"][b] = dict(base)
    r = A.compare(ps, ["g"])["g:drm-fwd"]
    assert r["wins"] == len(seeds) and r["losses"] == 0
    assert r["lo"] == pytest.approx(0.01) and r["hi"] == pytest.approx(0.01)
    assert r["p"] == pytest.approx(2 / 2 ** len(seeds))


def test_pooled3_traces_at_choice_and_cpu_hours(grid):
    raw, out, tuning, jobs = grid
    s = A.main(raw, out)
    p3 = np.mean([s["primary"][t]["drm"]["mean"] for t in A.DEPTH_TASKS])
    assert s["primary"]["pooled3"]["drm"]["mean"] == pytest.approx(p3)
    assert not any(k.startswith("pooled3") for k in s["tests"])        # descriptive only, not in Holm family
    assert set(A.DEPTH_TASKS) < set(MAIN_TASKS) == set(TASKS)
    # the fabricated traces only exist for S16_k1 -> only methods whose dev schedule is S16_k1 appear
    tac = s["traces_at_choice"]["flood"]
    assert set(tac) == {m for m in METHODS if SCHED[m] == [16, 1]}
    assert all(v["schedule"] == [16, 1] for v in tac.values())
    n_eval = len(run_grid.main_jobs(tuning))
    assert s["cpu_hours"]["eval"] == pytest.approx(n_eval * 1.0 / 3600)
    r = s["primary"]["gravity"]["drm"]
    assert r["t_lo"] <= r["mean"] <= r["t_hi"]
