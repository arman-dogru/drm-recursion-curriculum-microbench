"""End-to-end wiring of run_one / run_grid / tune: real (tiny) training runs."""
import json

import numpy as np
import pytest
import torch

import run_grid
import run_one
import tune
from drm.config import DEV_SEEDS, EVAL_SEEDS, MATCHED_MENU, METHODS, Config, code_hash, inference_grid
from drm.tasks import TASKS, make_task_data
from helpers import tiny

LR_GRID = run_grid.LR_GRID_FULL


def test_run_one_end_to_end(tmp_path):
    cfg = tiny("drm", split="eval")
    out = run_one.run(cfg, tmp_path)
    js, npz = run_one.out_paths(cfg, tmp_path)
    d = json.loads(js.read_text())
    assert d == json.loads(json.dumps(out))
    assert d["meta"]["code_hash"] == code_hash() and d["meta"]["config_hash"] == cfg.config_hash()
    assert set(d["eval"]) == {f"S{s}_k{k}" for s, k in inference_grid()}
    bits = np.load(npz)
    assert set(bits.files) == set(d["eval"])
    for key in bits.files:
        assert len(bits[key]) == cfg.n_eval                      # evaluated on the eval split
        assert bits[key].mean() == pytest.approx(d["eval"][key]["pass1"])
    assert set(d["traces"]) == {f"S{s}_k{k}" for s, k in run_one.TRACE_SCHEDULES}
    assert len(d["traces"]["S16_k1"]["own"]) == 16
    assert len(d["train"]["curve"]) == cfg.curve_points
    assert d["train"]["phi_used"] == cfg.phi_budget


def test_run_one_skip_and_recompute(tmp_path, capsys):
    cfg = tiny("fwd")
    assert run_one.run(cfg, tmp_path) is not None
    assert run_one.run(cfg, tmp_path) is None                    # up to date -> skipped
    js, _ = run_one.out_paths(cfg, tmp_path)
    d = json.loads(js.read_text())
    d["meta"]["code_hash"] = "stale"
    js.write_text(json.dumps(d))
    assert run_one.run(cfg, tmp_path) is not None                # stale code hash -> rerun
    assert run_one.run(cfg, tmp_path, force=True) is not None


def test_run_one_routes_splits(tmp_path, monkeypatch):
    """Eval metrics use the configured split; the learning curve uses dev only."""
    calls = []
    real = run_one.infer

    def spy(model, X, S, k, gen, **kw):
        calls.append(X.clone())
        return real(model, X, S, k, gen, **kw)

    monkeypatch.setattr(run_one, "infer", spy)
    cfg = tiny("drm", split="eval")
    run_one.run(cfg, tmp_path)
    data = make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval, cfg.data_seed)
    dev_c = torch.from_numpy(data["dev"][0][: cfg.curve_n])
    ev = torch.from_numpy(data["eval"][0])
    n_curve = cfg.curve_points
    assert all(torch.equal(c, dev_c) for c in calls[:n_curve])
    assert all(torch.equal(c, ev) for c in calls[n_curve:])
    assert len(calls) == n_curve + len(inference_grid()) + len(run_one.TRACE_SCHEDULES)


def test_cli_parse_defaults():
    cfg, out, force = run_one.parse(["--task", "flood", "--method", "drm", "--seed", "3"])
    assert cfg == Config(task="flood", method="drm", seed=3)
    assert cfg.split == "dev" and cfg.radius == 1 and not force


def test_seed_sets_disjoint():
    assert not set(DEV_SEEDS) & set(EVAL_SEEDS)
    assert len(EVAL_SEEDS) >= 10


def test_tune_jobs_cover_dev_grid_only():
    jobs = run_grid.tune_jobs()
    assert len(jobs) == len(TASKS) * len(METHODS) * len(LR_GRID) * len(DEV_SEEDS)
    assert {j.split for j in jobs} == {"dev"} and {j.seed for j in jobs} == set(DEV_SEEDS)
    assert {(j.task, j.method, j.lr) for j in jobs} == {(t, m, lr) for t in TASKS for m in METHODS for lr in LR_GRID}
    assert {j.k for j in jobs} == {4}


def _tuning(lr_of=lambda t, m: 1e-3):
    return {t: {m: {"lr": lr_of(t, m), "schedule": [4, 4]} for m in METHODS} for t in TASKS}


def test_main_jobs_use_tuned_lr_eval_split_eval_seeds():
    lr_of = lambda t, m: LR_GRID[(TASKS.index(t) + METHODS.index(m)) % 3]   # noqa: E731
    jobs = run_grid.main_jobs(_tuning(lr_of))
    assert len(jobs) == len(TASKS) * len(METHODS) * len(EVAL_SEEDS)
    assert {j.split for j in jobs} == {"eval"} and {j.seed for j in jobs} == set(EVAL_SEEDS)
    assert all(j.lr == lr_of(j.task, j.method) for j in jobs)


def test_ksweep_jobs():
    jobs = run_grid.ksweep_jobs(_tuning())
    assert {j.k for j in jobs} == set(run_grid.KSWEEP_K) and 4 not in run_grid.KSWEEP_K
    assert {j.seed for j in jobs} == set(run_grid.KSWEEP_SEEDS) <= set(EVAL_SEEDS)
    assert {j.split for j in jobs} == {"eval"}
    assert {j.phi_budget for j in jobs} == {Config("t", "drm", 0).phi_budget}   # FLOP-matched across k


def _fake_dev_run(cfg, scores):
    return {"meta": {"code_hash": "c", "config_hash": cfg.config_hash()},
            "eval": {f"S{s}_k{k}": {"S": s, "k_in": k, "pass1": scores(cfg, s, k)} for s, k in inference_grid()}}


def test_tune_choose_picks_best_lr_and_matched_menu_only():
    def scores(cfg, s, k):
        v = 0.1
        if cfg.lr == LR_GRID[2] and (s, k) == (8, 2):
            v = 0.6
        if (s, k) == (8, 8):                  # better but off-menu (64 Phi): must not be chosen
            v = 0.9
        if cfg.method == "fwd" and cfg.lr == LR_GRID[0] and (s, k) == (16, 1):
            v = 0.7 + 0.01 * cfg.seed
        return v
    runs = {c: _fake_dev_run(c, scores) for c in run_grid.tune_jobs()}
    choice, _, by_compute = tune.choose(runs)
    assert choice["flood"]["drm"] == {"lr": LR_GRID[2], "schedule": [8, 2], "dev_pass1": pytest.approx(0.6)}
    assert choice["gravity"]["fwd"]["lr"] == LR_GRID[0] and choice["gravity"]["fwd"]["schedule"] == [16, 1]
    assert choice["gravity"]["fwd"]["dev_pass1"] == pytest.approx(0.7 + 0.01 * np.mean(DEV_SEEDS))
    assert by_compute["flood"]["drm"]["64"]["schedule"] == [8, 8]
    assert by_compute["flood"]["drm"]["16"]["schedule"] == [8, 2]
    assert all(list(map(tuple, [c["schedule"]]))[0] in MATCHED_MENU for t in choice for c in choice[t].values())


def test_tune_refuses_incomplete_or_mixed(tmp_path):
    jobs = run_grid.tune_jobs()[:3]
    for c in jobs[:2]:
        js, _ = run_one.out_paths(c, tmp_path)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(_fake_dev_run(c, lambda *a: 0.0)))
    with pytest.raises(tune.GridError, match="missing"):
        tune.load_runs(jobs, tmp_path)
    js, _ = run_one.out_paths(jobs[2], tmp_path)
    d = _fake_dev_run(jobs[2], lambda *a: 0.0)
    d["meta"]["code_hash"] = "other"
    js.write_text(json.dumps(d))
    with pytest.raises(tune.GridError, match="mixed"):
        tune.load_runs(jobs, tmp_path)
    d["meta"]["code_hash"] = "c"
    d["meta"]["config_hash"] = "wrong"
    js.write_text(json.dumps(d))
    with pytest.raises(tune.GridError, match="config"):
        tune.load_runs(jobs, tmp_path)
