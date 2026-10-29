"""Round-2 review: tests that kill the reviewer's surviving mutants (Y5, Y7, Y10, Y11,
Y13, Y14), plus the <=16 headline menu (major 1) and run-name/overwrite guards (m5)."""
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

import aggregate as A
import run_one
import tune
from drm.config import LR_GRID, MATCHED_MENU, Config, inference_grid
from drm.stats import sign_flip_p
from helpers import tiny

ROOT = Path(__file__).resolve().parents[1]


# ---------- major 1: headline menu is "at most 16 Phi", ties keep the 16-Phi schedule

def test_headline_menu_is_every_schedule_with_at_most_16_phi():
    menu = tune.HEADLINE_MENU
    assert set(menu) == {(s, k) for s, k in inference_grid() if s * k <= 16}
    assert len(menu) == len(set(menu))
    assert tuple(menu[:len(MATCHED_MENU)]) == tuple(MATCHED_MENU)     # ties keep the 16-Phi choice
    assert (1, 4) in menu and (1, 32) not in menu


def _runs(score):
    cfg = Config("flood", "bwd_k", 1)
    return {dataclasses.replace(cfg, seed=s, lr=lr): {"meta": {}, "eval": {
        f"S{a}_k{b}": {"S": a, "k_in": b, "pass1": score(a, b, lr)} for a, b in inference_grid()}}
        for s in (1, 2) for lr in LR_GRID}


def test_tune_can_stop_early_but_never_exceeds_16():
    # best overall at 64 Phi (not allowed), best allowed at S1 k4 (4 Phi)
    ch, _, _ = tune.choose(_runs(lambda a, b, lr: {(1, 64): 0.99, (1, 4): 0.8}.get((a, b), 0.1)))
    assert ch["flood"]["bwd_k"]["schedule"] == [1, 4]
    # exact tie between a 16-Phi and a cheaper schedule -> the 16-Phi schedule is kept
    ch, _, _ = tune.choose(_runs(lambda a, b, lr: 0.5 if (a, b) in ((4, 4), (1, 4)) else 0.1))
    assert ch["flood"]["bwd_k"]["schedule"] == [4, 4]


# ---------- Y5: lower lr edge is flagged too

def test_lower_lr_edge_is_flagged():
    sc = {f"{lr:g}": {f"S{s}_k{k}": 0.0 for s, k in tune.HEADLINE_MENU} for lr in LR_GRID}
    sc[f"{LR_GRID[0]:g}"]["S4_k4"] = 0.5
    sc[f"{LR_GRID[1]:g}"]["S4_k4"] = 0.4
    choice = {"gravity": {"fwd": {"lr": LR_GRID[0], "dev_pass1": 0.5}}}
    assert tune.lr_edges(choice, {"gravity": {"fwd": sc}}) == [f"gravity:fwd:{LR_GRID[0]:g}"]


# ---------- Y7: started_at is taken at the start, finished_at at the end

def test_started_at_is_recorded_before_training(tmp_path, monkeypatch):
    clock = iter(f"2026-01-01T00:00:{i:02d}-04:00" for i in range(60))
    calls = []

    def fake_now():
        v = next(clock)
        calls.append(v)
        return v
    monkeypatch.setattr(run_one, "now_iso", fake_now)
    out = run_one.run(tiny("drm", split="eval"), tmp_path)
    m = out["meta"]
    assert m["started_at"] == calls[0] and m["finished_at"] == calls[-1]
    assert m["started_at"] < m["finished_at"]
    assert m["wall_seconds"] >= m["eval_seconds"] > 0


# ---------- Y10: significance is decided by the Holm-adjusted p, never the raw p

def test_significance_uses_holm_adjusted_p():
    seeds = range(12)
    diff = np.array([3, 2, 4, 1, 2, 3, 5, 2, 1, 3, -1, -1], float)   # raw p < 0.05
    ps = {}
    for g in ("t1", "t2", "t3", "t4", "t5", "pooled"):
        ps[g] = {A.TARGET: {s: 10.0 + diff[s] for s in seeds}}
        for b in A.BASELINES:
            ps[g][b] = {s: 10.0 for s in seeds}
    T = A.compare(ps, list(ps))
    assert len(T) == 6 * len(A.BASELINES)
    for r in T.values():
        assert r["p"] == pytest.approx(sign_flip_p(diff)) and r["p"] < 0.05
        assert r["p_holm"] >= 0.05 and r["significant"] is False
        assert r["significant"] == (r["p_holm"] < 0.05)


# ---------- Y11: CPU-hours are train + eval seconds

def test_cpu_hours_sum_train_and_eval_seconds(tmp_path):
    d = {"train": {"train_seconds": 3600.0}, "meta": {"eval_seconds": 1800.0,
                                                    "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T01:30:00"}}
    out = A.cpu_hours(tmp_path, {"dev": {1: d, 2: d}, "eval": {3: d}})
    assert out["dev"] == pytest.approx(3.0) and out["eval"] == pytest.approx(1.5)
    assert out["total"] == pytest.approx(4.5)


# ---------- Y13: every inference schedule gets its own generator

def test_each_schedule_gets_a_distinct_generator(tmp_path, monkeypatch):
    seen = {}
    real = run_one.infer

    def spy(model, X, S, k_in, gen, **kw):
        if "trace" not in kw:
            seen[(S, k_in)] = gen.initial_seed()
        return real(model, X, S, k_in, gen, **kw)
    monkeypatch.setattr(run_one, "infer", spy)
    run_one.run(tiny("drm", split="eval"), tmp_path)
    grid = [g for g in seen if g in set(inference_grid())]
    assert len(grid) == len(inference_grid())
    assert len({seen[g] for g in grid}) == len(grid)
    assert run_one.eval_gen(100, 1, 16).initial_seed() != run_one.eval_gen(100, 16, 1).initial_seed()


# ---------- Y14 / m2: pipeline commits ALL dev outputs, fails hard, and pushes before eval

def test_pipeline_commits_dev_outputs_before_eval():
    sh = (ROOT / "scripts" / "pipeline.sh").read_text()
    assert sh.startswith("#!/bin/sh") and "\nset -e\n" in sh
    block = sh[sh.index('if [ "${COMMIT:-0}" = 1 ]'):sh.index("stage main")]
    add = re.search(r"^\s*git add (.+)$", block, re.M).group(1).split()
    assert "results/tuning.json" in add and "results/raw/dev" in add
    assert "|| true" not in block and "git commit" in block and "git push" in block
    assert "git diff --quiet HEAD -- results/tuning.json results/raw/dev" in block
    assert sh.index("git push") < sh.index("run_grid.py main")


# ---------- m5: run names carry budget and radius; other non-default fields add a hash;
# ---------- an output with a different config is never silently overwritten; radius >= 0

def test_run_name_includes_budget_and_radius():
    a = Config("shift", "drm", 1, lr=0.016)
    names = {a.run_name(), dataclasses.replace(a, phi_budget=8).run_name(),
             dataclasses.replace(a, radius=0).run_name(), dataclasses.replace(a, radius=2).run_name(),
             dataclasses.replace(a, d=32).run_name(), dataclasses.replace(a, n_train=10).run_name(),
             dataclasses.replace(a, n_sup=2).run_name()}
    assert len(names) == 7
    assert "_b12000_" in a.run_name() and "_r1_" in a.run_name()


def test_names_are_unique_over_all_fields():
    a = Config("shift", "drm", 1)
    seen = {a.run_name(): a.config_hash()}
    for f in dataclasses.fields(Config):
        if f.name in ("task", "method", "split"):
            continue
        v = getattr(a, f.name)
        b = dataclasses.replace(a, **{f.name: v * 2 if v else 1})
        assert b.run_name() not in seen, f.name
        seen[b.run_name()] = b.config_hash()


def test_run_refuses_to_overwrite_a_different_config(tmp_path):
    cfg = tiny("drm")
    run_one.run(cfg, tmp_path)
    js, _ = run_one.out_paths(cfg, tmp_path)
    d = json.loads(js.read_text())
    d["meta"]["config_hash"] = "someoneelse"
    js.write_text(json.dumps(d))
    before = js.read_text()
    with pytest.raises(run_one.OverwriteError):
        run_one.run(cfg, tmp_path)
    assert js.read_text() == before
    run_one.run(cfg, tmp_path, force=True)
    assert json.loads(js.read_text())["meta"]["config_hash"] == cfg.config_hash()


def test_negative_radius_rejected():
    with pytest.raises(ValueError):
        Config("flood", "drm", 1, radius=-3)
    with pytest.raises(SystemExit) as e:
        run_one.parse(["--task", "flood", "--method", "drm", "--seed", "1", "--radius", "-3"])
    assert e.value.code == 2


# ---------- m3: k-sweep horizon-matched latent schedule is fixed by the config

def test_train_horizon_and_ksweep_horizon_schedule():
    assert A.train_horizon(Config("flood", "bwd_tbptt", 1, k=2)) == 8
    assert A.train_horizon(Config("flood", "bwd_tbptt", 1, k=8, n_sup=4)) == 32
    assert A.train_horizon(Config("flood", "drm", 1, k=8)) == 8
    cfg = Config("flood", "bwd_tbptt", 100, split="eval", k=2)
    ev = {f"S{s}_k{k}": {"S": s, "k_in": k, "pass1": 0.9 if (s, k) == (1, 8) else 0.1}
          for s, k in inference_grid()}
    runs_k = {dataclasses.replace(cfg, seed=s): {"eval": ev} for s in (100, 101, 102, 103)}
    ks = A.ksweep({}, runs_k)["flood"]["bwd_tbptt"]["2"]
    assert ks["horizon_k_in"] == 8 and ks["horizon"]["mean"] == pytest.approx(0.9)
    assert ks["latent16"]["mean"] == pytest.approx(0.1)
