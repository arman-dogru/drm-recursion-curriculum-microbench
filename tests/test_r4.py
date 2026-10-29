"""Round-4 review: tests for the round-3 code (m2: the reviewer's P1-P9) and the round-4 additions
(m1 same-schedule check, n4 mixed raw hashes in tuning.json)."""
import json

import numpy as np
import pytest

import aggregate as A
import claims
import run_grid
import same_schedule as SS
import tune
import verify_lineage as V
from drm import lineage as L
from drm.config import Config, code_hash
from drm.tasks import TASKS
from run_grid import MAIN_SEEDS, MAIN_TASKS
from test_aggregate import BASE, METHODS, SCHED, fake_run, grid, value, write  # noqa: F401  (fixture)


# ---------- P1 / P9 / P2 / P3: sensitivity p, wins/losses, sd and the 5-task pool

def _nat(grid):
    raw, out, tuning, jobs = grid
    return A.main(raw, out), jobs


def test_native_sensitivity_two_sided_p_and_wins_losses(grid):
    s, _ = _nat(grid)
    d = s["sensitivity"]["bwd_k_native"]["flood"]["drm_minus_bwd_k_native"]
    # fixture: bwd_k at S1/k4 is a 0.99 decoy, DRM ~0.5 -> DRM loses on all 12 seeds
    assert (d["wins"], d["losses"], d["n_seeds"]) == (0, 12, 12)
    assert d["p"] == pytest.approx(2 / 2 ** 12)          # exact two-sided sign flip, all 12 same sign
    h = s["sensitivity"]["bwd_k_native"]["flood"]["drm_minus_bwd_k_headline"]
    assert (h["wins"], h["losses"]) == (12, 0)            # headline bwd_k = 0.25 + ... < DRM


def test_seed_sd_uses_ddof_1(grid):
    s, _ = _nat(grid)
    want = np.std([value(Config("flood", "drm", sd), 4, 4) for sd in MAIN_SEEDS], ddof=1)
    assert s["primary"]["flood"]["drm"]["sd"] == pytest.approx(want)
    assert want > 0


def test_native_pool_covers_all_five_tasks(grid):
    s, _ = _nat(grid)
    nat = s["sensitivity"]["bwd_k_native"]
    assert set(nat) == set(MAIN_TASKS) | {A.POOLED} and len(MAIN_TASKS) == 5
    per_task = np.mean([nat[t]["drm_minus_bwd_k_native"]["diff"] for t in MAIN_TASKS])
    assert nat[A.POOLED]["drm_minus_bwd_k_native"]["diff"] == pytest.approx(per_task)
    three = np.mean([nat[t]["drm_minus_bwd_k_native"]["diff"] for t in ("shift", "gravity", "flood")])
    assert abs(per_task - three) > 1e-6


# ---------- P4: summary.json computed_under keeps the raw (ancestor) hash on regeneration

def test_summary_computed_under_keeps_the_ancestor_hash(tmp_path, monkeypatch):
    p = tmp_path / "code_lineage.json"
    p.write_text(json.dumps({"equivalent": {code_hash(): {"parent": "oldhash", "all_identical": True}}}))
    monkeypatch.setattr(L, "LINEAGE_PATH", p)
    tuning = {t: {m: {"lr": 1e-3, "schedule": SCHED[m]} for m in METHODS} for t in TASKS}
    by_compute = {t: {m: {"16": {"schedule": SCHED[m]}} for m in METHODS} for t in TASKS}
    tp = tmp_path / "tuning.json"
    tp.write_text(json.dumps({"meta": {"code_hash": "oldhash"}, "choice": tuning, "by_compute": by_compute}))
    monkeypatch.setattr(A, "TUNING_PATH", tp)
    for c in run_grid.main_jobs(tuning) + run_grid.ksweep_jobs(tuning):
        write(c, tmp_path / "raw", fake_run(c, code="oldhash"))
    s = A.main(tmp_path / "raw", tmp_path / "out")
    assert s["meta"]["computed_under"] == ["oldhash"] != [code_hash()]


# ---------- P5 / P6: claims tie list excludes colormap; superseded and r0 CPU-h include eval seconds

def test_claims_tie_cells_and_superseded_cpu_hours(grid, tmp_path, monkeypatch):
    raw, out, tuning, jobs = grid
    s = A.main(raw, out)
    R = tmp_path / "R"
    menu = [[1, 4], [1, 8]]
    tie = {"1e-3": {"S1_k4": 0.5, "S1_k8": 0.1}, "2e-3": {"S1_k4": 0.1, "S1_k8": 0.5}}
    no_tie = {"1e-3": {"S1_k4": 0.5, "S1_k8": 0.1}, "2e-3": {"S1_k4": 0.4, "S1_k8": 0.1}}
    within = {"1e-3": {"S1_k4": 0.5, "S1_k8": 0.5}, "2e-3": {"S1_k4": 0.1, "S1_k8": 0.1}}
    off_menu = {"1e-3": {"S1_k4": 0.5, "S1_k16": 0.9}, "2e-3": {"S1_k4": 0.1, "S1_k16": 0.9}}
    dev = {"colormap": {"drm": tie}, "flood": {"drm": tie, "bwd_k": no_tie, "fwd": within, "fwd_steps": off_menu}}
    (R / "r1_c4532b5" / "raw" / "eval" / "flood").mkdir(parents=True)
    (R / "r0_932be94" / "raw").mkdir(parents=True)
    (R / "tuning.json").write_text(json.dumps({"meta": {"menu": menu}, "dev_scores": dev}))
    for i, (tr, ev) in enumerate(((1800.0, 1800.0), (3600.0, 0.0))):
        (R / "r1_c4532b5" / "raw" / "eval" / "flood" / f"r{i}.json").write_text(
            json.dumps({"train": {"train_seconds": tr}, "meta": {"eval_seconds": ev}}))
    (R / "r0_932be94" / "raw" / "x.json").write_text(
        json.dumps({"train": {"train_seconds": 0.0}, "meta": {"eval_seconds": 7200.0}}))
    monkeypatch.setattr(claims, "R", R)
    s = json.loads(json.dumps(s))
    for t in s["tuning"]:
        for m in s["tuning"][t]:
            s["tuning"][t][m]["dev_pass1"] = 0.5
    mut = {"counts": {"KILLED": 2}, "n": 3, "mutants": [{"id": "P1", "status": "KILLED"},
                                                         {"id": "P2", "status": "SURVIVED"},
                                                         {"id": "W1", "status": "KILLED"}]}
    c = claims.build(s, mut, 3)
    assert (c["mut_rev4_killed"], c["mut_rev4_total"]) == ("1", "2")
    assert c["tie_cells"] == "flood drm" and c["n_tie_cells"] == "1"
    assert (c["cpu_h_superseded"], c["n_runs_superseded"]) == ("2.0", "2")
    assert (c["cpu_h_r0"], c["n_runs_r0"]) == ("2.0", "1")


# ---------- P7: verify_lineage exits 1 on a failed check and leaves code_lineage.json unchanged

def test_lineage_record_refuses_a_failed_check(tmp_path):
    p = tmp_path / "code_lineage.json"
    before = json.dumps({"equivalent": {"x": {"parent": "y", "all_identical": True}}})
    p.write_text(before)
    checks = [{"args": "a", "ok": True, "why": "identical"}, {"args": "committed eval/flood/x", "ok": False,
                                                               "why": "json differs in ['eval']"}]
    with pytest.raises(SystemExit) as e:
        V.record(checks, "cur", "old", "abc", [], path=p)
    assert e.value.code == 1
    assert p.read_text() == before


def test_lineage_record_writes_on_success(tmp_path):
    p = tmp_path / "code_lineage.json"
    V.record([{"args": "committed eval/flood/x", "ok": True, "why": "identical"}], "cur", "old", "abc", [], path=p)
    e = json.loads(p.read_text())["equivalent"]["cur"]
    assert e["parent"] == "old" and e["all_identical"] is True and e["full_tasks"] == ["flood"]


# ---------- P8: default_full() keeps one run per task, the k-sweep run and a dev run (no results/ needed)

def test_default_full_on_a_fabricated_tree(tmp_path, monkeypatch):
    lr = 1.6e-2
    tuning = {t: {"drm": {"lr": lr}, "bwd_tbptt": {"lr": lr}, "fwd": {"lr": lr}} for t in TASKS}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "tuning.json").write_text(json.dumps({"choice": tuning}))
    names = [f"eval/{t}/drm_k4_lr0p016_b12000_r1_s{V.DEFAULT_FULL_SEED}" for t in TASKS]
    names += [f"eval/gravity/bwd_tbptt_k1_lr0p016_b12000_r1_s{V.DEFAULT_FULL_SEED}", "dev/shift/fwd_k4_lr0p016_b12000_r1_s1"]
    for n in names:
        f = tmp_path / "results" / "raw" / f"{n}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    monkeypatch.setattr(V, "ROOT", tmp_path)
    assert sorted(V.default_full()) == sorted(names)


# ---------- m1: same-schedule DRM - bwd_k

def test_same_schedule_scores_both_methods_at_one_schedule(grid):
    raw, out, tuning, jobs = grid
    runs, _ = A.load_grid(run_grid.main_jobs(tuning), raw, False, "c")
    r = SS.same_schedule(runs, tuning)["flood"]
    assert r["drm_pick"]["schedule"] == SCHED["drm"] and r["bwd_k_native"]["schedule"] == [1, 4]
    # at DRM's pick (S4/k4) bwd_k scores its 0.99 decoy, DRM its signal
    want_drm = np.mean([value(Config("flood", "drm", sd), 4, 4) for sd in MAIN_SEEDS])
    assert r["drm_pick"]["drm"]["mean"] == pytest.approx(want_drm)
    assert r["drm_pick"]["bwd_k"]["mean"] == pytest.approx(0.99)
    # at bwd_k's pick (S1/k16) both are scored at S1/k16: DRM 0.0 decoy, bwd_k signal
    assert r["bwd_k_pick"]["drm"]["mean"] == pytest.approx(0.0)
    assert r["bwd_k_pick"]["bwd_k"]["mean"] == pytest.approx(s := np.mean(
        [value(Config("flood", "bwd_k", sd), 1, 16) for sd in MAIN_SEEDS]))
    assert r["bwd_k_pick"]["drm_minus_bwd_k"]["diff"] == pytest.approx(-s)


# ---------- n4: tuning.json lists every raw hash when the dev grid mixes linked hashes

def test_tuning_meta_lists_mixed_raw_hashes():
    runs = {i: {"meta": {"code_hash": h}} for i, h in enumerate(("new", "old", "old"))}
    m = tune.tuning_meta(runs, "new", [])
    assert m["computed_under"] == ["new", "old"] and m["code_hash"] == "new"
    single = tune.tuning_meta({0: {"meta": {"code_hash": "old"}}}, "old", [])
    assert "computed_under" not in single          # byte-identical tuning.json for the committed grid
