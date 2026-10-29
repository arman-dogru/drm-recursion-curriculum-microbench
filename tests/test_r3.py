"""Round-3 review: tests for the tie-break rule (m3), keeping the results' code hash separate
from the current one (m2), the lineage comparison and CLI guards (m4: W2, W3, W4, W7, W8, W10,
W11), the pipeline hygiene gate (n4), the lineage record (n5) and the sensitivity checks
added for major 1."""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import aggregate as A
import run_one
import tune
import verify_lineage as V
from drm import lineage as L
from drm.config import LR_GRID, Config, code_hash, inference_grid
from drm.tasks import TASKS
from test_aggregate import grid  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


# ---------- m3 / W1 / W2: the real tie-break rule (lowest lr first, then menu order)

def _runs(score, lrs=LR_GRID):
    cfg = Config("symfill", "bwd_tbptt", 1)
    return {dataclasses.replace(cfg, seed=s, lr=lr): {"meta": {}, "eval": {
        f"S{a}_k{b}": {"S": a, "k_in": b, "pass1": score(a, b, lr)} for a, b in inference_grid()}}
        for s in (1, 2) for lr in lrs}


def test_cross_lr_tie_goes_to_the_lowest_lr_even_over_a_16_phi_schedule():
    """The round-2 symfill bwd_tbptt case: 3.2e-2 at S1/k16 (16 Phi) and 1.6e-2 at S1/k8 (8 Phi)
    tie exactly on dev; lr is the outer loop with a strict '>', so 1.6e-2 wins."""
    def score(a, b, lr):
        return 0.288 if (lr, a, b) in ((3.2e-2, 1, 16), (1.6e-2, 1, 8)) else 0.1
    ch, _, _ = tune.choose(_runs(score))
    assert ch["symfill"]["bwd_tbptt"]["lr"] == 1.6e-2
    assert ch["symfill"]["bwd_tbptt"]["schedule"] == [1, 8]


def test_same_schedule_tie_across_lrs_goes_to_the_lowest_lr():
    ch, _, _ = tune.choose(_runs(lambda a, b, lr: 0.5 if (a, b) == (4, 4) and lr >= 4e-3 else 0.1))
    assert ch["symfill"]["bwd_tbptt"]["lr"] == 4e-3


def test_within_one_lr_ties_follow_menu_order():
    ch, _, _ = tune.choose(_runs(lambda a, b, lr: 0.5 if lr == 8e-3 and (a, b) in ((1, 4), (2, 8)) else 0.1))
    assert ch["symfill"]["bwd_tbptt"]["schedule"] == [2, 8]      # 16-Phi schedule listed first


def test_tie_rule_is_documented_consistently():
    for rel in ("scripts/tune.py", "docs/method.md", "docs/README.template.md"):
        assert "lowest lr" in (ROOT / rel).read_text().lower(), rel


# ---------- W8: tune refuses non-dev runs

def test_choose_rejects_eval_runs():
    runs = _runs(lambda a, b, lr: 0.5, lrs=LR_GRID[:1])
    cfg = next(iter(runs))
    runs[dataclasses.replace(cfg, split="eval", seed=100)] = runs[cfg]
    with pytest.raises(AssertionError):
        tune.choose(runs)


# ---------- m2 / W7 / W12: results keep the hash they were computed under

def _lineage(tmp_path, monkeypatch, parent="oldhash"):
    p = tmp_path / "code_lineage.json"
    p.write_text(json.dumps({"equivalent": {code_hash(): {"parent": parent, "all_identical": True}}}))
    monkeypatch.setattr(L, "LINEAGE_PATH", p)
    return parent


def test_tune_records_the_raw_hash_not_the_current_one(tmp_path, monkeypatch):
    import test_aggregate as TA
    old = _lineage(tmp_path, monkeypatch)
    cfgs = [Config("flood", "fwd", s, lr=1e-3) for s in (1, 2)]
    for c in cfgs:
        TA.write(c, tmp_path / "raw", TA.fake_run(c, code=old))
    _, h = tune.load_runs(cfgs, tmp_path / "raw")
    assert h == old != code_hash()


def test_claims_parent_hash_comes_from_the_raw_runs(tmp_path, monkeypatch):
    import claims
    import run_grid
    import test_aggregate as TA
    tuning = {t: {m: {"lr": 1e-3, "schedule": TA.SCHED[m], "dev_pass1": 0.5} for m in TA.METHODS} for t in TA.TASKS}
    by_compute = {t: {m: {"16": {"schedule": TA.SCHED[m]}} for m in TA.METHODS} for t in TA.TASKS}
    tp = tmp_path / "tuning.json"
    tp.write_text(json.dumps({"meta": {"code_hash": "c"}, "choice": tuning, "by_compute": by_compute}))
    monkeypatch.setattr(A, "TUNING_PATH", tp)
    for c in run_grid.main_jobs(tuning) + run_grid.ksweep_jobs(tuning):
        TA.write(c, tmp_path / "raw")
    s = A.main(tmp_path / "raw", tmp_path / "out")
    assert s["meta"]["computed_under"] == ["c"]
    c = claims.build(json.loads(json.dumps(s)), {"counts": {"KILLED": 1}, "n": 1}, 3)
    assert c["code_hash_parent"] == "c" and c["code_hash"] == code_hash()[:16]


@pytest.mark.skipif(not (RAW / "eval").exists(), reason="needs the committed results")
def test_committed_tuning_and_claims_name_the_raw_hash():
    raw_dev = {json.loads(f.read_text())["meta"]["code_hash"] for f in (RAW / "dev").rglob("*.json")}
    raw_eval = {json.loads(f.read_text())["meta"]["code_hash"] for f in (RAW / "eval").rglob("*.json")}
    tuning = json.loads((ROOT / "results" / "tuning.json").read_text())
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    c = json.loads((ROOT / "results" / "claim_numbers.json").read_text())
    assert tuning["meta"]["code_hash"] in raw_dev
    assert set(summary["meta"]["computed_under"]) >= raw_eval
    assert c["code_hash_parent"] == ", ".join(summary["meta"]["computed_under"])


@pytest.mark.skipif(not (RAW / "dev").exists(), reason="needs the committed results")
def test_regenerating_tuning_at_head_is_byte_identical(tmp_path):
    out = tmp_path / "tuning.json"
    tune.main(out=out)
    assert out.read_text() == (ROOT / "results" / "tuning.json").read_text()


# ---------- W3 / W4: verify_lineage.same() really compares outputs

def _pair(tmp_path, change=None, npz_change=False):
    base = {"meta": {"code_hash": "a", "started_at": "t0", "finished_at": "t1", "wall_seconds": 3.0},
            "train": {"train_seconds": 5.0, "loss": [[1, 4, 0.5, 1.2], [2, 8, 0.4, 2.4]], "curve": [[1, 4, 0.1]]},
            "eval": {"S1_k4": {"pass1": 0.5}}, "traces": {"S1_k4": {"own": [0.1]}}}
    paths = []
    for i in (0, 1):
        d = json.loads(json.dumps(base))
        if i and change:
            change(d)
        js = tmp_path / f"r{i}.json"
        js.write_text(json.dumps(d))
        np.savez(js.with_suffix(".npz"), a=np.arange(3) + (i if npz_change else 0))
        paths.append(js)
    return paths


def test_same_ignores_only_timing(tmp_path):
    def timing(d):
        d["meta"].update(code_hash="b", started_at="x", finished_at="y", wall_seconds=9.0)
        d["train"]["train_seconds"] = 77.0
        d["train"]["loss"] = [[1, 4, 0.5, 9.9], [2, 8, 0.4, 19.9]]
    assert V.same(*_pair(tmp_path, timing))[0]


@pytest.mark.parametrize("change", [
    lambda d: d["eval"]["S1_k4"].update(pass1=0.51),
    lambda d: d["traces"]["S1_k4"].update(own=[0.2]),
    lambda d: d["train"]["loss"][0].__setitem__(2, 0.49),
    lambda d: d["train"]["curve"][0].__setitem__(2, 0.2),
])
def test_same_detects_output_differences(tmp_path, change):
    assert not V.same(*_pair(tmp_path, change))[0]


def test_same_detects_npz_differences(tmp_path):
    assert not V.same(*_pair(tmp_path, npz_change=True))[0]


# ---------- W10: the CLI exits 2 on a run-name collision with a different config

def test_cli_refuses_to_overwrite_a_different_config(tmp_path):
    cfg = Config("flood", "drm", 1, lr=1.6e-2)
    js, _ = run_one.out_paths(cfg, tmp_path)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({"meta": {"config_hash": cfg.config_hash(), "code_hash": "x"}}))
    before = js.read_text()
    r = subprocess.run([sys.executable, "scripts/run_one.py", "--task", "flood", "--method", "drm", "--seed", "1",
                        "--lr", "0.0160000001", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert Config("flood", "drm", 1, lr=0.0160000001).run_name() == cfg.run_name()
    assert r.returncode == 2, r.stderr
    assert "refusing to overwrite" in r.stderr and js.read_text() == before


# ---------- W11: a cyclic lineage file cannot hang the loaders

def test_lineage_cycle_terminates(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"equivalent": {"c1": {"parent": "c2", "all_identical": True},
                                            "c2": {"parent": "c1", "all_identical": True}}}))
    code = f"from drm.lineage import accepted_hashes; from pathlib import Path; print(sorted(accepted_hashes('c1', Path({str(p)!r}))))"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        pytest.fail("accepted_hashes loops forever on a cyclic lineage file")
    assert r.stdout.strip() == "['c1', 'c2']"


# ---------- n4: the pipeline scans for machine paths/tokens before committing and pushing

def test_pipeline_runs_hygiene_scan_before_commit_and_push():
    s = (ROOT / "scripts" / "pipeline.sh").read_text()
    block = s[s.index('if [ "${COMMIT:-0}" = 1 ]'):]
    i = block.index("$PY -m pytest -q tests/test_hygiene.py")
    assert i < block.index("git commit") and i < block.index("git push")
    assert "set -e" in s


# ---------- n5: the lineage record carries the static argument and covers every task

def test_default_lineage_full_set_covers_every_task_and_a_ksweep_k():
    if not (ROOT / "results" / "tuning.json").exists():
        pytest.skip("needs the committed results")
    full = V.default_full()
    assert {x.split("/")[1] for x in full if x.startswith("eval/")} == set(TASKS)
    assert any("_k1_" in x for x in full) and any(x.startswith("dev/") for x in full)


@pytest.mark.skipif(not L.LINEAGE_PATH.exists(), reason="needs results/code_lineage.json")
def test_lineage_entry_records_static_argument_and_full_coverage():
    e = L.load()[code_hash()]
    assert e["all_identical"] and "never from run names" in e["static_argument"]
    assert set(e["full_tasks"]) == set(TASKS)
    assert any(c["args"].startswith("committed ") and "_k1_" in c["args"] for c in e["checks"])


# ---------- major 1: sensitivity checks

def test_native_horizon_sensitivity_scores_bwd_k_at_s1_k4(grid):
    import test_aggregate as TA
    raw, out, tuning, jobs = grid
    s = A.main(raw, out)
    nat = s["sensitivity"]["bwd_k_native"]["flood"]
    cfgs = [c for c in jobs if c.task == "flood" and c.method == "bwd_k" and c.k == 4 and c.split == "eval"]
    want = np.mean([TA.value(c, 1, 4) for c in cfgs])
    assert nat["bwd_k_native"]["mean"] == pytest.approx(want)
    assert nat["bwd_k_headline"]["mean"] == pytest.approx(s["primary"]["flood"]["bwd_k"]["mean"])
    assert "sd" in s["primary"]["flood"]["bwd_k"]


def test_tie_sensitivity_uses_the_archived_round1_runs(grid, tmp_path, monkeypatch):
    raw, out, tuning, jobs = grid
    r1t = tmp_path / "r1_tuning.json"
    r1t.write_text(json.dumps({"choice": {"symfill": {"bwd_tbptt": {"lr": 3.2e-2, "schedule": [1, 16]}}}}))
    monkeypatch.setattr(A, "R1_TUNING", r1t)
    d = raw.parent / "r1_c4532b5" / "raw" / "eval" / "symfill"
    d.mkdir(parents=True)
    for seed in range(100, 112):
        (d / f"bwd_tbptt_k4_lr0p032_s{seed}.json").write_text(json.dumps(
            {"meta": {"config": {"seed": seed}}, "eval": {"S1_k16": {"pass1": 0.9}}}))
    s = A.main(raw, out)
    tie = s["sensitivity"]["tie_symfill_tbptt_r1"]
    assert tie["symfill_bwd_tbptt"]["mean"] == pytest.approx(0.9)
    assert tie["r1_config"] == {"lr": 3.2e-2, "schedule": [1, 16]}
    # a much better TBPTT cell on symfill makes pooled DRM - TBPTT more negative
    assert tie["pooled_drm_minus_bwd_tbptt"]["diff"] < s["tests"][f"{A.POOLED}:drm-bwd_tbptt"]["diff"]
