"""Round-5 review: the claims rebuild (m1: the ss_* formatting was untested) and the pipeline order (n3)."""
import json
import re
from pathlib import Path

import pytest

import aggregate as A
import claims
from test_aggregate import grid  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


# ---------- m1: claim_numbers.json is exactly what claims.build() derives from results/

@pytest.mark.skipif(not ((RES / "summary.json").exists() and (RES / "claim_numbers.json").exists()),
                    reason="results not present (e.g. inside a mutation-check copy)")
def test_claims_rebuilt_from_results_equal_committed():
    committed = json.loads((RES / "claim_numbers.json").read_text())
    s = json.loads((RES / "summary.json").read_text())
    mp = RES / "mutation.json"
    mut = json.loads(mp.read_text()) if mp.exists() else None
    # n_tests is the only input not derived from results/ (claims.py counts it with pytest); pass it through
    rebuilt = claims.build(s, mut, int(committed["n_tests"]))
    diff = {k: (committed.get(k), rebuilt.get(k)) for k in set(committed) | set(rebuilt)
            if committed.get(k) != rebuilt.get(k)}
    assert not diff, diff


# ---------- m1: the ss_* won/lost and S/k labels on a fabricated, asymmetric same_schedule.json

def _row(sched, diff, lo, hi, p, wins, losses):
    return {"schedule": sched, "drm_minus_bwd_k": {"diff": diff, "lo": lo, "hi": hi, "p": p,
                                                   "wins": wins, "losses": losses}}


def test_same_schedule_claim_formatting(grid, tmp_path, monkeypatch):  # noqa: F811
    raw, out, tuning, jobs = grid
    s = json.loads(json.dumps(A.main(raw, out)))
    for t in s["tuning"]:
        for m in s["tuning"][t]:
            s["tuning"][t][m]["dev_pass1"] = 0.5
    R = tmp_path / "R"
    R.mkdir()
    rows = {"flood": {"bwd_k_native": _row([1, 4], -0.125, -0.15, -0.10, 0.0005, 0, 12),
                      "bwd_k_pick": _row([1, 8], 0.003, -0.034, 0.039, 0.86, 7, 5),
                      "drm_pick": _row([2, 8], 0.459, 0.40, 0.50, 0.0002, 12, 0)}}
    (R / "same_schedule.json").write_text(json.dumps({"same_schedule": rows}))
    monkeypatch.setattr(claims, "R", R)
    c = claims.build(s, None, 3)
    assert c["ss_flood_bwd_k_pick_sched"] == "S1/k8"
    assert c["ss_flood_drm_pick_sched"] == "S2/k8"
    assert c["ss_flood_bwd_k_pick_wl"] == "7/5"
    assert c["ss_flood_bwd_k_native_wl"] == "0/12"
    assert c["ss_flood_bwd_k_pick_d"] == "+0.3 pp [-3.4, +3.9]"
    assert c["ss_flood_bwd_k_pick_p"] == "0.86"
    assert c["ss_gravity_drm_pick_wl"] == "n/a"          # tasks absent from the file stay n/a


# ---------- n3: pipeline.sh runs same_schedule.py after aggregate.py and before claims.py

def test_pipeline_runs_same_schedule_between_aggregate_and_claims():
    steps = re.findall(r"^\$PY scripts/(\w+)\.py", (ROOT / "scripts" / "pipeline.sh").read_text(), re.M)
    for name in ("aggregate", "same_schedule", "claims", "render_readme"):
        assert steps.count(name) == 1, (name, steps)
    i = steps.index
    assert i("aggregate") < i("same_schedule") < i("claims") < i("render_readme")
