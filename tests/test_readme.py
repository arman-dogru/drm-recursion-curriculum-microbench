import json
from pathlib import Path

import pytest

import render_readme

ROOT = Path(__file__).resolve().parents[1]


def test_render_substitutes_and_rejects_unknown():
    assert render_readme.render("a {{x}} b", {"x": "1.0"}) == "a 1.0 b"
    with pytest.raises(KeyError):
        render_readme.render("{{nope}}", {})


def test_readme_in_sync_with_claim_numbers():
    tpl, claims, readme = (ROOT / "docs" / "README.template.md", ROOT / "results" / "claim_numbers.json",
                           ROOT / "README.md")
    if not (tpl.exists() and claims.exists()):
        pytest.skip("results not present (e.g. inside a mutation-check copy)")
    assert readme.read_text() == render_readme.render(tpl.read_text(), json.loads(claims.read_text()))


def test_every_template_key_is_produced_by_claims(tmp_path, monkeypatch):
    """Build claims from a fabricated grid and check the template has no orphan keys."""
    import aggregate as A
    import claims
    import test_aggregate as TA

    tuning = {t: {m: {"lr": 1e-3, "schedule": TA.SCHED[m], "dev_pass1": 0.5} for m in TA.METHODS} for t in TA.TASKS}
    by_compute = {t: {m: {str(c): {"schedule": [1, c]} for c in (1, 2, 4, 8, 16)} | {
        "32": {"schedule": [2, 16]}, "64": {"schedule": [4, 16]}} for m in TA.METHODS} for t in TA.TASKS}
    tp = tmp_path / "tuning.json"
    tp.write_text(json.dumps({"meta": {"code_hash": "c"}, "choice": tuning, "by_compute": by_compute}))
    monkeypatch.setattr(A, "TUNING_PATH", tp)
    raw = tmp_path / "raw"
    import run_grid
    for c in run_grid.main_jobs(tuning) + run_grid.ksweep_jobs(tuning):
        TA.write(c, raw)
    s = A.main(raw, tmp_path / "out")
    c = claims.build(json.loads(json.dumps(s)), {"counts": {"KILLED": 1}, "n": 1}, 3)
    keys = set(render_readme.PAT.findall((ROOT / "docs" / "README.template.md").read_text()))
    prose = {k for k in keys if k.isupper()}
    assert not (keys - prose - set(c)), sorted(keys - prose - set(c))
