"""Code-hash lineage (round 2 m5): only verified, identical-output ancestors are accepted."""
import json
from pathlib import Path

import pytest

from drm import lineage as L
from drm.config import code_hash

ROOT = Path(__file__).resolve().parents[1]


def test_accepted_hashes_follow_only_verified_links(tmp_path):
    p = tmp_path / "lin.json"
    p.write_text(json.dumps({"equivalent": {
        "c3": {"parent": "c2", "all_identical": True},
        "c2": {"parent": "c1", "all_identical": True},
        "c1": {"parent": "c0", "all_identical": False}}}))
    assert L.accepted_hashes("c3", p) == {"c3", "c2", "c1"}
    assert L.accepted_hashes("c2", p) == {"c2", "c1"}
    assert L.accepted_hashes("zz", p) == {"zz"}
    assert L.canonical("c1", "c3", p) == "c3" and L.canonical("c0", "c3", p) == "c0"
    assert L.accepted_hashes("c3", tmp_path / "missing.json") == {"c3"}


def test_committed_results_are_current_or_verified_ancestors():
    if not L.LINEAGE_PATH.exists() or not (ROOT / "results" / "raw").exists():
        pytest.skip("results not present (e.g. inside a mutation-check copy)")
    eq = L.load()
    ok = L.accepted_hashes()
    e = eq.get(code_hash())
    if e:
        assert e["all_identical"] and e["n_checks"] == len(e["checks"]) and all(c["ok"] for c in e["checks"])
    hashes = {json.loads(js.read_text())["meta"]["code_hash"] for js in (ROOT / "results" / "raw").rglob("*.json")}
    assert hashes <= ok, hashes - ok
