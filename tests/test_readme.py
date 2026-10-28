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
