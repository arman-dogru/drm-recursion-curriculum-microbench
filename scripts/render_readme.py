"""Render README.md from docs/README.template.md + results/claim_numbers.json.

Every {{key}} must exist in claim_numbers.json; unknown keys are an error, so
no number in the README can drift from the results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def render(template: str, claims: dict) -> str:
    missing = sorted({k for k in PAT.findall(template) if k not in claims})
    if missing:
        raise KeyError(f"unknown claim keys: {missing}")
    return PAT.sub(lambda m: claims[m.group(1)], template)


def main():
    tpl = (ROOT / "docs" / "README.template.md").read_text()
    claims = json.loads((ROOT / "results" / "claim_numbers.json").read_text())
    (ROOT / "README.md").write_text(render(tpl, claims))
    print("README.md rendered")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"render_readme: {e}", file=sys.stderr)
        sys.exit(2)
