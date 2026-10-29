"""Code-hash lineage: which earlier code hashes produce results identical to the current code.

Every raw result records the code hash of CODE_FILES. A change that cannot affect any
number (e.g. round-2 m5: run-file naming, an overwrite guard, input validation) still
changes that hash. Instead of silently widening the hash, such a change is recorded in
results/code_lineage.json as  {new_hash: {"parent": old_hash, ...evidence}}, and the
entry is written only by scripts/verify_lineage.py after re-running configs under BOTH
code versions and finding every output identical. Results carrying an ancestor hash are
then accepted as current. Any further code change breaks the chain again until verified.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT, code_hash

LINEAGE_PATH = ROOT / "results" / "code_lineage.json"


def load(path: Path | None = None) -> dict:
    p = path or LINEAGE_PATH
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("equivalent", {})


def accepted_hashes(current: str | None = None, path: Path | None = None) -> set[str]:
    """The current hash plus every ancestor reachable through verified lineage entries."""
    eq = load(path)
    h = current or code_hash()
    out = {h}
    while h in eq and eq[h].get("all_identical") is True:
        h = eq[h]["parent"]
        if h in out:
            break
        out.add(h)
    return out


def canonical(h: str, current: str | None = None, path: Path | None = None) -> str:
    """Map a recorded hash to the current code hash if it is a verified ancestor."""
    cur = current or code_hash()
    return cur if h in accepted_hashes(cur, path) else h
