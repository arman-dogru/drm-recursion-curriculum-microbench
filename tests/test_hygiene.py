"""The repo is public: no machine-specific absolute paths or token-like strings in text files."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "scratch", "checkpoints", "runs"}
PATH_RE = re.compile(r"/home/[A-Za-z0-9_.-]+/|/workspace/")
TOKEN_RE = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[ousr]_[A-Za-z0-9]{20,}|KGAT_[A-Za-z0-9]{8,}"
                      r"|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY)")
ALLOW = {"scripts/sanitize_logs.sh", "tests/test_hygiene.py"}


def text_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and not (SKIP_DIRS & set(p.relative_to(ROOT).parts)) \
                and p.suffix in {".py", ".md", ".sh", ".log", ".json", ".csv", ".txt", ".lock", ".mmd"}:
            yield p


def test_no_absolute_box_paths_or_tokens():
    bad = []
    for p in text_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in ALLOW or rel.endswith("_live.log"):
            continue
        s = p.read_text(errors="ignore")
        if PATH_RE.search(s) or TOKEN_RE.search(s):
            bad.append(rel)
    assert not bad, bad
