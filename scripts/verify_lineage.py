"""Verify that the current code produces results identical to an ancestor commit's code,
and if so record the ancestor's code hash in results/code_lineage.json (drm/lineage.py).

Checks, all of which must pass:
  1. A grid of short runs (every method x 2 tasks x radius {1, 0} x k {2, 4}, small Phi
     budget, full eval on 1000 examples x 25 schedules) is run under BOTH code versions
     (the ancestor is checked out with `git worktree`). Every output except timing and
     hash fields must be identical: eval dict, traces, learning curve, per-example bits.
  2. Full-budget committed runs are re-run under the current code and compared with the
     committed results the same way.
The training loss log's elapsed-seconds column, train_seconds and the timing/hash meta
fields are the only fields excluded from the comparison.
Usage: python scripts/verify_lineage.py PARENT_COMMIT [--full dev/flood/drm_k4_lr0p008 ...] [--tmp DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drm.config import CODE_FILES, code_hash  # noqa: E402
from drm.lineage import LINEAGE_PATH  # noqa: E402

STATIC_ARGUMENT = (
    "The lineage rests on two legs. (1) Static: the CODE_FILES diff between parent and current touches only "
    "run naming (Config.run_name, NAME_FIELDS), input validation (radius >= 0, CLI checks), the overwrite guard, "
    "hash acceptance (drm/lineage.py, up_to_date) and comments. No training, inference, data or RNG code "
    "changed; every RNG (data, init, corruption, eval_gen) derives from cfg fields (seed, data_seed, S, k), "
    "never from run names or paths, and checkpoints are only written, never read back by the pipeline. "
    "(2) Empirical: the checks below re-run configs under both versions and full committed runs under the "
    "current code; all outputs except timing fields are identical. Leg 2 alone is a sample, not a proof; "
    "it is sufficient here because of leg 1. A diff that touches numerics needs a full rerun, not a lineage link.")
DEFAULT_FULL_SEED = 100

VOLATILE_META = {"code_hash", "started_at", "finished_at", "wall_seconds", "eval_seconds",
                 "loadavg_start", "loadavg_end", "cpu_count"}
METHODS = ("fwd", "fwd_steps", "bwd_k", "bwd_tbptt", "drm")


def strip(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    d["meta"] = {k: v for k, v in d["meta"].items() if k not in VOLATILE_META}
    d["train"].pop("train_seconds", None)
    # loss rows are [step, phi_used, mean_loss, elapsed_seconds]: drop only the wall-clock column
    d["train"]["loss"] = [row[:3] for row in d["train"].get("loss", [])]
    return d


def same(js_a: Path, js_b: Path) -> tuple[bool, str]:
    a, b = json.loads(js_a.read_text()), json.loads(js_b.read_text())
    if strip(a) != strip(b):
        keys = [k for k in ("meta", "train", "eval", "traces") if strip(a).get(k) != strip(b).get(k)]
        return False, f"json differs in {keys}"
    za, zb = np.load(js_a.with_suffix(".npz")), np.load(js_b.with_suffix(".npz"))
    if set(za.files) != set(zb.files) or any(not np.array_equal(za[f], zb[f]) for f in za.files):
        return False, "npz differs"
    return True, "identical"


def run(tree: Path, py: str, args: list[str], out: Path) -> Path:
    done = list(out.rglob("*.json")) if out.exists() else []
    if len(done) == 1:                    # --tmp reuse: this exact run already finished in this dir
        return done[0]
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([py, str(tree / "scripts" / "run_one.py"), *args, "--out", str(out)], check=True,
                   capture_output=True, cwd=tree)
    found = list(out.rglob("*.json"))
    assert len(found) == 1, found
    return found[0]


def default_full() -> list[str]:
    """One committed eval run per task (DRM at its tuned lr, seed 100), one k-sweep run (k=1 TBPTT)
    and one dev run, so that every task and the k != 4 path are re-run in full (review r3 n5)."""
    tuning = json.loads((ROOT / "results" / "tuning.json").read_text())["choice"]
    out = []
    raw = ROOT / "results" / "raw"
    for t in sorted(tuning):
        lr = f"{tuning[t]['drm']['lr']:g}".replace(".", "p")
        out.append(f"eval/{t}/drm_k4_lr{lr}_b12000_r1_s{DEFAULT_FULL_SEED}")
    lr = f"{tuning['gravity']['bwd_tbptt']['lr']:g}".replace(".", "p")
    out.append(f"eval/gravity/bwd_tbptt_k1_lr{lr}_b12000_r1_s{DEFAULT_FULL_SEED}")
    lr = f"{tuning['shift']['fwd']['lr']:g}".replace(".", "p")
    out.append(f"dev/shift/fwd_k4_lr{lr}_b12000_r1_s1")
    return [x for x in out if (raw / f"{x}.json").exists()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parent")
    p.add_argument("--full", nargs="*", default=None,
                   help="committed runs, e.g. dev/flood/drm_k4_lr0p008_b12000_r1_s1 (default: default_full())")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=int, default=48)
    p.add_argument("--tmp", help="reuse the outputs of an earlier invocation in this scratch dir")
    a = p.parse_args()
    if a.full is None:
        a.full = default_full()
    py = sys.executable
    parent = subprocess.run(["git", "rev-parse", a.parent], cwd=ROOT, check=True, capture_output=True,
                            text=True).stdout.strip()
    cur = code_hash()
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="lineage_"))
    tree = tmp / "parent"
    if tree.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(tree), parent], cwd=ROOT, check=True, capture_output=True)
    try:
        old = subprocess.run([py, "-c", "from drm.config import code_hash; print(code_hash())"], cwd=tree,
                             check=True, capture_output=True, text=True).stdout.strip()
        if old == cur:
            sys.exit("parent and current code hash are equal; nothing to verify")
        grid = [["--task", t, "--method", m, "--seed", "7", "--split", sp, "--lr", "0.004", "--k", str(k),
                 "--phi_budget", str(a.budget), "--radius", str(r)]
                for t in ("flood", "gravity") for m in METHODS for r in (1, 0) for k in (2, 4)
                for sp in (("dev",) if r else ("eval",))]

        def one(i_args):
            i, args = i_args
            ja = run(tree, py, args, tmp / "old" / str(i))
            jb = run(ROOT, py, args, tmp / "new" / str(i))
            ok, why = same(ja, jb)
            return {"args": " ".join(args), "ok": ok, "why": why}
        with ThreadPoolExecutor(a.workers) as ex:
            checks = list(ex.map(one, enumerate(grid)))

        def full(name):
            committed = ROOT / "results" / "raw" / f"{name}.json"
            cfg = json.loads(committed.read_text())["meta"]["config"]
            args = ["--task", cfg["task"], "--method", cfg["method"], "--seed", str(cfg["seed"]),
                    "--split", cfg["split"], "--lr", repr(cfg["lr"]), "--k", str(cfg["k"]),
                    "--phi_budget", str(cfg["phi_budget"]), "--radius", str(cfg["radius"])]
            jb = run(ROOT, py, args, tmp / "full" / name.replace("/", "_"))
            ok, why = same(committed, jb)
            return {"args": f"committed {name}", "ok": ok, "why": why}
        with ThreadPoolExecutor(a.workers) as ex:
            checks += list(ex.map(full, a.full))
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, capture_output=True)
    diffstat = subprocess.run(["git", "diff", "--stat", parent, "--", *CODE_FILES], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip().splitlines()
    record(checks, cur, old, parent, diffstat)


def record(checks: list[dict], cur: str, old: str, parent: str, diffstat: list[str], path: Path | None = None):
    """Write the lineage link only if EVERY check passed; otherwise exit 1 and leave the file untouched."""
    path = path or LINEAGE_PATH
    ok = all(c["ok"] for c in checks)
    for c in checks:
        print(("OK  " if c["ok"] else "FAIL") + f" {c['args']}: {c['why']}")
    print(f"{sum(c['ok'] for c in checks)}/{len(checks)} identical; parent {old} -> current {cur}")
    if not ok:
        sys.exit(1)
    doc = json.loads(path.read_text()) if path.exists() else {"equivalent": {}}
    full_tasks = sorted({c["args"].split()[1].split("/")[1] for c in checks if c["args"].startswith("committed ")})
    doc["equivalent"][cur] = {"parent": old, "parent_commit": parent, "all_identical": ok,
                              "static_argument": STATIC_ARGUMENT, "full_tasks": full_tasks,
                              "verified_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                              "n_checks": len(checks), "checks": checks, "code_diffstat": diffstat}
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"recorded {cur} -> {old} in {path.name}")


if __name__ == "__main__":
    main()
