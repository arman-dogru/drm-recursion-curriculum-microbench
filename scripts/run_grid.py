"""Build and run the experiment stages.

  tune    dev split, DEV_SEEDS, every (task, method, lr in LR_GRID)
  main    eval split, EVAL_SEEDS, every (task, method) at its dev-tuned lr
  ksweep  eval split, KSWEEP_SEEDS, recursive methods at k in KSWEEP_K on KSWEEP_TASKS

The job lists are pure functions (tested). Outputs that already exist with a
matching code+config hash are skipped by run_one.run; anything stale is rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import datetime as dt
import os
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drm.config import DEV_SEEDS, EVAL_SEEDS, LR_GRID, METHODS, ROOT, Config  # noqa: E402
from drm.tasks import TASKS  # noqa: E402

# Round 1 review (M1): the earlier dev-based task cut was reverted. All five
# tasks and all 12 reserved eval seeds are evaluated. (The cut itself, its
# rationale and why it was reverted are described in docs/method.md.)
MAIN_TASKS = TASKS
MAIN_SEEDS = EVAL_SEEDS
KSWEEP_TASKS = ("gravity", "flood")
KSWEEP_METHODS = ("drm", "bwd_tbptt")
KSWEEP_K = (1, 2, 8)
KSWEEP_SEEDS = EVAL_SEEDS[:4]
TUNING_PATH = ROOT / "results" / "tuning.json"
# Dev-only extension (decided at ~15:55 EDT from dev results, before any eval run):
# flood bwd_k peaked at the 1.6e-2 edge of config.LR_GRID (0.47 at 8e-3 -> 0.79 at 1.6e-2),
# so one more point is added for EVERY task and method. Kept here rather than in
# drm/config.py so the code hash, and hence the finished dev runs, stay valid.
LR_EXT = (3.2e-2,)
LR_GRID_FULL = LR_GRID + LR_EXT  # same grid for every task and method


def tune_jobs(tasks=TASKS, methods=METHODS):
    return [Config(task=t, method=m, seed=s, split="dev", lr=lr)
            for t in tasks for m in methods for lr in LR_GRID_FULL for s in DEV_SEEDS]


def load_tuning(path: Path = TUNING_PATH) -> dict:
    return json.loads(Path(path).read_text())["choice"]


def main_jobs(tuning: dict, tasks=MAIN_TASKS, methods=METHODS):
    return [Config(task=t, method=m, seed=s, split="eval", lr=tuning[t][m]["lr"])
            for t in tasks for m in methods for s in MAIN_SEEDS]


def ksweep_jobs(tuning: dict):
    """k in KSWEEP_K at a fixed Phi budget. NOTE: reuses the lr tuned for k=4 (not re-tuned per k)."""
    return [Config(task=t, method=m, seed=s, split="eval", lr=tuning[t][m]["lr"], k=k)
            for t in KSWEEP_TASKS for m in KSWEEP_METHODS for k in KSWEEP_K for s in KSWEEP_SEEDS]


def _run(args):
    import run_one
    cfg, out, force = args
    try:
        run_one.run(cfg, Path(out), force)
        return cfg.run_name(), None
    except Exception as e:  # keep the pool alive; failures are reported at the end
        return cfg.run_name(), repr(e)


def jobs_for(stage: str):
    if stage == "tune":
        return tune_jobs()
    tuning = load_tuning()
    return main_jobs(tuning) if stage == "main" else ksweep_jobs(tuning)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["tune", "main", "ksweep"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=str(ROOT / "results" / "raw"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    jobs = jobs_for(a.stage)
    # longest jobs first
    jobs.sort(key=lambda c: (c.method == "fwd_steps", c.task, c.seed))
    print(f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {a.stage}: {len(jobs)} jobs", flush=True)
    failed = []
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (name, err) in enumerate(pool.imap_unordered(_run, [(c, a.out, a.force) for c in jobs]), 1):
            stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"{stamp} load={os.getloadavg()[0]:.1f} [{i}/{len(jobs)}] {name} {'FAILED ' + err if err else 'ok'}",
                  flush=True)
            if err:
                failed.append(name)
    print(f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} done; {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)
