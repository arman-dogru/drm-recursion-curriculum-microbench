"""Build and run the experiment stages.

  tune    dev split, DEV_SEEDS, every (task, method, lr in LR_GRID_FULL)
  main    eval split, EVAL_SEEDS, every (task, method) at its dev-tuned lr
  ksweep  eval split, KSWEEP_SEEDS, recursive methods at k in KSWEEP_K on KSWEEP_TASKS

The job lists are pure functions (tested). Outputs that already exist with a
matching code+config hash are skipped by run_one.run; anything stale is rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drm.config import DEV_SEEDS, EVAL_SEEDS, LR_GRID, METHODS, ROOT, Config  # noqa: E402
from drm.tasks import TASKS  # noqa: E402

# Scope cut made at ~10:00 EDT on 2026-10-28, after the box CPU slowed ~3.5x and
# before any eval run, using dev results only:
#  * colormap is solved (dev pass@1 = 1.00) by every method at every lr, and
#    symfill by none (dev pass@1 < 0.05) -> both are reported from dev tuning
#    only and dropped from the eval grid;
#  * the eval grid uses the first 10 of the 12 reserved eval seeds;
#  * the k-sweep runs on flood only with 4 seeds.
MAIN_TASKS = ("shift", "gravity", "flood")
MAIN_SEEDS = EVAL_SEEDS[:10]
KSWEEP_TASKS = ("flood",)
KSWEEP_METHODS = ("drm", "bwd_tbptt")
KSWEEP_K = (1, 2, 8)
KSWEEP_SEEDS = EVAL_SEEDS[:4]
TUNING_PATH = ROOT / "results" / "tuning.json"
# The dev optimum for the recursive methods sat at the 2e-3 edge of the initial
# grid (config.LR_GRID), so the dev grid was extended by one point for every
# method before any eval run. Decided on dev results only.
LR_GRID_FULL = LR_GRID + (4e-3,)
LR_EXT = (4e-3,)                 # extension applied to MAIN_TASKS only


def tune_jobs(tasks=TASKS, methods=METHODS):
    return [Config(task=t, method=m, seed=s, split="dev", lr=lr)
            for t in tasks for m in methods
            for lr in (LR_GRID_FULL if t in MAIN_TASKS else LR_GRID) for s in DEV_SEEDS]


def load_tuning(path: Path = TUNING_PATH) -> dict:
    return json.loads(Path(path).read_text())["choice"]


def main_jobs(tuning: dict, tasks=MAIN_TASKS, methods=METHODS):
    return [Config(task=t, method=m, seed=s, split="eval", lr=tuning[t][m]["lr"])
            for t in tasks for m in methods for s in MAIN_SEEDS]


def ksweep_jobs(tuning: dict):
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
    print(f"{a.stage}: {len(jobs)} jobs", flush=True)
    failed = []
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (name, err) in enumerate(pool.imap_unordered(_run, [(c, a.out, a.force) for c in jobs]), 1):
            print(f"[{i}/{len(jobs)}] {name} {'FAILED ' + err if err else 'ok'}", flush=True)
            if err:
                failed.append(name)
    print(f"done; {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)
