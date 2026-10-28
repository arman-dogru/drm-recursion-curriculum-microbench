"""Aggregate eval-split runs into results/summary.json, tables and claim_numbers.json.

Primary metric: eval pass@1 (exact match) at the inference schedule chosen on
dev for that (task, method) from the matched-compute menu (S x k_in = 16).
Comparisons are DRM minus each baseline, paired by training seed; two-sided
exact sign-flip test; Holm over the whole family (4 baselines x (3 main tasks + pooled)).

Refuses (exit 2) on a missing/extra/stale/mixed-code grid. --allow-partial is
for debugging only and must write to a directory other than results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drm.config import EVAL_SEEDS, MATCHED_MENU, METHODS, ROOT, inference_grid  # noqa: E402
from drm.stats import bootstrap_ci, holm, sign_flip_p  # noqa: E402
from drm.tasks import TASKS  # noqa: E402
from run_grid import (KSWEEP_METHODS, KSWEEP_SEEDS, KSWEEP_TASKS, MAIN_SEEDS, MAIN_TASKS,  # noqa: E402
                      TUNING_PATH, ksweep_jobs, main_jobs)  # noqa: E402
from run_one import out_paths  # noqa: E402
from tune import GridError  # noqa: E402

BASELINES = ("fwd", "fwd_steps", "bwd_k", "bwd_tbptt")
TARGET = "drm"
POOLED = "pooled"


def load_grid(jobs, raw: Path, allow_partial=False, code=None):
    runs, missing = {}, []
    for cfg in jobs:
        js, npz = out_paths(cfg, raw)
        if not (js.exists() and npz.exists()):
            missing.append(str(js))
            continue
        d = json.loads(js.read_text())
        if d["meta"]["config_hash"] != cfg.config_hash():
            raise GridError(f"stale config in {js}")
        if d["meta"]["config"]["split"] != "eval" or cfg.seed not in EVAL_SEEDS:
            raise GridError(f"non-eval run in eval grid: {js}")
        runs[cfg] = d
    if missing and not allow_partial:
        raise GridError(f"incomplete grid: {len(missing)} runs missing, e.g. {missing[:2]}")
    hashes = {d["meta"]["code_hash"] for d in runs.values()}
    if code is not None:
        hashes.add(code)
    if len(hashes) > 1:
        raise GridError(f"mixed code hashes: {sorted(hashes)}")
    return runs, missing


def check_no_extra(raw: Path, jobs):
    expected = {out_paths(c, raw)[0].resolve() for c in jobs}
    found = {p.resolve() for p in (raw / "eval").rglob("*.json")}
    extra = sorted(found - expected)
    if extra:
        raise GridError(f"unexpected files in eval grid ({len(extra)}), e.g. {[str(e) for e in extra[:2]]}")


def per_seed(runs, tuning, method_filter=None, k=4, schedule_fn=None):
    """{task: {method: {seed: pass1}}} at the dev-chosen (or given) schedule."""
    out = defaultdict(lambda: defaultdict(dict))
    for cfg, d in runs.items():
        if cfg.k != k or (method_filter and cfg.method not in method_filter):
            continue
        s, kin = schedule_fn(cfg) if schedule_fn else tuning[cfg.task][cfg.method]["schedule"]
        out[cfg.task][cfg.method][cfg.seed] = d["eval"][f"S{s}_k{kin}"]["pass1"]
    return out


def add_pooled(ps, tasks):
    for m in METHODS:
        seeds = set.intersection(*(set(ps[t][m]) for t in tasks)) if all(m in ps[t] for t in tasks) else set()
        for s in seeds:
            ps[POOLED][m][s] = float(np.mean([ps[t][m][s] for t in tasks]))
    return ps


def summarize(vals: dict):
    seeds = sorted(vals)
    mean, lo, hi = bootstrap_ci(np.array([vals[s] for s in seeds]))
    return {"mean": mean, "lo": lo, "hi": hi, "n_seeds": len(seeds)}


def compare(ps, groups):
    tests, raw_p = {}, {}
    for g in groups:
        for b in BASELINES:
            a_, b_ = ps[g][TARGET], ps[g][b]
            seeds = sorted(set(a_) & set(b_))
            if len(seeds) < 2:
                continue
            diff = np.array([a_[s] - b_[s] for s in seeds])
            mean, lo, hi = bootstrap_ci(diff)
            key = f"{g}:{TARGET}-{b}"
            raw_p[key] = sign_flip_p(diff)
            tests[key] = {"group": g, "baseline": b, "diff": mean, "lo": lo, "hi": hi,
                          "n_seeds": len(seeds), "wins": int((diff > 0).sum()),
                          "losses": int((diff < 0).sum()), "p": raw_p[key]}
    adj = holm(raw_p)
    for k in tests:
        tests[k]["p_holm"] = adj[k]
        tests[k]["significant"] = adj[k] < 0.05
    return tests


def schedule_table(runs):
    """Mean eval pass@1 per (task, method, S, k_in) over seeds, no selection."""
    acc = defaultdict(list)
    for cfg, d in runs.items():
        if cfg.k != 4:
            continue
        for key, r in d["eval"].items():
            acc[(cfg.task, cfg.method, key)].append(r["pass1"])
    out = defaultdict(lambda: defaultdict(dict))
    for (t, m, key), v in acc.items():
        out[t][m][key] = float(np.mean(v))
    return out


def compute_curves(runs, by_compute):
    """Eval pass@1 at the dev-chosen schedule for each test-compute level."""
    acc = defaultdict(lambda: defaultdict(list))
    for cfg, d in runs.items():
        if cfg.k != 4:
            continue
        for c, ch in by_compute[cfg.task][cfg.method].items():
            s, k = ch["schedule"]
            acc[(cfg.task, cfg.method)][int(c)].append(d["eval"][f"S{s}_k{k}"]["pass1"])
    out = defaultdict(dict)
    for (t, m), curve in acc.items():
        out[t][m] = {str(c): summarize(dict(enumerate(v))) for c, v in sorted(curve.items())}
    return out


def traces(runs):
    out = defaultdict(lambda: defaultdict(dict))
    for t in MAIN_TASKS:
        for m in METHODS:
            ds = [d for c, d in runs.items() if c.task == t and c.method == m and c.k == 4]
            if not ds:
                continue
            for sched in ds[0]["traces"]:
                rec = {}
                for kind in ("own", "oracle"):
                    steps = ds[0]["traces"][sched][kind]
                    rec[kind] = []
                    for i in range(len(steps)):
                        row = {"step": steps[i]["step"]}
                        for f in steps[i]:
                            if f == "step":
                                continue
                            v = [d["traces"][sched][kind][i][f] for d in ds]
                            v = [x for x in v if x == x]          # drop NaN
                            row[f] = float(np.mean(v)) if v else None
                        rec[kind].append(row)
                out[t][m][sched] = rec
    return out


def learning_curves(runs):
    out = defaultdict(dict)
    for t in MAIN_TASKS:
        for m in METHODS:
            ds = [d for c, d in runs.items() if c.task == t and c.method == m and c.k == 4]
            if not ds:
                continue
            n = min(len(d["train"]["curve"]) for d in ds)
            out[t][m] = [{"phi_used": ds[0]["train"]["curve"][i][1],
                          "dev_pass1": float(np.mean([d["train"]["curve"][i][2] for d in ds]))}
                         for i in range(n)]
    return out


def ksweep(runs_main, runs_k):
    """k in {1,2,4,8} for recursive methods on KSWEEP_TASKS, KSWEEP_SEEDS only.
    Pre-registered schedules: native (S = 16/k, k_in = k) and latent (S=1, k_in=16)."""
    out = defaultdict(lambda: defaultdict(dict))
    for t in KSWEEP_TASKS:
        for m in KSWEEP_METHODS:
            for k in (1, 2, 4, 8):
                src = runs_main if k == 4 else runs_k
                sel = {c: d for c, d in src.items() if c.task == t and c.method == m and c.k == k
                       and c.seed in KSWEEP_SEEDS}
                if not sel:
                    continue
                out[t][m][str(k)] = {
                    "native": summarize({c.seed: d["eval"][f"S{16 // k}_k{k}"]["pass1"] for c, d in sel.items()}),
                    "latent16": summarize({c.seed: d["eval"]["S1_k16"]["pass1"] for c, d in sel.items()})}
    return out


def main(raw: Path, out_dir: Path, allow_partial: bool = False):
    tuning_full = json.loads(TUNING_PATH.read_text())
    tuning = tuning_full["choice"]
    mj, kj = main_jobs(tuning), ksweep_jobs(tuning)
    code = tuning_full["meta"]["code_hash"]
    if not allow_partial:
        check_no_extra(raw, mj + kj)
    runs, miss_m = load_grid(mj, raw, allow_partial, code)
    runs_k, miss_k = load_grid(kj, raw, allow_partial, code)

    ps = per_seed(runs, tuning)
    ps = add_pooled(ps, MAIN_TASKS)
    groups = list(MAIN_TASKS) + [POOLED]
    primary = {g: {m: summarize(ps[g][m]) for m in METHODS if ps[g].get(m)} for g in groups}
    tests = compare(ps, groups)
    n_params = sorted({d["meta"]["n_params"] for d in runs.values()})
    summary = {
        "meta": {"code_hash": code, "n_runs_main": len(runs), "n_runs_ksweep": len(runs_k),
                 "missing": miss_m + miss_k, "partial": bool(miss_m or miss_k),
                 "eval_seeds": list(MAIN_SEEDS), "main_tasks": list(MAIN_TASKS), "n_params": n_params,
                 "holm_family_size": len(tests), "menu": [list(x) for x in MATCHED_MENU]},
        "tuning": tuning,
        "primary": primary,
        "per_seed": {g: {m: {str(s): v for s, v in sorted(ps[g][m].items())} for m in ps[g]} for g in groups},
        "tests": tests,
        "schedules": schedule_table(runs),
        "compute_curves": compute_curves(runs, tuning_full["by_compute"]),
        "traces": traces(runs),
        "learning_curves": learning_curves(runs),
        "ksweep": ksweep(runs, runs_k),
        "train_seconds": {m: float(np.mean([d["train"]["train_seconds"] for c, d in runs.items()
                                            if c.method == m])) for m in METHODS if any(c.method == m for c in runs)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    tdir = out_dir / "tables"
    tdir.mkdir(exist_ok=True)
    with open(tdir / "primary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "method", "mean_pass1", "ci_lo", "ci_hi", "n_seeds", "lr", "schedule"])
        for g in groups:
            for m, r in primary[g].items():
                ch = tuning.get(g, {}).get(m, {})
                w.writerow([g, m, f"{r['mean']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["n_seeds"],
                            ch.get("lr", ""), "x".join(map(str, ch.get("schedule", [])))])
    with open(tdir / "tests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "diff", "ci_lo", "ci_hi", "wins", "losses", "p", "p_holm", "significant"])
        for k, r in tests.items():
            w.writerow([k, f"{r['diff']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["wins"], r["losses"],
                        f"{r['p']:.6f}", f"{r['p_holm']:.6f}", r["significant"]])
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    if a.allow_partial and out.resolve() == (ROOT / "results").resolve():
        print("aggregate: --allow-partial must write to a separate --out, not results/", file=sys.stderr)
        sys.exit(2)
    try:
        s = main(Path(a.raw), out, a.allow_partial)
    except GridError as e:
        print(f"aggregate: refusing: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({g: {m: round(r["mean"], 4) for m, r in v.items()} for g, v in s["primary"].items()}, indent=1))
