"""Aggregate eval-split runs into results/summary.json, tables and claim_numbers.json.

Primary metric: eval pass@1 (exact match) at the inference schedule chosen on
dev for that (task, method) from the matched-compute menu (S x k_in = 16).
Comparisons are DRM minus each baseline, paired by training seed; two-sided
exact sign-flip test; Holm over the whole family (4 baselines x (5 tasks + pooled)).
The pooled mean over the three tasks that need depth (shift, gravity, flood) is
reported descriptively (CIs, no test), for continuity with round 0.

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
from drm.config import EVAL_SEEDS, METHODS, ROOT, inference_grid  # noqa: E402
from drm.stats import bootstrap_ci, holm, sign_flip_p, t_ci  # noqa: E402
from drm.tasks import TASKS  # noqa: E402
from run_grid import (KSWEEP_METHODS, KSWEEP_SEEDS, KSWEEP_TASKS, MAIN_SEEDS, MAIN_TASKS,  # noqa: E402
                      TUNING_PATH, ksweep_jobs, main_jobs, tune_jobs)  # noqa: E402
from run_one import out_paths  # noqa: E402
from tune import HEADLINE_MENU, GridError  # noqa: E402
from drm.lineage import canonical  # noqa: E402

BASELINES = ("fwd", "fwd_steps", "bwd_k", "bwd_tbptt")
TARGET = "drm"
POOLED = "pooled"
R1_TUNING = ROOT / "docs" / "r1_tuning.json"       # round-1 choices (copy of results/r1_c4532b5/tuning.json)
POOLED3 = "pooled3"
DEPTH_TASKS = ("shift", "gravity", "flood")


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
    # verified-equivalent ancestor hashes (results/code_lineage.json) count as the current code
    hashes = {canonical(d["meta"]["code_hash"]) for d in runs.values()}
    if code is not None:
        hashes.add(canonical(code))
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


def add_pooled(ps, tasks, name=POOLED):
    for m in METHODS:
        seeds = set.intersection(*(set(ps[t][m]) for t in tasks)) if all(m in ps[t] for t in tasks) else set()
        for s in seeds:
            ps[name][m][s] = float(np.mean([ps[t][m][s] for t in tasks]))
    return ps


def summarize(vals: dict):
    seeds = sorted(vals)
    v = np.array([vals[s] for s in seeds])
    mean, lo, hi = bootstrap_ci(v)
    _, tlo, thi = t_ci(v)
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0       # seed-to-seed spread (review r3: unstable picks)
    return {"mean": mean, "lo": lo, "hi": hi, "t_lo": tlo, "t_hi": thi, "n_seeds": len(seeds),
            "sd": sd, "min": float(v.min()), "max": float(v.max())}


def paired(a_: dict, b_: dict) -> dict:
    """Paired-by-seed difference a - b: bootstrap CI and exact sign-flip p (no multiplicity correction)."""
    common = sorted(set(a_) & set(b_))
    diff = np.array([a_[s] - b_[s] for s in common])
    mean, lo, hi = bootstrap_ci(diff)
    return {"diff": mean, "lo": lo, "hi": hi, "p": sign_flip_p(diff), "wins": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()), "n_seeds": len(common)}


def sensitivity(runs, tuning, ps, r1_dir: Path | None = None, r1_tuning: dict | None = None):
    """Descriptive robustness checks (review r3). NOT part of the Holm family; raw p only.

    bwd_k_native: bwd_k scored at its training horizon S=1, k_in=k (4), same lr and models as the
      headline, against DRM at its headline schedule. Shows how much DRM - bwd_k depends on which
      schedule dev picked for bwd_k (gravity: dev picked S=1, k_in=8, which is seed-unstable).
      Only bwd_k's schedule is moved; DRM's is NOT perturbed, so on flood this compares DRM at its
      16-Phi pick (S2/k8) with bwd_k at 4 Phi. Same-schedule comparisons of the two (review r4 m1)
      are in scripts/same_schedule.py -> results/same_schedule.json.
    tie_symfill_tbptt_r1: round 2 changed symfill bwd_tbptt from (3.2e-2, S1/k16) to (1.6e-2, S1/k8)
      only through an exact dev tie (0.288 both; ties go to the lower lr). Pooled DRM - TBPTT if the
      round-1 configuration (its archived eval runs) had been kept instead.
    """
    out = {}
    nat = per_seed(runs, tuning, method_filter=("drm", "bwd_k"),
                   schedule_fn=lambda c: (1, c.k) if c.method == "bwd_k" else tuple(tuning[c.task][c.method]["schedule"]))
    nat = add_pooled(nat, MAIN_TASKS)
    out["bwd_k_native"] = {g: {"bwd_k_native": summarize(nat[g]["bwd_k"]), "bwd_k_headline": summarize(ps[g]["bwd_k"]),
                               "drm_minus_bwd_k_native": paired(nat[g]["drm"], nat[g]["bwd_k"]),
                               "drm_minus_bwd_k_headline": paired(ps[g]["drm"], ps[g]["bwd_k"])}
                           for g in list(MAIN_TASKS) + [POOLED]}
    if r1_dir is not None and r1_tuning is not None:
        ch = r1_tuning["symfill"]["bwd_tbptt"]
        lr = f"{ch['lr']:g}".replace(".", "p")
        files = sorted((r1_dir / "symfill").glob(f"bwd_tbptt_k4_lr{lr}_s*.json"))
        if files:
            s, k = ch["schedule"]
            alt = {t: {m: dict(v) for m, v in ps[t].items()} for t in MAIN_TASKS}
            alt["symfill"]["bwd_tbptt"] = {}
            for f in files:
                d = json.loads(f.read_text())
                alt["symfill"]["bwd_tbptt"][d["meta"]["config"]["seed"]] = d["eval"][f"S{s}_k{k}"]["pass1"]
            pool = defaultdict(lambda: defaultdict(dict), alt)
            pool = add_pooled(pool, MAIN_TASKS)
            out["tie_symfill_tbptt_r1"] = {
                "r1_config": {"lr": ch["lr"], "schedule": [s, k]},
                "symfill_bwd_tbptt": summarize(alt["symfill"]["bwd_tbptt"]),
                "pooled_drm_minus_bwd_tbptt": paired(pool[POOLED]["drm"], pool[POOLED]["bwd_tbptt"])}
    return out


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
            _, tlo, thi = t_ci(diff)
            key = f"{g}:{TARGET}-{b}"
            raw_p[key] = sign_flip_p(diff)
            tests[key] = {"group": g, "baseline": b, "diff": mean, "lo": lo, "hi": hi, "t_lo": tlo, "t_hi": thi,
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
        out[t][m] = {str(c): {**summarize(dict(enumerate(v))), "schedule": by_compute[t][m][str(c)]["schedule"]}
                     for c, v in sorted(curve.items())}
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


def traces_at_choice(tr, by_compute):
    """Own-vs-oracle trace at each method's own dev-best schedule with exactly 16 Phi
    (review r1 m1). Traces are recorded only for the 16-Phi schedules, so since the
    round-2 headline menu (<= 16 Phi) this probe stays at the 16-Phi dev choice."""
    out = defaultdict(dict)
    for t in tr:
        for m in tr[t]:
            s, k = by_compute[t][m]["16"]["schedule"]
            rec = tr[t][m].get(f"S{s}_k{k}")
            if rec is not None:
                out[t][m] = {"schedule": [s, k], **rec}
    return out


def cpu_hours(raw: Path, runs_by_stage: dict) -> dict:
    """Sum of per-run train + eval seconds (single-threaded runs), and the wall-clock span."""
    out, starts, ends = {}, [], []
    for stage, runs in runs_by_stage.items():
        out[stage] = sum(d["train"].get("train_seconds", 0) + d["meta"].get("eval_seconds", 0)
                         for d in runs.values()) / 3600
        starts += [d["meta"].get("started_at") for d in runs.values() if d["meta"].get("started_at")]
        ends += [d["meta"].get("finished_at") for d in runs.values() if d["meta"].get("finished_at")]
    out["total"] = sum(out.values())
    out["first_start"] = min(starts) if starts else None
    out["last_finish"] = max(ends) if ends else None
    return out


def load_dev(raw: Path, code):
    runs = {}
    for cfg in tune_jobs():
        js, _ = out_paths(cfg, raw)
        if js.exists():
            d = json.loads(js.read_text())
            if canonical(d["meta"]["code_hash"]) == canonical(code):
                runs[cfg] = d
    return runs


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


def train_horizon(cfg) -> int:
    """Phi steps unrolled from a full mask in one training example."""
    return cfg.n_sup * cfg.k if cfg.method == "bwd_tbptt" else cfg.k


def ksweep(runs_main, runs_k):
    """k in {1,2,4,8} for recursive methods on KSWEEP_TASKS, KSWEEP_SEEDS only.
    Pre-registered schedules: native (S = 16/k, k_in = k) and fixed 16-step latent
    (S=1, k_in=16). Round 2 review (m3): plus the training-horizon-matched latent
    schedule S=1, k_in = horizon, where the horizon is fixed by the config: n_sup*k
    for bwd_tbptt (n_sup windows of k unrolled from a full mask), k otherwise.
    NOTE: k != 4 runs reuse the lr tuned at k = 4 (not re-tuned per k)."""
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
                    "latent16": summarize({c.seed: d["eval"]["S1_k16"]["pass1"] for c, d in sel.items()}),
                    "horizon": summarize({c.seed: d["eval"][f"S1_k{train_horizon(c)}"]["pass1"]
                                          for c, d in sel.items()}),
                    "horizon_k_in": train_horizon(next(iter(sel)))}
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
    ps = add_pooled(ps, DEPTH_TASKS, POOLED3)
    groups = list(MAIN_TASKS) + [POOLED]
    primary = {g: {m: summarize(ps[g][m]) for m in METHODS if ps[g].get(m)} for g in groups + [POOLED3]}
    tests = compare(ps, groups)
    n_params = sorted({d["meta"]["n_params"] for d in runs.values()})
    summary = {
        # code_hash: what tuning.json records; computed_under: the hashes actually stored in the raw
        # metas of the eval/k-sweep runs (review r3 m2: never relabelled by a regen at a newer HEAD)
        "meta": {"code_hash": code, "computed_under": sorted({d["meta"]["code_hash"] for d in list(runs.values())
                                                              + list(runs_k.values())}),
                 "n_runs_main": len(runs), "n_runs_ksweep": len(runs_k),
                 "missing": miss_m + miss_k, "partial": bool(miss_m or miss_k),
                 "eval_seeds": list(MAIN_SEEDS), "main_tasks": list(MAIN_TASKS), "n_params": n_params,
                 "holm_family_size": len(tests), "menu": [list(x) for x in HEADLINE_MENU]},
        "tuning": tuning,
        "primary": primary,
        "per_seed": {g: {m: {str(s): v for s, v in sorted(ps[g][m].items())} for m in ps[g]}
                     for g in groups + [POOLED3]},
        "tests": tests,
        "sensitivity": sensitivity(runs, tuning, ps, raw.parent / "r1_c4532b5" / "raw" / "eval",
                                   json.loads(R1_TUNING.read_text())["choice"] if R1_TUNING.exists() else None),
        "schedules": schedule_table(runs),
        "compute_curves": compute_curves(runs, tuning_full["by_compute"]),
        "traces": (tr := traces(runs)),
        "traces_at_choice": traces_at_choice(tr, tuning_full["by_compute"]),
        "cpu_hours": cpu_hours(raw, {"dev": load_dev(raw, code), "eval": runs, "ksweep": runs_k}),
        "tuning_meta": tuning_full["meta"],
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
        w.writerow(["group", "method", "mean_pass1", "ci_lo", "ci_hi", "t_lo", "t_hi", "sd_seeds", "n_seeds", "lr",
                    "schedule"])
        for g in groups + [POOLED3]:
            for m, r in primary[g].items():
                ch = tuning.get(g, {}).get(m, {})
                w.writerow([g, m, f"{r['mean']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                            f"{r['t_lo']:.4f}", f"{r['t_hi']:.4f}", f"{r['sd']:.4f}", r["n_seeds"],
                            ch.get("lr", ""), "x".join(map(str, ch.get("schedule", [])))])
    with open(tdir / "tests.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "diff", "ci_lo", "ci_hi", "t_lo", "t_hi", "wins", "losses", "p", "p_holm",
                    "significant"])
        for k, r in tests.items():
            w.writerow([k, f"{r['diff']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}", f"{r['t_lo']:.4f}",
                        f"{r['t_hi']:.4f}", r["wins"], r["losses"],
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
