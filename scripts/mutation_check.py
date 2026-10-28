"""Mutation check: does the test suite catch deliberate bugs in the pipeline?

For each mutant: copy the repo (code + tests only) to a temp dir, apply one
textual edit, run `pytest -x -q`. A mutant counts as KILLED only if pytest
exits with code 1 (tests ran and failed). Exit 0 = SURVIVED; any other code
(collection error, timeout, ...) = ERROR and is reported separately.
An unmutated control run must exit 0 first, or the check aborts.

Writes results/mutation.json and results/mutation.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ("drm", "scripts", "tests", "docs")
# Note: tests/test_readme.py::test_readme_in_sync skips in these copies (no results/), so
# README drift is checked only in the full tree.

# (id, file, old, new, what the bug would mean)
MUTANTS = [
    ("T1", "drm/tasks.py", "return x, COLORMAP_PERM[x]", "return x, x", "colormap target is identity"),
    ("T2", "drm/tasks.py", "y[SHIFT_DR:, SHIFT_DC:] = x[:H - SHIFT_DR, :W - SHIFT_DC]",
     "y[SHIFT_DR:, :W - SHIFT_DC] = x[:H - SHIFT_DR, SHIFT_DC:]", "shift goes the wrong way"),
    ("T3", "drm/tasks.py", "c0 = rng.integers(0, W // 2 - w + 1) + side * (W // 2)",
     "c0 = rng.integers(0, W - w + 1)", "occluder may hide both mirror cells (unrecoverable target)"),
    ("T4", "drm/tasks.py", "y[H - len(col):, c] = col", "y[H - len(col):, c] = col[::-1]", "gravity reverses order"),
    ("T5", "drm/tasks.py", "for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):",
     "for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)):", "flood leaks diagonally"),
    ("T6", "drm/tasks.py", "ev = make_split(task, \"eval\", n_eval, data_seed, exclude=keys(tr[0]) | keys(dv[0]))",
     "ev = make_split(task, \"eval\", n_eval, data_seed)", "eval split not de-duplicated against train/dev"),
    ("T7", "drm/tasks.py", "dv = make_split(task, \"dev\", n_dev, data_seed, exclude=keys(tr[0]))",
     "dv = make_split(task, \"dev\", n_dev, data_seed)", "dev split not de-duplicated against train"),
    ("T8", "drm/tasks.py", "y[src] = 2\n    return x, y", "return x, y", "flood source cell not marked"),
    ("M1", "drm/model.py", ".maximum((c[:, None] - c[None, :]).abs())) <= radius",
     ".maximum((c[:, None] - c[None, :]).abs())) <= radius + 1", "attention radius off by one"),
    ("M2", "drm/model.py", "    if radius <= 0:\n        return None", "    if radius <= 1:\n        return None",
     "local attention silently global"),
    ("M3", "drm/model.py", "        h = h + ex\n", "        h = h\n", "input not injected at each Phi step"),
    ("M4", "drm/model.py", "self.lab_emb = nn.Embedding(N_COLORS + 1, d)", "self.lab_emb = self.in_emb = nn.Embedding(N_COLORS + 1, d)",
     "label and input embeddings shared"),
    ("C1", "drm/corruption.py", "return 1.0 - torch.cos(math.pi * tau / 2) ** 2", "return tau",
     "linear instead of cosine schedule (tensor path)"),
    ("C2", "drm/corruption.py", "masked = ranks < n[:, None]", "masked = ranks >= n[:, None]", "masks the complement"),
    ("C3", "drm/corruption.py", "torch.full_like(y, MASK), y)", "torch.zeros_like(y), y)", "corruption writes colour 0 instead of MASK"),
    ("R1", "drm/train.py", "if cfg.method in (\"fwd\", \"fwd_steps\", \"drm\"):", "if cfg.method in (\"fwd\", \"fwd_steps\"):",
     "DRM starts from full mask (= bwd_k)"),
    ("R2", "drm/train.py", "if cfg.method in (\"fwd\", \"fwd_steps\", \"drm\"):",
     "if cfg.method in (\"fwd\", \"fwd_steps\", \"drm\", \"bwd_tbptt\"):", "TBPTT starts from corrupted target"),
    ("R3", "drm/train.py", "            h = h.detach()                       # TBPTT stop-gradient (Eq. 4)\n", "",
     "no stop-gradient between TBPTT windows"),
    ("R4", "drm/train.py", "inner = 1 if cfg.method in (\"fwd\", \"fwd_steps\") else cfg.k", "inner = cfg.k",
     "one-step denoiser unrolled k steps"),
    ("R5", "drm/train.py", "windows = cfg.n_sup if cfg.method == \"bwd_tbptt\" else 1", "windows = 1",
     "TBPTT has a single window"),
    ("R6", "drm/train.py", "y.reshape(-1))\n", "x.reshape(-1))\n", "loss target is the input grid"),
    ("R7", "drm/train.py", "torch.manual_seed(cfg.seed)", "torch.manual_seed(0)", "init ignores seed"),
    ("R8", "drm/train.py", "return base * (step + 1) / warmup", "return base", "no warmup"),
    ("R9", "drm/train.py", "        h = initial_state(model, cfg, y, gen)\n        for _w in range(windows):",
     "        for _w in range(windows):\n            h = initial_state(model, cfg, y, gen)", "TBPTT restarts every window"),
    ("G1", "drm/config.py", "return self.phi_budget // self.ref_k", "return self.phi_budget", "fwd_steps gets FLOP-matched steps"),
    ("G2", "drm/config.py", "\"drm\": self.k}[self.method]", "\"drm\": 1}[self.method]", "DRM budget miscounted"),
    ("G3", "drm/config.py", "EVAL_SEEDS = tuple(range(100, 112))", "EVAL_SEEDS = tuple(range(1, 13))", "eval seeds overlap dev seeds"),
    ("I1", "drm/infer.py", "            s = 1.0 - i / S\n", "            s = 1.0 - (i - 1) / S\n", "remask schedule off by one step"),
    ("I2", "drm/infer.py", "            tokens = corrupt(pred, tau, gen)", "            tokens = corrupt(tokens, tau, gen)",
     "inference ignores its own predictions"),
    ("I3", "drm/infer.py", "    tokens = full_mask(n)\n    for i", "    tokens = Y if Y is not None else full_mask(n)\n    for i",
     "inference starts from the answer when traced"),
    ("I4", "drm/infer.py", "h = model.unroll(model.embed_y(tokens), ex, k_in)\n        pred = model.logits(h).argmax(-1)\n        if trace",
     "h = model.unroll(model.embed_y(tokens), ex, 1)\n        pred = model.logits(h).argmax(-1)\n        if trace",
     "inner steps ignored at inference"),
    ("O1", "scripts/run_one.py", "Xev, Yev = (torch.from_numpy(a) for a in data[cfg.split])",
     "Xev, Yev = (torch.from_numpy(a) for a in data[\"dev\"])", "always evaluates on dev"),
    ("O2", "scripts/run_one.py", "Xc, Yc = (torch.from_numpy(a[: cfg.curve_n]) for a in data[\"dev\"])",
     "Xc, Yc = (torch.from_numpy(a[: cfg.curve_n]) for a in data[cfg.split])", "learning curve peeks at eval"),
    ("O3", "scripts/run_one.py", "return meta.get(\"code_hash\") == code_hash() and meta.get(\"config_hash\") == cfg.config_hash()",
     "return meta.get(\"config_hash\") == cfg.config_hash()", "stale code not recomputed"),
    ("O4", "scripts/run_one.py", "bits[f\"S{S}_k{k}\"] = em.astype(np.uint8)", "bits[f\"S{S}_k{k}\"] = em[:10].astype(np.uint8)",
     "per-example bits truncated"),
    ("O5", "scripts/run_one.py", "    if not force and up_to_date(js, cfg) and npz.exists():", "    if up_to_date(js, cfg) and npz.exists():",
     "--force ignored"),
    ("J1", "scripts/run_grid.py", "seed=s, split=\"eval\", lr=tuning[t][m][\"lr\"])\n", "seed=s, split=\"dev\", lr=tuning[t][m][\"lr\"])\n",
     "main grid on dev split"),
    ("J2", "scripts/run_grid.py", "seed=s, split=\"eval\", lr=tuning[t][m][\"lr\"])\n", "seed=s, split=\"eval\", lr=1e-3)\n",
     "tuned lr ignored"),
    ("J3", "scripts/run_grid.py", "for t in tasks for m in methods for s in MAIN_SEEDS]", "for t in tasks for m in methods for s in DEV_SEEDS]",
     "main grid on dev seeds"),
    ("J4", "scripts/run_grid.py", "lr=tuning[t][m][\"lr\"], k=k)", "lr=tuning[t][m][\"lr\"], k=k, phi_budget=3000 * k)",
     "k-sweep not FLOP-matched"),
    ("U1", "scripts/tune.py", "for s, k in MATCHED_MENU:\n                    v", "for s, k in inference_grid():\n                    v",
     "tuning picks off-menu (more test compute) schedules"),
    ("U2", "scripts/tune.py", "if best is None or v > best[0] + 1e-12:", "if best is None or v < best[0] - 1e-12:", "tuning minimises"),
    ("U3", "scripts/tune.py", "    if len(hashes) != 1:\n        raise GridError(f\"mixed code hashes",
     "    if len(hashes) > 9:\n        raise GridError(f\"mixed code hashes", "tuning accepts mixed code"),
    ("U4", "scripts/tune.py", "    if missing:\n        raise GridError", "    if False:\n        raise GridError", "tuning accepts missing runs"),
    ("A1", "scripts/aggregate.py", "s, kin = schedule_fn(cfg) if schedule_fn else tuning[cfg.task][cfg.method][\"schedule\"]",
     "s, kin = (4, 4)", "primary ignores dev-chosen schedule"),
    ("A2", "scripts/aggregate.py", "ps[name][m][s] = float(np.mean([ps[t][m][s] for t in tasks]))",
     "ps[name][m][s] = float(ps[tasks[0]][m][s])", "pooled = first task"),
    ("A3", "scripts/aggregate.py", "diff = np.array([a_[s] - b_[s] for s in seeds])", "diff = np.array([b_[s] - a_[s] for s in seeds])",
     "difference sign flipped"),
    ("A4", "scripts/aggregate.py", "diff = np.array([a_[s] - b_[s] for s in seeds])",
     "diff = np.array([a_[s] - b_[t] for s, t in zip(seeds, seeds[::-1])])", "comparison not paired by seed"),
    ("A5", "scripts/aggregate.py", "    adj = holm(raw_p)\n", "    adj = dict(raw_p)\n", "no multiplicity correction"),
    ("A6", "scripts/aggregate.py", "    if not allow_partial:\n        check_no_extra(raw, mj + kj)", "    pass",
     "extra files accepted"),
    ("A7", "scripts/aggregate.py", "if d[\"meta\"][\"config\"][\"split\"] != \"eval\" or cfg.seed not in EVAL_SEEDS:",
     "if cfg.seed not in EVAL_SEEDS:", "dev-split run accepted in eval grid"),
    ("A8", "scripts/aggregate.py", "and c.seed in KSWEEP_SEEDS}", "}", "k=4 sweep point uses all seeds"),
    ("A9", "scripts/aggregate.py", "    if missing and not allow_partial:", "    if False:", "incomplete grid accepted"),
    ("A10", "scripts/aggregate.py", "    if a.allow_partial and out.resolve() == (ROOT / \"results\").resolve():",
     "    if False:", "--allow-partial overwrites results/"),
    ("A11", "scripts/aggregate.py", "            raise GridError(f\"stale config in {js}\")", "            pass", "stale config accepted"),
    ("K1", "scripts/run_grid.py", "MAIN_TASKS = TASKS\n", "MAIN_TASKS = (\"shift\", \"gravity\", \"flood\")\n",
     "post-hoc task cut re-applied to the eval grid"),
    ("K2", "scripts/run_grid.py", "for t in tasks for m in methods for lr in LR_GRID_FULL for s in DEV_SEEDS]",
     "for t in tasks for m in methods for lr in LR_GRID_FULL[:4] for s in DEV_SEEDS]", "lr extension dropped"),
    ("K3", "scripts/run_grid.py", "MAIN_SEEDS = EVAL_SEEDS\n", "MAIN_SEEDS = EVAL_SEEDS[:10]\n", "too few eval seeds"),
    ("F1", "scripts/make_figures.py", "    if s[\"meta\"].get(\"partial\"):\n        print(\"make_figures",
     "    if False:\n        print(\"make_figures", "figures drawn from a partial summary"),
    ("W1", "scripts/render_readme.py", "    if missing:\n        raise KeyError", "    if False:\n        raise KeyError",
     "README renders with unknown claim keys"),
    # reviewer's round-1 mutants (my_mutants.py), adapted to the round-1 code
    ("X1", "scripts/aggregate.py", "    adj = holm(raw_p)\n",
     "    adj = {}\n    for g in groups:\n        adj.update(holm({k: v for k, v in raw_p.items() if k.startswith(g + ':')}))\n",
     "Holm per group instead of one family"),
    ("X2", "scripts/run_grid.py", "return [Config(task=t, method=m, seed=s, split=\"dev\", lr=lr)",
     "return [Config(task=t, method=m, seed=s, split=\"eval\", lr=lr)", "tuning on eval split"),
    ("X3", "drm/stats.py", "lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])",
     "lo, hi = np.quantile(means, [alpha, 1 - alpha])", "90% CI labelled 95%"),
    ("X4", "drm/stats.py", "return float(np.mean(stats >= obs - 1e-12))", "return float(np.mean(stats > obs + 1e-12))",
     "anti-conservative ties"),
    ("X5", "drm/train.py", "            opt.zero_grad(set_to_none=True)\n", "", "grads accumulate across steps"),
    ("X6", "drm/train.py", "tau = torch.rand(y.shape[0], generator=gen)", "tau = 0.5 * torch.rand(y.shape[0], generator=gen)",
     "curriculum truncated to tau<=0.5"),
    ("X7", "scripts/tune.py", "f\"S{s}_k{k}\": float(np.mean(v))", "f\"S{s}_k{k}\": float(np.max(v))",
     "tuning on max over dev seeds"),
    ("X8", "drm/config.py", "DEV_SEEDS = (1, 2)", "DEV_SEEDS = (100, 101)", "dev seeds overlap eval seeds"),
    ("X9", "drm/config.py", "\"bwd_tbptt\": self.k,", "\"bwd_tbptt\": self.k * self.n_sup,", "TBPTT gets 1/4 of the budget"),
    ("X10", "scripts/aggregate.py", "    if len(hashes) > 1:\n        raise GridError(f\"mixed code hashes",
     "    if len(hashes) > 2:\n        raise GridError(f\"mixed code hashes", "aggregate accepts two code hashes"),
    ("X11", "drm/train.py", "    windows = cfg.n_sup if cfg.method == \"bwd_tbptt\" else 1",
     "    windows = cfg.n_sup if cfg.method in (\"bwd_tbptt\", \"drm\") else 1", "DRM silently gets TBPTT windows"),
    ("X12", "drm/infer.py", "    model.eval()\n    n = X.shape[0]\n    ex = model.embed_x(X)\n    tokens = full_mask(n)",
     "    model.eval()\n    n = X.shape[0]\n    ex = model.embed_x(X)\n    tokens = full_mask(n)\n"
     "    gen = torch.Generator().manual_seed(0)", "inference RNG ignores seed"),
    ("X13", "drm/tasks.py", "key = x.astype(np.int8).tobytes()", "key = np.concatenate([x, y]).astype(np.int8).tobytes()",
     "dedup on (x,y) instead of x"),
    ("X14", "scripts/aggregate.py", "ps[name][m][s] = float(np.mean([ps[t][m][s] for t in tasks]))",
     "ps[name][m][s] = float(np.mean([ps[t][m][s] for t in tasks[:2]]))", "pooled drops tasks"),
    ("X15", "drm/corruption.py", "n = n_masked(tau)                                   # (B,)",
     "n = n_masked(tau).clamp(max=M - 1)", "never fully masked"),
    # round-1 code paths
    ("Q1", "drm/config.py", "K_GRID = (1, 2, 4, 8, 16, 32, 64)", "K_GRID = (1, 2, 4, 8, 16)",
     "latent recursion capped at 16 (review M3)"),
    ("Q2", "drm/config.py", "LR_GRID = (5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2)", "LR_GRID = (5e-4, 1e-3, 2e-3, 4e-3)",
     "lr grid stops at 4e-3 (review M2)"),
    ("Q3", "drm/config.py", "lr = f\"{self.lr:g}\".replace(\".\", \"p\")", "lr = f\"{self.lr:.0e}\"",
     "lr run names collide (review m7)"),
    ("Q4", "drm/config.py", "        if self.task not in TASKS:\n            raise", "        if False:\n            raise",
     "unknown task accepted"),
    ("Q5", "drm/config.py", "        if self.k < 1 or self.n_sup < 1", "        if self.n_sup < 1", "k = 0 accepted"),
    ("Q6", "scripts/run_one.py", "    if a.phi_budget <= 0:\n        p.error", "    if False:\n        p.error",
     "CLI accepts a non-positive budget"),
    ("Q7", "scripts/run_one.py", "p.add_argument(\"--method\", required=True, choices=METHODS)",
     "p.add_argument(\"--method\", required=True)", "CLI accepts unknown method"),
    ("Q8", "scripts/run_one.py", "    torch.save(model.state_dict(), ck)\n", "", "checkpoint not saved"),
    ("Q9", "scripts/run_one.py", "                 \"started_at\": started, ", "                 ",
     "no timestamp in raw meta (review m4)"),
    ("Q10", "scripts/run_one.py", "TRACE_SCHEDULES = ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16))",
     "TRACE_SCHEDULES = ((16, 1), (4, 4), (1, 16))", "mismatch probe missing some menu schedules"),
    ("Q11", "scripts/tune.py", "                if ch[\"dev_pass1\"] > nb_best + 1e-12:", "                if False:",
     "lr edge never flagged"),
    ("Q12", "drm/stats.py", "st.t.ppf(1 - alpha / 2, len(v) - 1)", "st.norm.ppf(1 - alpha / 2)", "t-interval uses z"),
    ("Q13", "scripts/aggregate.py", "            s, k = tuning[t][m][\"schedule\"]\n            rec = tr[t][m].get",
     "            s, k = 16, 1\n            rec = tr[t][m].get", "mismatch at own schedule uses S16_k1"),
    ("Q14", "scripts/aggregate.py", "    groups = list(MAIN_TASKS) + [POOLED]\n",
     "    groups = list(MAIN_TASKS) + [POOLED, POOLED3]\n", "descriptive pooled3 enters the Holm family"),
    ("S1", "drm/stats.py", "stats = np.abs((signs * d).mean(axis=1))", "stats = (signs * d).mean(axis=1)", "one-sided p"),
    ("S2", "drm/stats.py", "idx = rng.integers(0, len(v), size=(n_boot, len(v)))",
     "idx = np.tile(np.arange(len(v)), (n_boot, 1))", "bootstrap does not resample"),
    ("S3", "drm/stats.py", "running = max(running, min(1.0, (m - i) * pvals[k]))", "running = min(1.0, (m - i) * pvals[k])",
     "Holm not monotone"),
]


def make_copy(dst: Path):
    for d in COPY:
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (dst / "results").mkdir()


def run_pytest(cwd: Path, timeout: int) -> int:
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"],
                           cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode
    except subprocess.TimeoutExpired:
        return -999


def run_mutant(m, timeout):
    mid, f, old, new, why = m
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        make_copy(td)
        p = td / f
        src = p.read_text()
        n = src.count(old)
        if n != 1:
            return {"id": mid, "file": f, "why": why, "status": "NOT_APPLICABLE", "exit": None, "matches": n}
        p.write_text(src.replace(old, new))
        code = run_pytest(td, timeout)
    status = "KILLED" if code == 1 else "SURVIVED" if code == 0 else "ERROR"
    return {"id": mid, "file": f, "why": why, "status": status, "exit": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        make_copy(Path(td))
        control = run_pytest(Path(td), a.timeout)
    print(f"control (unmutated) pytest exit = {control}", flush=True)
    if control != 0:
        print("control run failed; aborting", file=sys.stderr)
        sys.exit(2)
    muts = [m for m in MUTANTS if not a.only or m[0] in a.only]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda m: run_mutant(m, a.timeout), muts))
    for r in res:
        print(f"{r['id']:4s} {r['status']:14s} {r['why']}", flush=True)
    counts = {s: sum(r["status"] == s for r in res) for s in ("KILLED", "SURVIVED", "ERROR", "NOT_APPLICABLE")}
    out = {"control_exit": control, "n": len(res), "counts": counts, "mutants": res}
    if not a.only:
        (ROOT / "results" / "mutation.json").write_text(json.dumps(out, indent=1))
        with open(ROOT / "results" / "mutation.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "file", "status", "exit", "why"], extrasaction="ignore")
            w.writeheader()
            w.writerows(res)
    print(json.dumps(counts))
    sys.exit(0 if counts["SURVIVED"] == counts["ERROR"] == counts["NOT_APPLICABLE"] == 0 else 1)


if __name__ == "__main__":
    main()
