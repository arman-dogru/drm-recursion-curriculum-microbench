"""Train one (task, method, seed, lr, k) model and evaluate it on one split.

Writes results/raw/<split>/<task>/<name>.json (+ .npz with per-example exact
match for every inference schedule). An existing output is skipped only if its
recorded code_hash and config_hash match; otherwise it is recomputed.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drm.config import ROOT, Config, code_hash, inference_grid  # noqa: E402
from drm.infer import exact_match, infer, oracle_state_acc  # noqa: E402
from drm.model import n_params  # noqa: E402
from drm.config import METHODS  # noqa: E402
from drm.tasks import TASKS, make_task_data  # noqa: E402

TRACE_SCHEDULES = ((16, 1), (8, 2), (4, 4), (2, 8), (1, 16))   # the whole matched-compute menu


def eval_gen(seed: int, S: int, k: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + S * 101 + k)


def out_paths(cfg: Config, out_root: Path):
    base = out_root / f"{cfg.run_name()}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def loadavg() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


def ckpt_path(cfg: Config, out_root: Path) -> Path:
    """Model weights are kept next to (not inside) the raw results tree; not committed."""
    return out_root.parent / "checkpoints" / f"{cfg.run_name()}.pt"


def up_to_date(js: Path, cfg: Config) -> bool:
    if not js.exists():
        return False
    try:
        meta = json.loads(js.read_text())["meta"]
    except (json.JSONDecodeError, KeyError):
        return False
    return meta.get("code_hash") == code_hash() and meta.get("config_hash") == cfg.config_hash()


def run(cfg: Config, out_root: Path, force: bool = False) -> dict | None:
    js, npz = out_paths(cfg, out_root)
    if not force and up_to_date(js, cfg) and npz.exists():
        print(f"skip (up to date) {js}")
        return None
    torch.set_num_threads(1)
    started, load0, wall0 = now_iso(), loadavg(), time.time()
    data = make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval, cfg.data_seed)
    Xtr, Ytr = data["train"]
    Xev, Yev = (torch.from_numpy(a) for a in data[cfg.split])
    # learning curve is always measured on a dev subset (never on eval)
    Xc, Yc = (torch.from_numpy(a[: cfg.curve_n]) for a in data["dev"])

    def curve_fn(model):
        g = eval_gen(cfg.seed, 4, 4)
        return float(exact_match(infer(model, Xc, 4, 4, g), Yc).mean())

    model, log = train(cfg, Xtr, Ytr, curve_fn=curve_fn)
    t0 = time.time()
    res, bits = {}, {}
    for S, k in inference_grid():
        pred = infer(model, Xev, S, k, eval_gen(cfg.seed, S, k))
        em = exact_match(pred, Yev)
        res[f"S{S}_k{k}"] = {"S": S, "k_in": k, "test_phi": S * k, "pass1": float(em.mean()),
                             "cell_acc": float((pred == Yev).float().mean())}
        bits[f"S{S}_k{k}"] = em.astype(np.uint8)
    traces = {}
    for S, k in TRACE_SCHEDULES:
        tr: list = []
        infer(model, Xev, S, k, eval_gen(cfg.seed, S, k), trace=tr, Y=Yev)
        traces[f"S{S}_k{k}"] = {"own": tr,
                                "oracle": oracle_state_acc(model, Xev, Yev, S, k, eval_gen(cfg.seed + 1, S, k))}
    out = {
        "meta": {"code_hash": code_hash(), "config_hash": cfg.config_hash(),
                 "config": dataclasses.asdict(cfg), "n_params": n_params(model),
                 "torch": torch.__version__, "eval_seconds": time.time() - t0,
                 "started_at": started, "finished_at": now_iso(), "wall_seconds": time.time() - wall0,
                 "loadavg_start": load0, "loadavg_end": loadavg(), "cpu_count": os.cpu_count()},
        "train": log, "eval": res, "traces": traces,
    }
    js.parent.mkdir(parents=True, exist_ok=True)
    ck = ckpt_path(cfg, out_root)
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck)
    np.savez_compressed(npz, **bits)
    js.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {js}  pass1(S4,k4)={res['S4_k4']['pass1']:.3f}  train={log['train_seconds']:.0f}s")
    return out


from drm.train import train  # noqa: E402  (after sys.path tweak)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split", default="dev", choices=["dev", "eval"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--phi_budget", type=int, default=12000)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "results" / "raw"))
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.k < 1:
        p.error("--k must be >= 1")
    if a.phi_budget <= 0:
        p.error("--phi_budget must be > 0")
    if a.lr <= 0:
        p.error("--lr must be > 0")
    cfg = Config(task=a.task, method=a.method, seed=a.seed, split=a.split, lr=a.lr, k=a.k,
                 phi_budget=a.phi_budget, radius=a.radius)
    return cfg, Path(a.out), a.force


if __name__ == "__main__":
    cfg, out, force = parse()
    run(cfg, out, force)
