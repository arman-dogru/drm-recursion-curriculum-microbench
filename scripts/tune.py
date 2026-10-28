"""Pick, per (task, method), the learning rate and the matched-compute inference
schedule (S x k_in = 16) that maximise mean dev pass@1 over DEV_SEEDS.

Also records, for every test-compute level c = S*k_in, the dev-best schedule
(used for the pass@1-vs-compute curves). Only dev-split runs are read.
Refuses to run on an incomplete or mixed-code tuning grid.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drm.config import DEV_SEEDS, MATCHED_MENU, ROOT, inference_grid  # noqa: E402
from run_grid import LR_GRID_FULL as LR_GRID, TUNING_PATH, tune_jobs  # noqa: E402
from run_one import out_paths  # noqa: E402


class GridError(RuntimeError):
    pass


def load_runs(jobs, raw: Path):
    runs, missing = {}, []
    for cfg in jobs:
        js, _ = out_paths(cfg, raw)
        if not js.exists():
            missing.append(str(js))
            continue
        d = json.loads(js.read_text())
        if d["meta"]["config_hash"] != cfg.config_hash():
            raise GridError(f"config mismatch in {js}")
        runs[cfg] = d
    if missing:
        raise GridError(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    hashes = {d["meta"]["code_hash"] for d in runs.values()}
    if len(hashes) != 1:
        raise GridError(f"mixed code hashes: {sorted(hashes)}")
    return runs, hashes.pop()


def choose(runs: dict):
    by = defaultdict(lambda: defaultdict(list))       # (task, method, lr) -> sched -> [pass1]
    for cfg, d in runs.items():
        assert cfg.split == "dev" and cfg.seed in DEV_SEEDS
        for key, r in d["eval"].items():
            by[(cfg.task, cfg.method, cfg.lr)][(r["S"], r["k_in"])].append(r["pass1"])
    choice, scores, by_compute = {}, {}, {}
    for (t, m, lr), sched in by.items():
        scores.setdefault(t, {}).setdefault(m, {})[f"{lr:g}"] = {
            f"S{s}_k{k}": float(np.mean(v)) for (s, k), v in sorted(sched.items())}
    for t in scores:
        for m in scores[t]:
            best = None
            for lr in [x for x in LR_GRID if f"{x:g}" in scores[t][m]]:   # fixed order -> deterministic ties
                sc = scores[t][m][f"{lr:g}"]
                for s, k in MATCHED_MENU:
                    v = sc[f"S{s}_k{k}"]
                    if best is None or v > best[0] + 1e-12:
                        best = (v, lr, (s, k))
            choice.setdefault(t, {})[m] = {"lr": best[1], "schedule": list(best[2]), "dev_pass1": best[0]}
            sc = scores[t][m][f"{best[1]:g}"]
            curve = {}
            for c in sorted({s * k for s, k in inference_grid()}):
                cands = [(sc[f"S{s}_k{k}"], (s, k)) for s, k in inference_grid() if s * k == c]
                v, sk = max(cands, key=lambda z: (z[0], -z[1][0]))
                curve[str(c)] = {"schedule": list(sk), "dev_pass1": v}
            by_compute.setdefault(t, {})[m] = curve
    return choice, scores, by_compute


def lr_edges(choice: dict, scores: dict) -> list[str]:
    """(task, method) cells whose chosen lr is an end point of the grid AND strictly
    better on dev than its only neighbour, i.e. the optimum may lie outside the grid."""
    out = []
    for t, ms in choice.items():
        for m, ch in ms.items():
            lrs = [x for x in LR_GRID if f"{x:g}" in scores[t][m]]
            i = lrs.index(ch["lr"])
            if i in (0, len(lrs) - 1) and len(lrs) > 1:
                nb = lrs[1] if i == 0 else lrs[-2]
                nb_best = max(scores[t][m][f"{nb:g}"][f"S{s}_k{k}"] for s, k in MATCHED_MENU)
                if ch["dev_pass1"] > nb_best + 1e-12:
                    out.append(f"{t}:{m}:{ch['lr']:g}")
    return sorted(out)


def main(raw: Path = ROOT / "results" / "raw", out: Path = TUNING_PATH, fail_on_edge: bool = False):
    runs, code = load_runs(tune_jobs(), raw)
    choice, scores, by_compute = choose(runs)
    edges = lr_edges(choice, scores)
    out.write_text(json.dumps({"meta": {"code_hash": code, "dev_seeds": list(DEV_SEEDS),
                                        "lr_grid": list(LR_GRID), "menu": [list(x) for x in MATCHED_MENU],
                                        "lr_edges": edges},
                               "choice": choice, "by_compute": by_compute, "dev_scores": scores},
                              indent=1, sort_keys=True))
    print(json.dumps(choice, indent=1))
    print(f"lr edges: {edges}")
    if fail_on_edge and edges:
        print("tune: chosen lr at a grid edge; extend the grid on dev before eval", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    try:
        main(fail_on_edge="--fail-on-edge" in sys.argv)
    except GridError as e:
        print(f"tune: refusing: {e}", file=sys.stderr)
        sys.exit(2)
