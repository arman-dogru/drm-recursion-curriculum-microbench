"""Same-schedule DRM - bwd_k (review r4 m1). Descriptive only: NOT part of the Holm family, raw p only.

`sensitivity.bwd_k_native` (aggregate.py) moves only bwd_k to its native S=1, k_in=4 schedule and
leaves DRM at its own dev pick, so on flood it compares DRM at 16 Phi with bwd_k at 4 Phi. This
script scores BOTH methods at the SAME schedule, using the same tuned lrs and the same eval runs
as the headline, for three schedules per task: bwd_k's native S1/k4, bwd_k's dev pick and DRM's
dev pick. No new runs; reads results/raw/eval and results/tuning.json, writes
results/same_schedule.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402
from drm.config import ROOT  # noqa: E402
from run_grid import MAIN_TASKS, main_jobs  # noqa: E402

OUT = ROOT / "results" / "same_schedule.json"
NATIVE = (1, 4)


def schedules(tuning: dict, task: str) -> dict:
    """role -> (S, k_in); roles whose schedule coincides are kept (the label says which picked it)."""
    return {"bwd_k_native": NATIVE, "bwd_k_pick": tuple(tuning[task]["bwd_k"]["schedule"]),
            "drm_pick": tuple(tuning[task]["drm"]["schedule"])}


def same_schedule(runs: dict, tuning: dict) -> dict:
    out = {}
    for t in MAIN_TASKS:
        out[t] = {}
        for role, (s, k) in schedules(tuning, t).items():
            ps = A.per_seed({c: d for c, d in runs.items() if c.task == t}, tuning,
                            method_filter=("drm", "bwd_k"), schedule_fn=lambda c, s=s, k=k: (s, k))
            out[t][role] = {"schedule": [s, k], "drm": A.summarize(ps[t]["drm"]),
                            "bwd_k": A.summarize(ps[t]["bwd_k"]),
                            "drm_minus_bwd_k": A.paired(ps[t]["drm"], ps[t]["bwd_k"])}
    return out


def main(raw: Path = ROOT / "results" / "raw", out: Path = OUT, tuning_path: Path | None = None):
    tf = json.loads((tuning_path or A.TUNING_PATH).read_text())
    tuning = tf["choice"]
    runs, _ = A.load_grid(main_jobs(tuning), raw, False, tf["meta"]["code_hash"])
    res = {"meta": {"note": "descriptive, outside the Holm family; same tuned lrs and eval runs as the headline",
                    "roles": ["bwd_k_native", "bwd_k_pick", "drm_pick"]},
           "same_schedule": same_schedule(runs, tuning)}
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res


if __name__ == "__main__":
    try:
        r = main()
    except A.GridError as e:
        print(f"same_schedule: refusing: {e}", file=sys.stderr)
        sys.exit(2)
    for t, rows in r["same_schedule"].items():
        for role, x in rows.items():
            d = x["drm_minus_bwd_k"]
            print(f"{t:9s} {role:13s} S{x['schedule'][0]}/k{x['schedule'][1]:<3d} DRM-bwd_k {100*d['diff']:+.1f} "
                  f"[{100*d['lo']:+.1f}, {100*d['hi']:+.1f}] p {d['p']:.2g} w/l {d['wins']}/{d['losses']}")
