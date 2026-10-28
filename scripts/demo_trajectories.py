"""Re-train one DRM and one forward (one-step) model on flood at eval seed 100
with their dev-tuned lr, and record full token-grid trajectories for a few eval
examples. Output: results/demo/trajectories.json (used for the trajectory figure).

The retrained models are bit-identical to the main-grid seed-100 runs (same
config, deterministic training), which the script checks via pass@1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drm.config import ROOT, Config  # noqa: E402
from drm.corruption import corrupt  # noqa: E402
from drm.model import full_mask  # noqa: E402
from drm.tasks import make_task_data  # noqa: E402
from drm.train import train  # noqa: E402
from run_grid import load_tuning  # noqa: E402
from run_one import eval_gen, out_paths  # noqa: E402

TASK, SEED, N_SHOW = "flood", 100, 6


@torch.no_grad()
def trajectory(model, X, S, k, gen):
    model.eval()
    ex = model.embed_x(X)
    tokens = full_mask(X.shape[0])
    steps = []
    for i in range(1, S + 1):
        pred = model.logits(model.unroll(model.embed_y(tokens), ex, k)).argmax(-1)
        steps.append({"input": tokens.tolist(), "pred": pred.tolist()})
        if i < S:
            tokens = corrupt(pred, torch.full((X.shape[0],), 1.0 - i / S), gen)
    return steps, pred


def main():
    torch.set_num_threads(1)
    tuning = load_tuning()
    out = {"task": TASK, "seed": SEED, "methods": {}}
    for m in ("drm", "fwd"):
        cfg = Config(task=TASK, method=m, seed=SEED, split="eval", lr=tuning[TASK][m]["lr"])
        data = make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval, cfg.data_seed)
        model, _ = train(cfg, *data["train"])
        X, Y = (torch.from_numpy(a) for a in data["eval"])
        S, k = tuning[TASK][m]["schedule"]
        _, full = trajectory(model, X, S, k, eval_gen(cfg.seed, S, k))
        p1 = float((full == Y).all(1).float().mean())
        ref = json.loads(out_paths(cfg, ROOT / "results" / "raw")[0].read_text())["eval"][f"S{S}_k{k}"]["pass1"]
        assert abs(p1 - ref) < 1e-9, (m, p1, ref)
        idx = list(range(N_SHOW))
        steps, _ = trajectory(model, X[idx], S, k, torch.Generator().manual_seed(0))
        out["methods"][m] = {"schedule": [S, k], "pass1_check": p1, "x": X[idx].tolist(), "y": Y[idx].tolist(),
                             "steps": steps}
        print(m, S, k, p1, flush=True)
    d = ROOT / "results" / "demo"
    d.mkdir(parents=True, exist_ok=True)
    (d / "trajectories.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
