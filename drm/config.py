"""Run configuration, budget accounting and provenance hashes."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fwd", "fwd_steps", "bwd_k", "bwd_tbptt", "drm")
DEV_SEEDS = (1, 2)
EVAL_SEEDS = tuple(range(100, 112))           # 12 seeds, disjoint from DEV_SEEDS
# Round 1 review (M2): extended to 1.6e-2 for every task and method so dev optima are interior.
LR_GRID = (5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2)
# inference menu at matched test compute (outer steps S x inner steps k_in = 16)
MATCHED_MENU = ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1))
S_GRID = (1, 2, 4, 8, 16)
# Round 1 review (M3): k_in up to 64 so pure latent recursion (S=1) is offered at every budget.
K_GRID = (1, 2, 4, 8, 16, 32, 64)
MAX_TEST_PHI = 64


def inference_grid():
    return [(s, k) for s in S_GRID for k in K_GRID if s * k <= MAX_TEST_PHI]


@dataclasses.dataclass(frozen=True)
class Config:
    task: str
    method: str
    seed: int
    split: str = "dev"            # evaluation split: dev (tuning) or eval (final)
    lr: float = 1e-3
    k: int = 4                    # recursion window (gradient-bearing Phi steps)
    n_sup: int = 4                # TBPTT windows per batch (bwd_tbptt only)
    phi_budget: int = 12000       # gradient-bearing Phi applications per batch row, summed over training
    ref_k: int = 4                # k used to define the budget for step-matched baselines
    batch: int = 64
    wd: float = 0.1
    warmup: int = 100
    d: int = 64
    heads: int = 4
    layers: int = 1
    radius: int = 1               # local attention radius (0 = global)
    n_train: int = 2000
    n_dev: int = 500
    n_eval: int = 1000
    data_seed: int = 0
    curve_points: int = 8         # learning-curve checkpoints on a dev subset
    curve_n: int = 200

    def __post_init__(self):
        from .tasks import TASKS
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; choose from {TASKS}")
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.split not in ("dev", "eval"):
            raise ValueError(f"split must be dev or eval, got {self.split!r}")
        if self.k < 1 or self.n_sup < 1 or self.ref_k < 1:
            raise ValueError("k, n_sup and ref_k must be >= 1")
        if self.lr <= 0 or self.phi_budget < 0 or self.batch < 1:
            raise ValueError("lr must be > 0, phi_budget >= 0, batch >= 1")

    def phi_per_step(self) -> int:
        """Gradient-bearing Phi applications per optimiser step."""
        return {"fwd": 1, "fwd_steps": 1, "bwd_k": self.k, "bwd_tbptt": self.k,
                "drm": self.k}[self.method]

    def opt_steps(self) -> int:
        if self.method == "fwd_steps":       # step-matched: same #opt steps as DRM, 1/k FLOPs
            return self.phi_budget // self.ref_k
        return self.phi_budget // self.phi_per_step()

    def phi_used(self) -> int:
        return self.opt_steps() * self.phi_per_step()

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def run_name(self) -> str:
        lr = f"{self.lr:g}".replace(".", "p")    # exact (1.6e-2 -> 0p016), so 1.6e-2 and 2e-2 cannot collide
        return f"{self.split}/{self.task}/{self.method}_k{self.k}_lr{lr}_s{self.seed}"


CODE_FILES = ("drm/__init__.py", "drm/tasks.py", "drm/model.py", "drm/corruption.py",
              "drm/train.py", "drm/infer.py", "drm/config.py", "scripts/run_one.py")


def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:16]
