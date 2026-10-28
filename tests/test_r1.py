"""Tests added in response to the round-1 review (m3, m7, M3, m4)."""
import json

import numpy as np
import pytest
import torch

import drm.infer as I
import drm.train as T
import run_one
import tune
from drm.config import LR_GRID, MATCHED_MENU, Config, inference_grid
from drm.model import RecursiveModel
from drm.stats import bootstrap_ci, t_ci
from drm.tasks import make_task_data
from helpers import tiny


# ---------- m3: reviewer mutants X5 (zero_grad), X12 (inference RNG), X3 (CI level)

@pytest.mark.parametrize("method", ["fwd", "bwd_tbptt", "drm"])
def test_grads_are_zeroed_before_every_backward(method, monkeypatch):
    """Each optimizer step must use only the current loss's gradient (no accumulation)."""
    events = []
    orig_zero, orig_step = torch.optim.AdamW.zero_grad, torch.optim.AdamW.step
    orig_backward = torch.Tensor.backward

    def zero(self, set_to_none=True):
        events.append("Z")
        return orig_zero(self, set_to_none=set_to_none)

    def step(self, *a, **k):
        events.append("S")
        return orig_step(self, *a, **k)

    def backward(self, *a, **k):
        events.append("B")
        return orig_backward(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", zero)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    cfg = tiny(method)
    T.train(cfg, *make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval)["train"])
    assert "".join(events) == "ZBS" * cfg.opt_steps()


def test_infer_uses_the_given_generator(monkeypatch):
    seen = []
    orig = I.corrupt

    def spy(pred, tau, gen):
        seen.append(gen)
        return orig(pred, tau, gen)

    monkeypatch.setattr(I, "corrupt", spy)
    torch.manual_seed(0)
    model = RecursiveModel(16, 2, 1)
    X = torch.randint(0, 8, (5, 36))
    g = torch.Generator().manual_seed(123)
    I.infer(model, X, 4, 1, g)
    assert len(seen) == 3 and all(s is g for s in seen)


def test_inference_output_depends_on_generator_seed():
    torch.manual_seed(0)
    model = RecursiveModel(16, 2, 1)
    X = torch.randint(0, 8, (64, 36))
    a = I.infer(model, X, 8, 1, torch.Generator().manual_seed(1))
    b = I.infer(model, X, 8, 1, torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_run_one_passes_per_seed_eval_generators(tmp_path, monkeypatch):
    seeds = []
    real = run_one.infer

    def spy(model, X, S, k, gen, **kw):
        seeds.append((S, k, gen.initial_seed()))
        return real(model, X, S, k, gen, **kw)

    monkeypatch.setattr(run_one, "infer", spy)
    cfg = tiny("drm", split="eval", seed=5)
    run_one.run(cfg, tmp_path)
    grid = seeds[cfg.curve_points: cfg.curve_points + len(inference_grid())]
    assert {(S, k): s for S, k, s in grid} == {(S, k): run_one.eval_gen(5, S, k).initial_seed()
                                               for S, k in inference_grid()}
    assert run_one.eval_gen(5, 4, 4).initial_seed() != run_one.eval_gen(6, 4, 4).initial_seed()


def test_bootstrap_ci_is_95_percent():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, size=400)
    m, lo, hi = bootstrap_ci(v, n_boot=20_000)
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert (hi - lo) / (2 * se) == pytest.approx(1.96, rel=0.05)     # a 90% CI would give 1.645


def test_t_ci():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    m, lo, hi = t_ci(v)
    assert m == pytest.approx(0.25)
    assert hi - m == pytest.approx(3.182446 * v.std(ddof=1) / 2, rel=1e-5)
    assert t_ci(np.ones(5)) == (1.0, 1.0, 1.0)


# ---------- m7: input validation and run-name collisions

@pytest.mark.parametrize("kw", [dict(task="bogus"), dict(method="bogus"), dict(k=0), dict(lr=0.0),
                                dict(phi_budget=-5), dict(split="test"), dict(n_sup=0)])
def test_config_rejects_bad_values(kw):
    base = dict(task="flood", method="drm", seed=1)
    base.update(kw)
    with pytest.raises(ValueError):
        Config(**base)


@pytest.mark.parametrize("argv", [["--task", "bogus"], ["--method", "bogus"], ["--k", "0"],
                                  ["--phi_budget", "-5"], ["--phi_budget", "0"], ["--lr", "-1"]])
def test_cli_rejects_bad_values(argv):
    base = {"--task": "flood", "--method": "drm", "--seed": "1"}
    args = []
    for k, v in base.items():
        if k not in argv:
            args += [k, v]
    with pytest.raises(SystemExit) as e:
        run_one.parse(args + argv)
    assert e.value.code == 2


def test_run_names_never_collide(tmp_path):
    lrs = LR_GRID + (2e-2, 1.5e-3, 3e-3, 3.2e-2, 1e-2)
    names = {Config("flood", "drm", 1, lr=lr).run_name() for lr in lrs}
    assert len(names) == len(lrs)
    paths = {run_one.out_paths(Config("flood", "drm", 1, lr=lr), tmp_path)[0] for lr in lrs}
    assert len(paths) == len(lrs)
    assert all("." not in p.stem for p in paths)


# ---------- M3: latent inference is offered at every compute budget

def test_inference_grid_offers_pure_latent_at_every_budget():
    grid = set(inference_grid())
    for c in sorted({s * k for s, k in grid}):
        assert (1, c) in grid
    assert (1, 32) in grid and (1, 64) in grid
    assert set(MATCHED_MENU) <= grid


def test_by_compute_can_pick_latent_beyond_16():
    cfg = Config("flood", "bwd_tbptt", 1)

    def d(c, lr):
        return {"meta": {}, "eval": {f"S{s}_k{k}": {"S": s, "k_in": k,
                                                     "pass1": 0.9 if (s, k) == (1, 64) else 0.1}
                                      for s, k in inference_grid()}}
    import dataclasses
    runs = {dataclasses.replace(cfg, task=t, method=m, seed=s, lr=lr): d(None, lr)
            for t in ("flood",) for m in ("bwd_tbptt",) for s in (1, 2) for lr in LR_GRID}
    _, _, by = tune.choose(runs)
    assert by["flood"]["bwd_tbptt"]["64"]["schedule"] == [1, 64]


def test_lr_edges():
    sc = {f"{lr:g}": {f"S{s}_k{k}": 0.0 for s, k in MATCHED_MENU} for lr in LR_GRID}
    sc[f"{LR_GRID[-1]:g}"]["S4_k4"] = 0.5
    sc[f"{LR_GRID[-2]:g}"]["S4_k4"] = 0.4
    choice = {"flood": {"drm": {"lr": LR_GRID[-1], "dev_pass1": 0.5}}}
    assert tune.lr_edges(choice, {"flood": {"drm": sc}}) == [f"flood:drm:{LR_GRID[-1]:g}"]
    sc[f"{LR_GRID[-2]:g}"]["S4_k4"] = 0.5          # tie with neighbour -> not an edge
    choice = {"flood": {"drm": {"lr": LR_GRID[-2], "dev_pass1": 0.5}}}
    assert tune.lr_edges(choice, {"flood": {"drm": sc}}) == []


# ---------- m4: provenance (timestamps, load, wall time) and checkpoints

def test_meta_has_timestamps_and_checkpoint_reproduces(tmp_path):
    cfg = tiny("drm", split="eval")
    out = run_one.run(cfg, tmp_path)
    meta = out["meta"]
    for k in ("started_at", "finished_at", "wall_seconds", "loadavg_start", "loadavg_end", "cpu_count"):
        assert k in meta
    assert meta["started_at"] <= meta["finished_at"] and "T" in meta["started_at"]
    ck = run_one.ckpt_path(cfg, tmp_path)
    assert ck.exists()
    model = RecursiveModel(cfg.d, cfg.heads, cfg.layers, cfg.radius)
    model.load_state_dict(torch.load(ck))
    data = make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval, cfg.data_seed)
    Xe, Ye = (torch.from_numpy(a) for a in data["eval"])
    p = I.exact_match(I.infer(model, Xe, 4, 4, run_one.eval_gen(cfg.seed, 4, 4)), Ye).mean()
    assert p == pytest.approx(out["eval"]["S4_k4"]["pass1"])
    js, _ = run_one.out_paths(cfg, tmp_path)
    assert json.loads(js.read_text())["meta"]["started_at"] == meta["started_at"]
