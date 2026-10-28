import math

import numpy as np
import pytest
import torch

import drm.train as T
from drm.config import Config
from drm.model import RecursiveModel
from drm.tasks import MASK, make_task_data
from helpers import tiny


def _data(cfg):
    return make_task_data(cfg.task, cfg.n_train, cfg.n_dev, cfg.n_eval)["train"]


class Recorder:
    """Wraps model methods during training to record what the trainer feeds them."""

    def __init__(self, monkeypatch):
        self.embed_y_tokens, self.unroll_calls, self.ce_targets, self.phi_calls = [], [], [], 0
        rec = self
        orig_embed_y, orig_unroll, orig_phi = (RecursiveModel.embed_y, RecursiveModel.unroll,
                                               RecursiveModel.phi)

        def embed_y(self_, tok):
            rec.embed_y_tokens.append(tok.clone())
            return orig_embed_y(self_, tok)

        def unroll(self_, h, ex, k):
            rec.unroll_calls.append((h.grad_fn is None, k))
            return orig_unroll(self_, h, ex, k)

        def phi(self_, h, ex):
            rec.phi_calls += 1
            return orig_phi(self_, h, ex)

        orig_ce = T.F.cross_entropy

        def ce(logits, target):
            rec.ce_targets.append(target.clone())
            return orig_ce(logits, target)

        monkeypatch.setattr(RecursiveModel, "embed_y", embed_y)
        monkeypatch.setattr(RecursiveModel, "unroll", unroll)
        monkeypatch.setattr(RecursiveModel, "phi", phi)
        monkeypatch.setattr(T.F, "cross_entropy", ce)


@pytest.mark.parametrize("method", ["fwd", "fwd_steps", "bwd_k", "bwd_tbptt", "drm"])
def test_budget_and_step_accounting(method, monkeypatch):
    cfg = tiny(method)
    rec = Recorder(monkeypatch)
    _, log = T.train(cfg, *_data(cfg))
    assert log["opt_steps"] == cfg.opt_steps() == len(rec.ce_targets)
    assert log["phi_used"] == rec.phi_calls == cfg.phi_used()
    if method == "fwd_steps":
        assert cfg.opt_steps() == tiny("drm").opt_steps()
        assert log["phi_used"] == cfg.phi_budget // cfg.ref_k
    else:
        assert log["phi_used"] == cfg.phi_budget        # FLOP-matched
    inner = 1 if method.startswith("fwd") else cfg.k
    assert all(k == inner for _, k in rec.unroll_calls)


@pytest.mark.parametrize("method", ["fwd", "drm"])
def test_forward_and_drm_start_from_corrupted_target(method, monkeypatch):
    cfg = tiny(method, phi_budget=200)
    rec = Recorder(monkeypatch)
    X, Y = _data(cfg)
    T.train(cfg, X, Y)
    toks = torch.cat(rec.embed_y_tokens)
    targets = torch.cat(rec.ce_targets).reshape(toks.shape)
    vis = toks != MASK
    assert vis.any() and (~vis).any()                       # partially masked states seen
    assert torch.equal(toks[vis], targets[vis])             # visible cells are the clean target
    frac = (~vis).float().mean(1)
    assert frac.min() < 0.2 and frac.max() > 0.8            # tau spans easy..hard


@pytest.mark.parametrize("method", ["bwd_k", "bwd_tbptt"])
def test_backward_starts_from_full_mask(method, monkeypatch):
    cfg = tiny(method)
    rec = Recorder(monkeypatch)
    T.train(cfg, *_data(cfg))
    assert all((t == MASK).all() for t in rec.embed_y_tokens)


def test_loss_targets_are_clean_labels(monkeypatch):
    cfg = tiny("drm")
    rec = Recorder(monkeypatch)
    X, Y = _data(cfg)
    T.train(cfg, X, Y)
    ys = {tuple(r) for r in Y.tolist()}
    for t in rec.ce_targets:
        for row in t.reshape(cfg.batch, -1).tolist():
            assert tuple(row) in ys


def test_tbptt_detaches_between_windows(monkeypatch):
    cfg = tiny("bwd_tbptt")
    rec = Recorder(monkeypatch)
    T.train(cfg, *_data(cfg))
    detached = [d for d, _ in rec.unroll_calls]
    # first window of each batch starts from an embedding (has grad_fn), later windows are detached
    expect = ([False] + [True] * (cfg.n_sup - 1)) * (cfg.opt_steps() // cfg.n_sup)
    assert detached == expect
    assert len(rec.embed_y_tokens) == cfg.opt_steps() // cfg.n_sup


def test_drm_single_window_no_detach(monkeypatch):
    cfg = tiny("drm")
    rec = Recorder(monkeypatch)
    T.train(cfg, *_data(cfg))
    assert all(not d for d, _ in rec.unroll_calls)
    assert len(rec.embed_y_tokens) == cfg.opt_steps()


def test_gradient_flows_through_all_k_steps():
    torch.manual_seed(0)
    m = RecursiveModel(16, 2, 1)
    x = torch.randint(0, 8, (2, 36))
    h0 = m.embed_y(torch.full((2, 36), MASK)).detach().requires_grad_(True)
    h = m.unroll(h0, m.embed_x(x), 3)
    h.sum().backward()
    assert h0.grad is not None and h0.grad.abs().sum() > 0


def test_lr_schedule():
    assert T.lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert T.lr_at(9, 100, 1.0, 10) == pytest.approx(1.0)
    assert T.lr_at(100, 100, 1.0, 10) == pytest.approx(0.1)
    assert T.lr_at(55, 100, 1.0, 10) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))


def test_training_is_seed_deterministic():
    cfg = tiny("drm")
    X, Y = _data(cfg)
    m1, l1 = T.train(cfg, X, Y)
    m2, l2 = T.train(cfg, X, Y)
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    m3, _ = T.train(tiny("drm", seed=2), X, Y)
    assert not all(torch.equal(a, b) for a, b in zip(m1.parameters(), m3.parameters()))


def test_training_reduces_loss():
    cfg = tiny("drm", phi_budget=400, task="colormap")
    _, log = T.train(cfg, *_data(cfg))
    assert log["loss"][-1][2] < log["loss"][0][2]
