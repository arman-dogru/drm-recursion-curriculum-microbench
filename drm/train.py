"""The three training objectives on a shared Phi (paper §3.1.1, §3.1.2, §3.2).

fwd        forward one-step denoising: H0 = E_y(q_tau(Y)), one Phi, CE to clean Y   (Eq. 3)
fwd_steps  same objective, step-matched instead of FLOP-matched
bwd_k      backward recursion from full mask, k Phi steps, CE at the end (single window)
bwd_tbptt  backward recursion from full mask, n_sup windows of k steps, stop-gradient
           at window boundaries, CE + optimiser step at the end of every window    (Eq. 4-5)
drm        DRM: H0 = E_y(q_tau(Y)), k Phi steps, CE to clean Y                    (Eq. 6)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .corruption import corrupt
from .model import RecursiveModel, full_mask


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def initial_state(model, cfg: Config, y, gen):
    """H_0 for a training batch: corrupted target (fwd/drm) or full mask (bwd)."""
    if cfg.method in ("fwd", "fwd_steps", "drm"):
        tau = torch.rand(y.shape[0], generator=gen)
        return model.embed_y(corrupt(y, tau, gen))
    return model.embed_y(full_mask(y.shape[0]))


def train(cfg: Config, Xtr: np.ndarray, Ytr: np.ndarray, curve_fn=None):
    """Train a fresh model. Returns (model, log dict)."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed * 7919 + 17)
    model = RecursiveModel(cfg.d, cfg.heads, cfg.layers, cfg.radius)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    X = torch.from_numpy(Xtr)
    Y = torch.from_numpy(Ytr)
    total = cfg.opt_steps()
    windows = cfg.n_sup if cfg.method == "bwd_tbptt" else 1
    n_batches = total // windows
    inner = 1 if cfg.method in ("fwd", "fwd_steps") else cfg.k
    curve_every = max(1, total // cfg.curve_points)
    log = {"loss": [], "curve": [], "opt_steps": 0, "phi_used": 0}
    t0 = time.time()
    step = 0
    run_loss = 0.0
    for _ in range(n_batches):
        idx = torch.randint(0, X.shape[0], (cfg.batch,), generator=gen)
        x, y = X[idx], Y[idx]
        h = initial_state(model, cfg, y, gen)
        for _w in range(windows):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg.lr, cfg.warmup)
            ex = model.embed_x(x)
            h = model.unroll(h, ex, inner)
            loss = F.cross_entropy(model.logits(h).reshape(-1, model.dec.out_features), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            h = h.detach()                       # TBPTT stop-gradient (Eq. 4)
            log["phi_used"] += inner
            step += 1
            run_loss += loss.item()
            if step % 50 == 0:
                log["loss"].append([step, log["phi_used"], run_loss / 50, time.time() - t0])
                run_loss = 0.0
            if curve_fn is not None and (step % curve_every == 0):
                log["curve"].append([step, log["phi_used"], curve_fn(model)])
    log["opt_steps"] = step
    log["train_seconds"] = time.time() - t0
    return model, log
