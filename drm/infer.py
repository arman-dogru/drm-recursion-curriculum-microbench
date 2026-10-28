"""Matched inference: start from a full mask and iterate (paper §3.2 / §4.3).

S outer denoising steps; each outer step re-embeds the current token grid,
applies Phi k_in times, decodes the argmax grid, then re-masks floor(r(s_i) M)
uniformly random cells, with s_i = 1 - i/S (so the last step leaves no mask).
S = 1 is pure latent recursion of k_in steps (backward-style inference).
Tokens are not frozen (paper: "we do not freeze tokens").
"""
from __future__ import annotations

import numpy as np
import torch

from .corruption import corrupt, n_masked
from .model import full_mask
from .tasks import MASK


@torch.no_grad()
def infer(model, X: torch.Tensor, S: int, k_in: int, gen: torch.Generator, trace=None, Y=None):
    model.eval()
    n = X.shape[0]
    ex = model.embed_x(X)
    tokens = full_mask(n)
    for i in range(1, S + 1):
        h = model.unroll(model.embed_y(tokens), ex, k_in)
        pred = model.logits(h).argmax(-1)
        if trace is not None:
            vis = tokens != MASK
            wrong_vis = vis & (tokens != Y)
            fixed = wrong_vis & (pred == Y)
            trace.append({
                "step": i,
                "visible_frac": vis.float().mean().item(),
                "visible_wrong_frac": (wrong_vis.sum() / vis.sum().clamp(min=1)).item(),
                "self_correct_rate": (fixed.sum() / wrong_vis.sum().clamp(min=1)).item()
                if wrong_vis.any() else float("nan"),
                "cell_acc": (pred == Y).float().mean().item(),
                "exact": (pred == Y).all(1).float().mean().item(),
            })
        if i < S:
            s = 1.0 - i / S
            tau = torch.full((n,), s)
            tokens = corrupt(pred, tau, gen)
    model.train()
    return pred


@torch.no_grad()
def oracle_state_acc(model, X, Y, S: int, k_in: int, gen: torch.Generator):
    """Per-step accuracy when the input state is the *training* distribution:
    the clean target corrupted at the same mask level the sampler would use."""
    model.eval()
    n = X.shape[0]
    ex = model.embed_x(X)
    out = []
    for i in range(1, S + 1):
        s_prev = 1.0 - (i - 1) / S
        tokens = corrupt(Y, torch.full((n,), s_prev), gen)
        h = model.unroll(model.embed_y(tokens), ex, k_in)
        pred = model.logits(h).argmax(-1)
        out.append({"step": i, "mask_frac": (tokens == MASK).float().mean().item(),
                    "cell_acc": (pred == Y).float().mean().item(),
                    "exact": (pred == Y).all(1).float().mean().item()})
    model.train()
    return out


def exact_match(pred: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    return (pred == Y).all(1).numpy()
