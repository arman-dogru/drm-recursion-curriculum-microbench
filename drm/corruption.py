"""Masking forward process q_tau (paper §4.3, Eq. 8).

alpha_bar(tau) = cos^2(pi*tau/2); fraction masked r(tau) = 1 - alpha_bar(tau).
floor(r*M) positions are chosen uniformly at random and replaced by MASK.
"""
from __future__ import annotations

import math

import torch

from .tasks import M, MASK


def mask_fraction(tau):
    if isinstance(tau, torch.Tensor):
        return 1.0 - torch.cos(math.pi * tau / 2) ** 2
    return 1.0 - math.cos(math.pi * tau / 2) ** 2


def n_masked(tau) -> torch.Tensor:
    return torch.floor(mask_fraction(tau) * M + 1e-6).long()


def corrupt(y: torch.Tensor, tau: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Mask floor(r(tau_i)*M) uniformly-random cells of each row of y."""
    n = n_masked(tau)                                   # (B,)
    scores = torch.rand(y.shape, generator=gen)
    ranks = scores.argsort(dim=1).argsort(dim=1)        # random rank per cell
    masked = ranks < n[:, None]
    return torch.where(masked, torch.full_like(y, MASK), y)
