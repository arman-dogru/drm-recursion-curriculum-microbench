"""Shared recursive block Phi, embeddings and decoder (paper §3.1, Eq. 1-2).

H_t = Phi(H_{t-1}, X) = Block(H_{t-1} + E_in(X)), with E_in including 2-D
positional embeddings. The output/label stream has its own embedding table
(colours + MASK), as in paper §4.3 ("separate token embeddings for mapping the
masked ground truth into latent space").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import H, W, M, N_COLORS, MASK


def local_mask(radius: int) -> torch.Tensor | None:
    """Boolean (M, M) attention mask: cell i may attend to cell j iff their
    Chebyshev distance is <= radius. radius <= 0 means global attention.
    With a local mask, information moves at most `radius` cells per Phi step,
    so each task has a known minimum recursion depth."""
    if radius <= 0:
        return None
    r = torch.arange(M) // W
    c = torch.arange(M) % W
    return ((r[:, None] - r[None, :]).abs().maximum((c[:, None] - c[None, :]).abs())) <= radius


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_mult: int = 4, radius: int = 0):
        super().__init__()
        self.register_buffer("attn_mask", local_mask(radius), persistent=False)
        self.ln1 = nn.LayerNorm(d)
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def forward(self, h):
        b, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask).transpose(1, 2).reshape(b, n, d)
        h = h + self.proj(a)
        return h + self.mlp(self.ln2(h))


class RecursiveModel(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, layers: int = 1, radius: int = 1):
        super().__init__()
        self.d = d
        self.in_emb = nn.Embedding(N_COLORS, d)
        self.lab_emb = nn.Embedding(N_COLORS + 1, d)   # + MASK
        self.row = nn.Parameter(torch.randn(H, d) * 0.02)
        self.col = nn.Parameter(torch.randn(W, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads, radius=radius) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        self.dec = nn.Linear(d, N_COLORS)

    def pos(self):
        return (self.row[:, None, :] + self.col[None, :, :]).reshape(M, self.d)

    def embed_x(self, x):
        return self.in_emb(x) + self.pos()

    def embed_y(self, y_tokens):
        """y_tokens in 0..MASK -> initial latent H_0."""
        return self.lab_emb(y_tokens) + self.pos()

    def phi(self, h, ex):
        """One application of the shared transition operator."""
        h = h + ex
        for b in self.blocks:
            h = b(h)
        return h

    def logits(self, h):
        return self.dec(self.ln_out(h))

    def unroll(self, h, ex, k: int):
        for _ in range(k):
            h = self.phi(h, ex)
        return h


def full_mask(n: int) -> torch.Tensor:
    return torch.full((n, M), MASK, dtype=torch.long)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
