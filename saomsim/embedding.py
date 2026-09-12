"""Learned embedding of a network panel for the M2 estimator (torch).

The flow in ``benchmarks/npe_m2.py`` sees 29 hand summaries. This module
replaces that input with a learned, permutation-invariant embedding of the raw
panel — both waves, the covariates and n — with the hand summaries appended
as a residual, so the network only has to learn what the summaries miss.

Input layout (one flat float32 vector per panel, produced by ``pack_panel``):

    [ X0 packed (N_MAX * N_MAX/8 bytes) | X1 packed | v (N_MAX) | g (N_MAX, -1 pad)
      | n | hand summaries (transformed, S) ]

The packed bytes travel as floats holding integers 0..255 so sbi can store
the whole training set as one tensor; ``PanelEmbedding.forward`` unpacks the
bits on the device. Actors beyond n are masked everywhere.

Architecture: per-actor features (degrees, mutual and changed ties in each
wave, v, one-hot g) -> MLP -> two rounds of masked mean aggregation over
out-neighbours and in-neighbours in X0 and X1 -> masked mean + max pooling
-> concatenated with n and the hand summaries -> MLP -> embedding.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .population import N_MAX

PACK_BYTES = N_MAX * (N_MAX // 8)
G_CATS = 4


def pack_panel(
    X0p: np.ndarray, X1p: np.ndarray, v: np.ndarray, g: np.ndarray, n: np.ndarray, S: np.ndarray
):
    """Assemble the flat float32 input from stored (packed) arrays. Shapes:
    X0p/X1p (B, N_MAX, N_MAX/8) uint8, v (B, N_MAX), g (B, N_MAX), n (B,), S (B, S)."""
    B = X0p.shape[0]
    parts = [
        X0p.reshape(B, -1).astype(np.float32),
        X1p.reshape(B, -1).astype(np.float32),
        v.astype(np.float32),
        g.astype(np.float32),
        n.astype(np.float32)[:, None],
        S.astype(np.float32),
    ]
    return np.concatenate(parts, axis=1)


def layout(n_summaries: int) -> dict:
    o = 0
    out = {}
    for name, size in (
        ("X0", PACK_BYTES),
        ("X1", PACK_BYTES),
        ("v", N_MAX),
        ("g", N_MAX),
        ("n", 1),
        ("S", n_summaries),
    ):
        out[name] = (o, o + size)
        o += size
    out["total"] = o
    return out


def _unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """(B, N_MAX * N_MAX/8) float in 0..255 -> (B, N_MAX, N_MAX) float 0/1 (big-endian bits)."""
    B = packed.shape[0]
    x = packed.round().to(torch.int64).view(B, N_MAX, N_MAX // 8, 1)
    shifts = torch.arange(7, -1, -1, device=packed.device)
    bits = (x >> shifts) & 1
    return bits.view(B, N_MAX, N_MAX).to(torch.float32)


def _masked_mean(h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """mean over j of h_j weighted by A_ij; zero where the row has no ties. h (B,N,d), A (B,N,N)."""
    deg = A.sum(-1, keepdim=True)
    return (A @ h) / deg.clamp(min=1.0)


class PanelEmbedding(nn.Module):
    def __init__(self, n_summaries: int, hidden: int = 64, out_dim: int = 64, rounds: int = 2):
        super().__init__()
        self.lay = layout(n_summaries)
        self.rounds = rounds
        f_in = 10 + G_CATS  # per-actor raw features
        self.actor_in = nn.Sequential(
            nn.Linear(f_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.msg = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(5 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                for _ in range(rounds)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 1 + n_summaries, 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, out_dim),
        )

    def split(self, x: torch.Tensor):
        L = self.lay
        sl = {k: x[:, L[k][0] : L[k][1]] for k in ("X0", "X1", "v", "g", "n", "S")}
        X0 = _unpack_bits(sl["X0"])
        X1 = _unpack_bits(sl["X1"])
        n = sl["n"]  # (B, 1)
        idx = torch.arange(N_MAX, device=x.device)[None, :]
        mask = (idx < n).to(torch.float32)  # (B, N_MAX)
        return X0, X1, sl["v"], sl["g"], n, mask, sl["S"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X0, X1, v, g, n, mask, S = self.split(x)
        m2 = mask[:, :, None] * mask[:, None, :]
        X0 = X0 * m2
        X1 = X1 * m2
        scale = 1.0 / n.clamp(min=1.0)  # normalise degrees by n
        out0, in0 = X0.sum(-1), X0.sum(-2)
        out1, in1 = X1.sum(-1), X1.sum(-2)
        mut0 = (X0 * X0.transpose(1, 2)).sum(-1)
        mut1 = (X1 * X1.transpose(1, 2)).sum(-1)
        ch = (X0 != X1).to(torch.float32)
        ch_out, ch_in = ch.sum(-1), ch.sum(-2)
        g_int = g.round().to(torch.int64).clamp(min=0, max=G_CATS - 1)
        g_oh = nn.functional.one_hot(g_int, G_CATS).to(torch.float32) * mask[:, :, None]
        feats = (
            torch.stack([out0, in0, out1, in1, mut0, mut1, ch_out, ch_in], dim=-1)
            * scale[:, :, None]
        )
        feats = torch.cat([feats, v[:, :, None], mask[:, :, None], g_oh], dim=-1)
        h = self.actor_in(feats) * mask[:, :, None]
        for mlp in self.msg:
            agg = torch.cat(
                [
                    h,
                    _masked_mean(h, X0),
                    _masked_mean(h, X0.transpose(1, 2)),
                    _masked_mean(h, X1),
                    _masked_mean(h, X1.transpose(1, 2)),
                ],
                dim=-1,
            )
            h = (h + mlp(agg)) * mask[:, :, None]
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled_mean = h.sum(1) / denom
        pooled_max = (h + (mask[:, :, None] - 1.0) * 1e4).max(1).values
        z = torch.cat([pooled_mean, pooled_max, torch.log(n.clamp(min=1.0)), S], dim=-1)
        return self.head(z)
