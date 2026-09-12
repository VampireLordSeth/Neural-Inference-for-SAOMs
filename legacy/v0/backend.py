"""
Batched array kernels.

Everything in this package that touches raw array layout lives here, so that
porting to a GPU framework is a localised change rather than a rewrite. Each
function below has a direct equivalent in PyTorch:

    numpy                             torch
    ---------------------------------------------------------------
    X[batch_idx, actor]               X[torch.arange(B), actor]
    A @ B                             torch.bmm(A, B)
    softmax (implemented here)        torch.softmax
    rng.random / rng.poisson          torch.rand / torch.poisson

The per-micro-step cost is three batched matrix-vector products, so O(B n^2),
not O(B n^3). That is the property that makes many parallel chains affordable.
"""

from __future__ import annotations

import numpy as np


def batched_row_products(X: np.ndarray, actor: np.ndarray):
    """Precompute the row quantities every effect needs.

    Parameters
    ----------
    X : (B, n, n) float array
        Adjacency matrices with zero diagonal.
    actor : (B,) int array
        Focal actor for each chain.

    Returns
    -------
    xi : (B, n)          X[b, i, :]          outgoing ties of focal actor
    xTi : (B, n)         X[b, :, i]          incoming ties of focal actor
    two_path : (B, n)    sum_h x_ih x_hj
    shared_out : (B, n)  sum_h x_ih x_jh
    cyc : (B, n)         sum_h x_jh x_hi
    """
    B = X.shape[0]
    rows = np.arange(B)

    xi = X[rows, actor, :]                      # (B, n)
    xTi = X[rows, :, actor]                     # (B, n)

    # sum_h x_ih x_hj  ->  row vector times matrix
    two_path = np.einsum("bh,bhj->bj", xi, X)

    # sum_h x_ih x_jh  ->  matrix times row vector
    shared_out = np.einsum("bjh,bh->bj", X, xi)

    # sum_h x_jh x_hi  ->  matrix times column vector
    cyc = np.einsum("bjh,bh->bj", X, xTi)

    return xi, xTi, two_path, shared_out, cyc


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def categorical_sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one index per row from a (B, n) array of probabilities.

    Uses inverse-CDF sampling on a single uniform draw per row, which keeps the
    number of random draws per micro-step fixed at one per chain. That matters
    for reproducibility: the stream position depends only on the step count,
    not on the data.
    """
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0                      # guard against float drift
    u = rng.random((probs.shape[0], 1))
    return (u > cdf).sum(axis=1)


def toggle(X: np.ndarray, chain: np.ndarray, i: np.ndarray, j: np.ndarray) -> None:
    """Flip entries X[chain, i, j] in place (0 <-> 1)."""
    X[chain, i, j] = 1.0 - X[chain, i, j]
