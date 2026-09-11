"""Array-layout kernels.

Everything that knows a batch of networks is a ``(B, n, n)`` array with a zero
diagonal, and that ``actor`` is a ``(B,)`` integer array naming the focal actor
of each chain, lives here. Nothing else in the package indexes into the network
layout directly, so a torch port only has to replace this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float64
ALL_PRODUCTS = frozenset({"two_path", "shared_out", "back_path"})


@dataclass(frozen=True)
class RowProducts:
    """Per-chain row quantities for the focal actor ``i``. All ``(B, n)``.

    out_row[b, j]    = x_ij
    in_col[b, j]     = x_ji
    two_path[b, j]   = sum_h x_ih x_hj        i -> h -> j
    shared_out[b, j] = sum_h x_ih x_jh        i -> h <- j
    back_path[b, j]  = sum_h x_jh x_hi        j -> h -> i
    """

    out_row: np.ndarray
    in_col: np.ndarray
    two_path: np.ndarray | None
    shared_out: np.ndarray | None
    back_path: np.ndarray | None


def batched_row_products(
    X: np.ndarray, actor: np.ndarray, needs: frozenset[str] = ALL_PRODUCTS
) -> RowProducts:
    """Compute the row products change statistics are built from.

    ``needs`` names the derived products to compute (``two_path``,
    ``shared_out``, ``back_path``); the others come back as ``None``.
    ``out_row`` and ``in_col`` are always computed.
    """
    B = X.shape[0]
    rows = np.arange(B)
    Xf = X if X.dtype == DTYPE else X.astype(DTYPE)
    out_row = Xf[rows, actor, :]
    in_col = Xf[rows, :, actor]
    two_path = shared_out = back_path = None
    if "two_path" in needs:
        # sum_h x_ih x_hj : row vector times matrix
        two_path = (out_row[:, None, :] @ Xf)[:, 0, :]
    right = []
    if "shared_out" in needs:
        right.append(out_row)  # sum_h x_jh x_ih : matrix times column
    if "back_path" in needs:
        right.append(in_col)  # sum_h x_jh x_hi : matrix times column
    if right:
        cols = Xf @ np.stack(right, axis=-1)
        c = 0
        if "shared_out" in needs:
            shared_out = cols[:, :, c]
            c += 1
        if "back_path" in needs:
            back_path = cols[:, :, c]
    return RowProducts(out_row, in_col, two_path, shared_out, back_path)


def softmax(f: np.ndarray, axis: int = -1) -> np.ndarray:
    f = f - f.max(axis=axis, keepdims=True)
    e = np.exp(f)
    return e / e.sum(axis=axis, keepdims=True)


def categorical_sample(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draw from each row of ``p`` using exactly one uniform per row.

    Taking ``u`` as an argument (rather than an RNG) is what keeps the number of
    random draws per ministep fixed, which is what makes runs reproducible.
    """
    cdf = np.cumsum(p, axis=-1)
    idx = (u[:, None] > cdf).sum(axis=-1)
    return np.minimum(idx, p.shape[-1] - 1)


def toggle(X: np.ndarray, actor: np.ndarray, target: np.ndarray, active: np.ndarray) -> None:
    """In place: flip ``x[actor, target]`` where ``active`` and ``target != actor``.

    ``target == actor`` is the no-change option and leaves the network alone.
    """
    do = active & (target != actor)
    rows = np.arange(X.shape[0])[do]
    a, t = actor[do], target[do]
    X[rows, a, t] = 1 - X[rows, a, t]


def zero_diagonal(X: np.ndarray) -> np.ndarray:
    idx = np.arange(X.shape[-1])
    X[..., idx, idx] = 0
    return X
