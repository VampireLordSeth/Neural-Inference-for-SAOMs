"""Array-layout kernels (NumPy) and the backend interface.

Everything that knows a batch of networks is a ``(B, n, n)`` array with a zero
diagonal, and that ``actor`` is a ``(B,)`` integer array naming the focal actor
of each chain, lives in a backend. Nothing else in the package indexes into the
network layout directly. ``NumpyBackend`` wraps the module-level kernels below;
``backend_torch.TorchBackend`` implements the same interface on torch tensors.

Use ``get_backend("numpy")``, ``get_backend("torch")`` (default device: CUDA if
available), ``get_backend("torch:cpu")`` or pass a backend instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DTYPE = np.float64
OUT_DTYPE = np.int8
ALL_PRODUCTS = frozenset({"two_path", "shared_out", "back_path"})


@dataclass(frozen=True)
class RowProducts:
    """Per-chain row quantities for the focal actor ``i``. All ``(B, n)``.

    out_row[b, j]    = x_ij
    in_col[b, j]     = x_ji
    two_path[b, j]   = sum_h x_ih x_hj        i -> h -> j
    shared_out[b, j] = sum_h x_ih x_jh        i -> h <- j
    back_path[b, j]  = sum_h x_jh x_hi        j -> h -> i

    Derived products not requested via ``needs`` are ``None``.
    """

    out_row: object
    in_col: object
    two_path: object
    shared_out: object
    back_path: object


# --------------------------------------------------------------- numpy kernels


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


# ------------------------------------------------------------ backend objects


class NumpyBackend:
    """The reference backend. Every method here has a twin in ``TorchBackend``.

    Arrays handed to the model (``network``, ``array``, ``covariate``) live on
    this backend; ``to_numpy`` brings anything back.
    """

    name = "numpy"
    device = "cpu"
    dtype = DTYPE

    # -- conversion
    def network(self, X0) -> np.ndarray:
        """Float working copy of a ``(B, n, n)`` 0/1 array."""
        return np.array(X0, dtype=DTYPE, copy=True)

    def array(self, a) -> np.ndarray:
        return np.asarray(a, dtype=DTYPE)

    def int_array(self, a) -> np.ndarray:
        return np.asarray(a, dtype=np.int64)

    def covariate(self, v) -> np.ndarray:
        return np.asarray(v, dtype=DTYPE)

    def to_numpy(self, a) -> np.ndarray:
        return np.asarray(a)

    def finalize(self, X) -> np.ndarray:
        return X.astype(OUT_DTYPE)

    # -- construction
    def empty(self, shape) -> np.ndarray:
        return np.empty(shape, dtype=DTYPE)

    def arange(self, n: int) -> np.ndarray:
        return np.arange(n)

    def broadcast_to(self, a, shape):
        return np.broadcast_to(a, shape)

    def as_float(self, a):
        return a.astype(DTYPE)

    # -- randomness: the numpy Generator is the state
    def rng_state(self, rng: np.random.Generator):
        return rng

    def integers(self, n: int, B: int, state) -> np.ndarray:
        return state.integers(0, n, size=B)

    def random(self, B: int, state) -> np.ndarray:
        return state.random(B)

    # -- kernels
    def row_products(self, X, actor, needs=ALL_PRODUCTS) -> RowProducts:
        return batched_row_products(X, actor, needs)

    def softmax(self, f):
        return softmax(f)

    def categorical_sample(self, p, u):
        return categorical_sample(p, u)

    def toggle(self, X, actor, target, active) -> None:
        toggle(X, actor, target, active)

    def hamming(self, X, Y):
        """Per-chain number of differing entries: ``(B,)`` int."""
        return (X != Y).sum(axis=(1, 2))

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def clamp_min(self, a, v):
        return np.maximum(a, v)

    def __repr__(self) -> str:
        return "NumpyBackend()"


NUMPY = NumpyBackend()


def get_backend(spec="numpy"):
    """Resolve ``"numpy"``, ``"torch"``, ``"torch:<device>"`` or a backend instance."""
    if not isinstance(spec, str):
        return spec
    if spec == "numpy":
        return NUMPY
    if spec == "torch" or spec.startswith("torch:"):
        from .backend_torch import TorchBackend

        device = spec.partition(":")[2] or None
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend {spec!r}")
