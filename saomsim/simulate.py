"""Batched simulation of the stochastic actor-oriented model.

One period of the basic SAOM (constant rate, no endowment/creation split):

  * the number of ministeps in a chain is Poisson(n * rate)
  * at each ministep one actor ``i`` is chosen uniformly
  * ``i`` chooses among the ``n`` options {toggle tie to j : j != i} plus
    {do nothing} with multinomial-logit probabilities proportional to
    ``exp(f_i(x^(+-j)))``, where ``f`` is the linear objective in ``effects.py``
  * the chosen tie is toggled

``B`` chains run in lockstep. Chains whose Poisson count is exhausted keep
drawing random numbers but stop changing, so the stream consumed per ministep
is fixed and a run is fully determined by the seed.

Nothing here ever filters or resamples a chain. Empty and complete networks are
legitimate outcomes and are returned as such.
"""

from __future__ import annotations

import numpy as np

from .backend import DTYPE, batched_row_products, categorical_sample, softmax, toggle, zero_diagonal
from .effects import Model

OUT_DTYPE = np.int8


def random_network(B: int, n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi ``(B, n, n)`` digraphs with zero diagonal, dtype int8."""
    X = (rng.random((B, n, n)) < density).astype(OUT_DTYPE)
    return zero_diagonal(X)


def _broadcast_params(theta, rate, B: int, K: int):
    theta = np.asarray(theta, dtype=DTYPE)
    if theta.ndim == 1:
        if theta.shape[0] != K:
            raise ValueError(f"theta has {theta.shape[0]} entries, model has {K} effects")
        theta = np.broadcast_to(theta, (B, K))
    elif theta.shape != (B, K):
        raise ValueError(f"theta must be (K,) or (B, K) = ({B}, {K}); got {theta.shape}")
    rate = np.broadcast_to(np.asarray(rate, dtype=DTYPE), (B,))
    if np.any(rate < 0):
        raise ValueError("rate must be non-negative")
    return theta, rate


def simulate_period(
    X0: np.ndarray,
    theta,
    rate,
    model: Model,
    rng: np.random.Generator,
    *,
    return_n_steps: bool = False,
):
    """Simulate one period from ``X0``.

    X0     ``(B, n, n)`` 0/1 with zero diagonal
    theta  ``(K,)`` shared across chains, or ``(B, K)`` per chain
    rate   scalar or ``(B,)``; expected ministeps per actor
    rng    the one and only ``numpy.random.Generator`` used

    Returns ``X1`` as ``(B, n, n)`` int8 (and the per-chain ministep counts if
    ``return_n_steps``).
    """
    X0 = np.asarray(X0)
    if X0.ndim != 3 or X0.shape[1] != X0.shape[2]:
        raise ValueError(f"X0 must be (B, n, n); got {X0.shape}")
    B, n, _ = X0.shape
    theta, rate = _broadcast_params(theta, rate, B, model.K)

    X = X0.astype(DTYPE)
    needs = model.needs
    n_steps = rng.poisson(rate * n)
    for t in range(int(n_steps.max()) if B else 0):
        active = t < n_steps
        actor = rng.integers(0, n, size=B)
        u = rng.random(B)
        rows = batched_row_products(X, actor, needs)
        f = model.objective(rows, actor, theta)
        target = categorical_sample(softmax(f), u)
        toggle(X, actor, target, active)

    X1 = X.astype(OUT_DTYPE)
    return (X1, n_steps) if return_n_steps else X1


def simulate_panel(
    X0: np.ndarray,
    theta,
    rate,
    model: Model,
    rng: np.random.Generator,
    waves: int = 2,
) -> np.ndarray:
    """Simulate ``waves - 1`` consecutive periods. Returns ``(waves, B, n, n)``.

    ``theta`` may be ``(K,)``/``(B, K)`` (shared across periods) or carry a
    leading period axis of length ``waves - 1``. Likewise ``rate`` may be a
    scalar, ``(B,)``, or ``(waves - 1,)`` / ``(waves - 1, B)``.
    """
    if waves < 2:
        raise ValueError("a panel needs at least two waves")
    X0 = np.asarray(X0)
    B, n, _ = X0.shape
    M = waves - 1
    K = model.K

    theta = np.asarray(theta, dtype=DTYPE)
    per_period_theta = theta.ndim == 3 or (theta.ndim == 2 and theta.shape != (B, K))
    if per_period_theta and theta.shape[0] != M:
        raise ValueError(f"per-period theta must have leading length {M}; got {theta.shape}")

    rate = np.asarray(rate, dtype=DTYPE)
    per_period_rate = rate.ndim == 2 or (rate.ndim == 1 and rate.shape[0] == M and M != B)
    if per_period_rate and rate.shape[0] != M:
        raise ValueError(f"per-period rate must have leading length {M}; got {rate.shape}")

    out = np.empty((waves, B, n, n), dtype=OUT_DTYPE)
    out[0] = X0
    for m in range(M):
        th = theta[m] if per_period_theta else theta
        rt = rate[m] if per_period_rate else rate
        out[m + 1] = simulate_period(out[m], th, rt, model, rng)
    return out


def rate_statistic(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """Number of tie changes between waves per chain (the rate target statistic)."""
    return (np.asarray(X0) != np.asarray(X1)).sum(axis=(1, 2)).astype(DTYPE)


def statistics(X: np.ndarray, model: Model) -> np.ndarray:
    """Target statistics ``(B, K)`` of a batch of networks."""
    return model.statistics(np.asarray(X))
