"""Batched simulation of the stochastic actor-oriented model.

One period of the basic SAOM (constant rate, evaluation function only):

  * at each ministep one actor ``i`` is chosen uniformly
  * ``i`` chooses among the ``n`` options {toggle tie to j : j != i} plus
    {do nothing} with multinomial-logit probabilities proportional to
    ``exp(f_i(x^(+-j)))``, where ``f`` is the linear objective in ``effects.py``
    (equivalently: the option maximising ``f + Gumbel noise``)
  * the chosen tie is toggled

Three stopping rules, exactly one of which is given:

  rate      unconditional. Actors get opportunities at rate ``rate`` each, so
            over a unit-length period the number of ministeps is exactly
            Poisson(n * rate). This is the process RSiena simulates with
            ``cond = FALSE`` and the one validated against it in benchmarks/.
  n_steps   fixed number of ministeps per chain. Useful for tests and for
            stationarity checks; not a process RSiena has.
  distance  conditional. Each chain runs until the Hamming distance between the
            current network and ``X0`` reaches ``distance`` (RSiena's
            ``cond = TRUE`` default: condition on the observed number of
            changed tie variables, which removes the rate from the moment
            problem). ``max_steps`` guards against chains that cannot get there.

``B`` chains run in lockstep. Chains that have stopped keep drawing random
numbers but no longer change, so the stream consumed per ministep is fixed and
a run is fully determined by the seed. The Poisson step count is always drawn
from the numpy Generator, whichever backend runs the ministeps.

Nothing here ever filters or resamples a chain. Empty and complete networks are
legitimate outcomes and are returned as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import DTYPE, OUT_DTYPE, get_backend, zero_diagonal
from .effects import Model


@dataclass
class SimulationInfo:
    """Per-chain diagnostics from ``simulate_period``."""

    n_steps: np.ndarray  # ministeps run (for ``distance`` runs: steps until stopped)
    n_changes: np.ndarray  # ministeps that toggled a tie (not the no-change option)
    reached: np.ndarray  # ``distance`` runs only: chain hit the target distance


def random_network(B: int, n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Erdos-Renyi ``(B, n, n)`` digraphs with zero diagonal, dtype int8."""
    X = (rng.random((B, n, n)) < density).astype(OUT_DTYPE)
    return zero_diagonal(X)


def _broadcast_theta(theta, B: int, K: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=DTYPE)
    if theta.ndim == 1:
        if theta.shape[0] != K:
            raise ValueError(f"theta has {theta.shape[0]} entries, model has {K} effects")
        theta = np.broadcast_to(theta, (B, K))
    elif theta.shape != (B, K):
        raise ValueError(f"theta must be (K,) or (B, K) = ({B}, {K}); got {theta.shape}")
    return theta


def _step_schedule(rate, n_steps, distance, B: int, n: int, rng, max_steps):
    """Return (steps, target_distance) as numpy arrays; exactly one rule must be set."""
    given = sum(x is not None for x in (rate, n_steps, distance))
    if given != 1:
        raise ValueError("give exactly one of rate, n_steps, distance")
    if rate is not None:
        rate = np.broadcast_to(np.asarray(rate, dtype=DTYPE), (B,))
        if np.any(rate < 0):
            raise ValueError("rate must be non-negative")
        return rng.poisson(rate * n).astype(np.int64), None
    if n_steps is not None:
        steps = np.broadcast_to(np.asarray(n_steps, dtype=np.int64), (B,))
        if np.any(steps < 0):
            raise ValueError("n_steps must be non-negative")
        return steps.copy(), None
    target = np.broadcast_to(np.asarray(distance, dtype=np.int64), (B,))
    if np.any(target < 0) or np.any(target > n * (n - 1)):
        raise ValueError("distance must lie in [0, n(n-1)]")
    if max_steps is None:
        # a chain that has not covered its distance in this many ministeps is stuck
        # (e.g. the network emptied and the parameters will not re-create ties)
        max_steps = int(10 * n + 5 * target.max())
    return np.full(B, max_steps, dtype=np.int64), target.copy()


def simulate_period(
    X0: np.ndarray,
    theta,
    rate=None,
    model: Model = None,
    rng: np.random.Generator = None,
    *,
    n_steps=None,
    distance=None,
    max_steps: int | None = None,
    backend="numpy",
    return_n_steps: bool = False,
    return_info: bool = False,
):
    """Simulate one period from ``X0``.

    X0        ``(B, n, n)`` 0/1 with zero diagonal
    theta     ``(K,)`` shared across chains, or ``(B, K)`` per chain
    rate      scalar or ``(B,)``: unconditional, Poisson(n * rate) ministeps
    n_steps   scalar or ``(B,)``: run exactly this many ministeps
    distance  scalar or ``(B,)``: run until Hamming distance from X0 reaches it
    model     the ``Model`` (effects + covariates)
    rng       the one and only ``numpy.random.Generator`` used
    backend   ``"numpy"`` (default), ``"torch"``, ``"torch:cpu"``, or an instance

    Returns ``X1`` as ``(B, n, n)`` int8, plus ``n_steps`` if ``return_n_steps``
    or a ``SimulationInfo`` if ``return_info``.
    """
    if model is None or rng is None:
        raise TypeError("simulate_period needs model= and rng=")
    bk = get_backend(backend)
    X0 = np.asarray(X0)
    if X0.ndim != 3 or X0.shape[1] != X0.shape[2]:
        raise ValueError(f"X0 must be (B, n, n); got {X0.shape}")
    B, n, _ = X0.shape
    theta_np = _broadcast_theta(theta, B, model.K)
    steps_np, target_np = _step_schedule(rate, n_steps, distance, B, n, rng, max_steps)
    conditional = target_np is not None

    X = bk.network(X0)
    theta_d = bk.array(theta_np)
    steps = bk.int_array(steps_np)
    state = bk.rng_state(rng)
    needs = model.needs
    n_changes = bk.int_array(np.zeros(B, dtype=np.int64))
    if conditional:
        X0_d = bk.network(X0)
        target = bk.int_array(target_np)
        dist = bk.int_array(np.zeros(B, dtype=np.int64))
        stopped_at = bk.int_array(np.full(B, -1, dtype=np.int64))

    # Early exit once no chain is active. For torch this is a device sync, so
    # check only every few steps; the draws are deterministic given the seed
    # either way, so stopping early does not affect reproducibility.
    check_every = 1 if bk.name == "numpy" else 8
    for t in range(int(steps_np.max()) if B else 0):
        active = t < steps
        if conditional:
            active = active & (dist < target)
        if t % check_every == 0 and not bool(active.any()):
            break
        actor = bk.integers(n, B, state)
        u = bk.random(B, state)
        rows = bk.row_products(X, actor, needs)
        f = model.objective(rows, actor, theta_d, bk)
        chosen = bk.categorical_sample(bk.softmax(f), u)
        changed = active & (chosen != actor)
        bk.toggle(X, actor, chosen, active)
        n_changes += changed
        if conditional:
            dist = bk.hamming(X, X0_d)
            just_done = (dist >= target) & (stopped_at < 0)
            stopped_at[just_done] = t + 1

    X1 = bk.finalize(X)
    if conditional:
        stopped_np = bk.to_numpy(stopped_at)
        reached = stopped_np >= 0
        run_steps = np.where(reached, stopped_np, steps_np)
        # a zero target is reached before any step
        run_steps[target_np == 0] = 0
        reached |= target_np == 0
    else:
        run_steps, reached = steps_np, np.ones(B, dtype=bool)
    if return_info:
        return X1, SimulationInfo(run_steps, bk.to_numpy(n_changes).astype(np.int64), reached)
    if return_n_steps:
        return X1, run_steps
    return X1


def simulate_panel(
    X0: np.ndarray,
    theta,
    rate,
    model: Model,
    rng: np.random.Generator,
    waves: int = 2,
    *,
    backend="numpy",
) -> np.ndarray:
    """Simulate ``waves - 1`` consecutive unconditional periods. Returns ``(waves, B, n, n)``.

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
        out[m + 1] = simulate_period(out[m], th, rt, model, rng, backend=backend)
    return out


def rate_statistic(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """Number of tie changes between waves per chain (the rate target statistic)."""
    return (np.asarray(X0) != np.asarray(X1)).sum(axis=(1, 2)).astype(DTYPE)


def statistics(X: np.ndarray, model: Model) -> np.ndarray:
    """Target statistics ``(B, K)`` of a batch of networks (numpy)."""
    return model.statistics(np.asarray(X))
