"""
Batched simulation of the stochastic actor-oriented model.

The model. A directed network on ``n`` actors evolves in continuous time.
Each actor receives change opportunities at rate ``lam``. On receiving one,
actor ``i`` may toggle one outgoing tie (i, j) or do nothing, choosing the
option that maximises

    f_i(x') = theta . delta_i(x -> x')  +  Gumbel noise,

which gives multinomial logit choice probabilities over the n options
(n - 1 tie toggles plus the no-change option, whose change statistic is zero
by construction).

Two simulation regimes.

*Unconditional.* With a constant rate ``lam`` per actor, the total event rate
is ``n * lam``, so over a unit-length period the number of micro-steps is
exactly Poisson(n * lam). We draw it and then run that many steps. This is
exact, not an approximation of the continuous-time chain.

*Conditional.* RSiena conditions by default on the observed number of
differing tie entries between waves, which removes the rate parameter from the
moment-matching problem. Supplying ``n_steps`` reproduces that: the chain runs
a fixed number of micro-steps and the rate parameter plays no role.

Reproducibility. All randomness flows from a single ``numpy.random.Generator``.
Exactly two draws are consumed per micro-step per batch (one to pick the actor,
one to pick the option), so a run is reproducible from its seed and step count
alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import categorical_sample, softmax, toggle
from .effects import StepContext, statistics


@dataclass
class SimulationResult:
    """Output of :func:`simulate_period`."""

    X: np.ndarray                 #: (B, n, n) networks at end of period
    n_steps: np.ndarray           #: (B,) micro-steps actually taken
    n_changes: np.ndarray         #: (B,) steps that changed a tie (not no-change)

    def statistics(self, effects, cov=None):
        return statistics(self.X, effects, cov)


def _check_inputs(X0, theta, effects):
    if X0.ndim != 3 or X0.shape[1] != X0.shape[2]:
        raise ValueError(f"X0 must be (B, n, n); got {X0.shape}")
    if len(theta) != len(effects):
        raise ValueError(
            f"theta has {len(theta)} entries but {len(effects)} effects were given"
        )
    if np.any(np.diagonal(X0, axis1=1, axis2=2) != 0):
        raise ValueError("X0 must have a zero diagonal (no self-ties)")
    if not np.all(np.isin(X0, (0.0, 1.0))):
        raise ValueError("X0 must be binary")


def simulate_period(
    X0: np.ndarray,
    theta,
    effects,
    cov: dict | None = None,
    *,
    n_steps=None,
    lam: float | None = None,
    rng: np.random.Generator | None = None,
    record_every: int | None = None,
) -> SimulationResult:
    """Simulate one period of network evolution for a batch of chains.

    Parameters
    ----------
    X0 : (B, n, n) array
        Starting networks. Copied, not modified.
    theta : (p,) array
        Objective-function parameters, aligned with ``effects``.
    effects : list of Effect
    cov : dict, optional
        Covariate name -> (n,) array.
    n_steps : int or (B,) int array, optional
        Conditional simulation: run exactly this many micro-steps.
    lam : float, optional
        Unconditional simulation: per-actor rate over a unit-length period.
        Number of micro-steps is drawn as Poisson(n * lam) per chain.
        Exactly one of ``n_steps`` and ``lam`` must be given.
    rng : numpy.random.Generator, optional
    record_every : int, optional
        If set, also return a trajectory (used for diagnostics, not estimation).

    Returns
    -------
    SimulationResult
    """
    rng = rng or np.random.default_rng()
    cov = cov or {}
    theta = np.asarray(theta, dtype=np.float64)
    X = np.array(X0, dtype=np.float64, copy=True)
    _check_inputs(X, theta, effects)

    B, n = X.shape[0], X.shape[1]

    if (n_steps is None) == (lam is None):
        raise ValueError("supply exactly one of n_steps (conditional) or lam (unconditional)")

    if n_steps is not None:
        steps = np.broadcast_to(np.asarray(n_steps, dtype=np.int64), (B,)).copy()
    else:
        if lam <= 0:
            raise ValueError("lam must be positive")
        steps = rng.poisson(n * lam, size=B).astype(np.int64)

    max_steps = int(steps.max()) if B else 0
    n_changes = np.zeros(B, dtype=np.int64)
    rows = np.arange(B)

    for t in range(max_steps):
        active = steps > t
        if not active.any():
            break
        idx = rows[active]
        Xa = X[idx]

        # constant rate => focal actor uniform on 1..n
        actor = rng.integers(0, n, size=idx.shape[0])

        ctx = StepContext(Xa, actor, cov)
        utility = np.zeros((idx.shape[0], n), dtype=np.float64)
        for k, eff in enumerate(effects):
            if theta[k] != 0.0:
                utility += theta[k] * eff.delta(ctx)

        # the no-change option is "toggle (i, i)", whose change statistic is 0
        utility[np.arange(idx.shape[0]), actor] = 0.0

        choice = categorical_sample(softmax(utility, axis=1), rng)

        changed = choice != actor
        if changed.any():
            ch = np.nonzero(changed)[0]
            toggle(X, idx[ch], actor[ch], choice[ch])
            n_changes[idx[ch]] += 1

    return SimulationResult(X=X, n_steps=steps, n_changes=n_changes)


def simulate_panel(
    X0: np.ndarray,
    theta,
    effects,
    cov: dict | None = None,
    *,
    n_waves: int = 2,
    lam: float | None = None,
    n_steps=None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a multi-wave panel.

    Returns
    -------
    (B, n_waves, n, n) array, wave 0 being ``X0``.
    """
    rng = rng or np.random.default_rng()
    B, n = X0.shape[0], X0.shape[1]
    out = np.empty((B, n_waves, n, n), dtype=np.float64)
    out[:, 0] = X0
    cur = X0
    for w in range(1, n_waves):
        res = simulate_period(cur, theta, effects, cov,
                              lam=lam, n_steps=n_steps, rng=rng)
        out[:, w] = res.X
        cur = res.X
    return out


def observed_distance(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Number of differing tie entries between two waves (Hamming distance).

    This is the quantity RSiena conditions on, and the moment target for the
    rate parameter when estimating unconditionally.
    """
    X1 = np.atleast_3d(X1) if X1.ndim == 3 else X1[None]
    X2 = np.atleast_3d(X2) if X2.ndim == 3 else X2[None]
    return np.abs(X1 - X2).sum(axis=(1, 2)).astype(np.int64)


def random_network(n: int, density: float, rng: np.random.Generator,
                   B: int = 1) -> np.ndarray:
    """Bernoulli random digraph(s) with zero diagonal."""
    X = (rng.random((B, n, n)) < density).astype(np.float64)
    idx = np.arange(n)
    X[:, idx, idx] = 0.0
    return X
