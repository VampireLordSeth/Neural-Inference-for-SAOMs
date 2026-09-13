"""Reference implementations: slow, loop-based, obviously correct.

These exist to be compared against. They are written to read like the formulas
in the RSiena manual and nothing else. Do not optimize them. When a vectorized
kernel in ``effects.py`` disagrees with a function here, the kernel is wrong.

All functions take a single ``(n, n)`` network, not a batch.
"""

from __future__ import annotations

import numpy as np

from .effects import Effect, as_effect


def _cov(covariates, name):
    v = np.asarray(covariates[name], dtype=float)
    if v.ndim != 1:
        raise ValueError("reference functions take a single (n,) covariate vector")
    return v


def actor_statistic(x: np.ndarray, i: int, effect, covariates: dict | None = None) -> float:
    """``s_i(x)`` for one effect, by explicit summation."""
    e: Effect = as_effect(effect)
    n = x.shape[0]
    s = 0.0
    if e.kind == "density":
        for j in range(n):
            s += x[i, j]
    elif e.kind == "recip":
        for j in range(n):
            s += x[i, j] * x[j, i]
    elif e.kind == "transTrip":
        for j in range(n):
            for h in range(n):
                s += x[i, j] * x[i, h] * x[h, j]
    elif e.kind == "cycle3":
        for j in range(n):
            for h in range(n):
                s += x[i, j] * x[j, h] * x[h, i]
    else:
        v = _cov(covariates, e.covariate)
        if e.kind == "sameX":
            for j in range(n):
                s += x[i, j] * (1.0 if v[i] == v[j] else 0.0)
        elif e.kind == "altX":
            for j in range(n):
                s += x[i, j] * v[j]
        elif e.kind == "egoX":
            for j in range(n):
                s += x[i, j] * v[i]
    return float(s)


def change_statistic(x: np.ndarray, i: int, j: int, effect, covariates=None) -> float:
    """``s_i(x with x_ij toggled) - s_i(x)``; zero for the no-change option ``j == i``."""
    if i == j:
        return 0.0
    x2 = np.array(x, copy=True)
    x2[i, j] = 1 - x2[i, j]
    return actor_statistic(x2, i, effect, covariates) - actor_statistic(x, i, effect, covariates)


def change_statistics_row(x: np.ndarray, i: int, effects, covariates=None) -> np.ndarray:
    """All ``delta_ijk`` for actor ``i``: shape ``(n, K)``."""
    n = x.shape[0]
    effects = [as_effect(e) for e in effects]
    out = np.zeros((n, len(effects)))
    for j in range(n):
        for k, e in enumerate(effects):
            out[j, k] = change_statistic(x, i, j, e, covariates)
    return out


def statistics(x: np.ndarray, effects, covariates=None) -> np.ndarray:
    """Target statistics, RSiena convention: shape ``(K,)``.

    ``sum_i s_ik(x)`` for every effect except ``cycle3``, whose RSiena target
    counts each 3-cycle once rather than once per member actor (so ``/ 3``).
    Verified against RSiena 1.6.6 in ``benchmarks/``.
    """
    n = x.shape[0]
    effects = [as_effect(e) for e in effects]
    out = np.zeros(len(effects))
    for k, e in enumerate(effects):
        for i in range(n):
            out[k] += actor_statistic(x, i, e, covariates)
        if e.kind == "cycle3":
            out[k] /= 3.0
    return out


def objective(x, i, theta, effects, covariates=None) -> np.ndarray:
    """``f_ij = sum_k theta_k delta_ijk`` for every ``j``: shape ``(n,)``."""
    return change_statistics_row(x, i, effects, covariates) @ np.asarray(theta, dtype=float)


def choice_probabilities(x, i, theta, effects, covariates=None) -> np.ndarray:
    """Multinomial-logit probabilities over actor ``i``'s ``n`` options, by brute force."""
    f = objective(x, i, theta, effects, covariates)
    e = np.exp(f - f.max())
    return e / e.sum()


# ------------------------------------------------------------ behaviour (M3b)


def behaviour_actor_statistic(x, z, i, effect, zbar, sim_mean, z_range) -> float:
    """``s_i(x, z)`` for one behaviour effect, by explicit loops (single network, (n,) z)."""
    n = x.shape[0]
    zt = [z[j] - zbar for j in range(n)]
    if effect == "linear":
        return float(zt[i])
    if effect == "quad":
        return float(zt[i] ** 2)
    out = [j for j in range(n) if x[i, j] == 1]
    if not out:
        return 0.0
    if effect == "avAlt":
        return float(zt[i] * sum(zt[j] for j in out) / len(out))
    if effect == "avSim":
        return float(sum((1 - abs(z[i] - z[j]) / z_range) - sim_mean for j in out) / len(out))
    raise KeyError(effect)


def behaviour_change(x, z, i, d, effect, zbar, sim_mean, z_range) -> float:
    """``s_i(x, z with z_i += d) - s_i(x, z)``."""
    z2 = np.array(z, dtype=float, copy=True)
    z2[i] += d
    return behaviour_actor_statistic(
        x, z2, i, effect, zbar, sim_mean, z_range
    ) - behaviour_actor_statistic(x, z, i, effect, zbar, sim_mean, z_range)


def selection_creation(z, i, j, effect, zbar, sim_mean, z_range) -> float:
    """Creation contribution of a selection effect for the tie i -> j."""
    if effect == "egoZ":
        return float(z[i] - zbar)
    if effect == "altZ":
        return float(z[j] - zbar)
    if effect == "simZ":
        return float((1 - abs(z[i] - z[j]) / z_range) - sim_mean)
    raise KeyError(effect)
