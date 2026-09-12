"""
Naive, obviously-correct reference implementations.

These are deliberately written as explicit loops straight from the algebraic
definitions. They are slow and are never used in simulation; their only job is
to be the thing the vectorised kernels in :mod:`saomsim.effects` are checked
against, entry by entry, in the test suite.

If a vectorised change statistic and its reference here disagree, the
vectorised one is wrong. An incorrect simulator produces a confidently wrong
posterior with no diagnostic signature, which is why this module exists before
anything downstream is built.
"""

from __future__ import annotations

import numpy as np


def actor_statistic(X: np.ndarray, i: int, effect_name: str,
                    v: np.ndarray | None = None) -> float:
    """Actor-level statistic s_i(x), by direct summation.

    Parameters
    ----------
    X : (n, n) array
    i : int
    effect_name : {'density', 'recip', 'transTrip', 'cycle3',
                   'sameX', 'altX', 'egoX'}
    v : (n,) array, required for covariate effects
    """
    n = X.shape[0]

    if effect_name == "density":
        return float(sum(X[i, j] for j in range(n)))

    if effect_name == "recip":
        return float(sum(X[i, j] * X[j, i] for j in range(n)))

    if effect_name == "transTrip":
        total = 0.0
        for j in range(n):
            for h in range(n):
                total += X[i, j] * X[i, h] * X[h, j]
        return float(total)

    if effect_name == "cycle3":
        total = 0.0
        for j in range(n):
            for h in range(n):
                total += X[i, j] * X[j, h] * X[h, i]
        return float(total)

    if effect_name == "sameX":
        return float(sum(X[i, j] * (v[i] == v[j]) for j in range(n)))

    if effect_name == "altX":
        return float(sum(X[i, j] * v[j] for j in range(n)))

    if effect_name == "egoX":
        return float(v[i] * sum(X[i, j] for j in range(n)))

    raise KeyError(effect_name)


def network_statistic(X: np.ndarray, effect_name: str,
                      v: np.ndarray | None = None) -> float:
    """Network statistic sum_i s_i(x), by direct summation."""
    return float(sum(actor_statistic(X, i, effect_name, v)
                     for i in range(X.shape[0])))


def reference_delta(X: np.ndarray, i: int, effect_name: str,
                    v: np.ndarray | None = None) -> np.ndarray:
    """Change statistics for actor i over all options j, by brute force.

    Returns an (n,) array whose j-th entry is
    ``s_i(toggle(X, i, j)) - s_i(X)``, with the j == i entry set to zero
    (the no-change option).
    """
    n = X.shape[0]
    base = actor_statistic(X, i, effect_name, v)
    out = np.zeros(n)
    for j in range(n):
        if j == i:
            continue
        Y = X.copy()
        Y[i, j] = 1.0 - Y[i, j]
        out[j] = actor_statistic(Y, i, effect_name, v) - base
    return out


def reference_choice_probabilities(X: np.ndarray, i: int, theta, effect_names,
                                   v: np.ndarray | None = None) -> np.ndarray:
    """Multinomial logit choice probabilities for actor i, by brute force."""
    n = X.shape[0]
    utility = np.zeros(n)
    for th, name in zip(theta, effect_names):
        utility += th * reference_delta(X, i, name, v)
    utility[i] = 0.0
    e = np.exp(utility - utility.max())
    return e / e.sum()
