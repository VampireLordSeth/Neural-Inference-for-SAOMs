"""
Effect definitions for the stochastic actor-oriented model.

Each effect supplies two things:

1. ``delta(ctx)`` -- the *change statistic* for the focal actor: a (B, n) array
   whose [b, j] entry is  s_i(x toggled at (i,j)) - s_i(x)  for chain b, where
   i is that chain's focal actor. This is what drives the multinomial choice.

2. ``statistic(X, cov)`` -- the network-level statistic  sum_i s_i(x), a (B,)
   array. These are the moment targets used by method-of-moments estimation.

The invariant tying them together, and the thing the test suite checks
exhaustively against naive loop implementations, is:

    delta[b, j] == s_i(toggle(X[b], i, j)) - s_i(X[b])

Actor-level statistics implemented here (x is the adjacency matrix, v a
covariate, all diagonals zero):

    density      s_i = sum_j x_ij
    recip        s_i = sum_j x_ij x_ji
    transTrip    s_i = sum_{j,h} x_ij x_ih x_hj
    cycle3       s_i = sum_{j,h} x_ij x_jh x_hi
    sameX        s_i = sum_j x_ij 1[v_i == v_j]
    altX         s_i = sum_j x_ij v_j
    egoX         s_i = v_i sum_j x_ij

NOTE ON CONVENTION: statistics are defined here as sum_i s_i with the actor
statistics above. RSiena's Appendix B conventions must be confirmed term by
term before any claim of numerical agreement is made. See README.
"""

from __future__ import annotations

import numpy as np

from .backend import batched_row_products


class StepContext:
    """Per-micro-step precomputed quantities shared across effects.

    Attributes
    ----------
    X : (B, n, n) float array
        Current adjacency matrices, diagonal zero.
    actor : (B,) int array
        Focal actor index for each chain.
    xi : (B, n)
        Outgoing ties of the focal actor: xi[b, j] = X[b, i, j].
    xTi : (B, n)
        Incoming ties of the focal actor: xTi[b, j] = X[b, j, i].
    two_path : (B, n)
        sum_h x_ih x_hj  -- i to j via an intermediary.
    shared_out : (B, n)
        sum_h x_ih x_jh  -- outgoing ties i and j have in common.
    cyc : (B, n)
        sum_h x_jh x_hi  -- j back to i via an intermediary.
    sign : (B, n)
        1 - 2 x_ij: +1 if toggling creates a tie, -1 if it dissolves one.
    """

    __slots__ = ("X", "actor", "xi", "xTi", "two_path", "shared_out", "cyc",
                 "sign", "cov", "n", "B")

    def __init__(self, X: np.ndarray, actor: np.ndarray, cov=None):
        self.X = X
        self.actor = actor
        self.cov = cov
        self.B, self.n = X.shape[0], X.shape[1]

        xi, xTi, two_path, shared_out, cyc = batched_row_products(X, actor)
        self.xi = xi
        self.xTi = xTi
        self.two_path = two_path
        self.shared_out = shared_out
        self.cyc = cyc
        self.sign = 1.0 - 2.0 * xi


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------

class Effect:
    """Base class. Subclasses define ``name``, ``delta``, ``statistic``."""

    name = "effect"
    #: name of the covariate this effect needs, or None
    covariate = None

    def delta(self, ctx: StepContext) -> np.ndarray:
        raise NotImplementedError

    def statistic(self, X: np.ndarray, cov: dict) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class Density(Effect):
    """Outdegree / density.  s_i = sum_j x_ij"""

    name = "density"

    def delta(self, ctx):
        return ctx.sign

    def statistic(self, X, cov):
        return X.sum(axis=(1, 2))


class Reciprocity(Effect):
    """Reciprocity.  s_i = sum_j x_ij x_ji

    Toggling (i, j) changes this by +/- x_ji.
    """

    name = "recip"

    def delta(self, ctx):
        return ctx.sign * ctx.xTi

    def statistic(self, X, cov):
        return (X * np.swapaxes(X, 1, 2)).sum(axis=(1, 2))


class TransitiveTriplets(Effect):
    """Transitive triplets.  s_i = sum_{j,h} x_ij x_ih x_hj

    The derivative with respect to x_ij collects the two ways x_ij enters:
    as the closing tie of a two-path i->h->j, and as the middle tie of
    i->j->h with i->h present. Hence  sum_h x_ih x_hj + sum_h x_ih x_jh.
    """

    name = "transTrip"

    def delta(self, ctx):
        return ctx.sign * (ctx.two_path + ctx.shared_out)

    def statistic(self, X, cov):
        # sum_{i,j,h} x_ij x_ih x_hj  ==  sum_{i,j} x_ij (X @ X)_{ij}
        return (X * (X @ X)).sum(axis=(1, 2))


class ThreeCycles(Effect):
    """Three-cycles.  s_i = sum_{j,h} x_ij x_jh x_hi

    Derivative with respect to x_ij is sum_h x_jh x_hi.
    """

    name = "cycle3"

    def delta(self, ctx):
        return ctx.sign * ctx.cyc

    def statistic(self, X, cov):
        # sum_{i,j} x_ij (X @ X)_{ji}
        return (X * np.swapaxes(X @ X, 1, 2)).sum(axis=(1, 2))


class SameCovariate(Effect):
    """Categorical homophily.  s_i = sum_j x_ij 1[v_i == v_j]"""

    def __init__(self, covariate: str = "v"):
        self.covariate = covariate
        self.name = f"same_{covariate}"

    def delta(self, ctx):
        v = ctx.cov[self.covariate]                 # (n,)
        vi = v[ctx.actor][:, None]                  # (B, 1)
        same = (v[None, :] == vi).astype(ctx.X.dtype)
        return ctx.sign * same

    def statistic(self, X, cov):
        v = cov[self.covariate]
        same = (v[:, None] == v[None, :]).astype(X.dtype)
        return (X * same[None, :, :]).sum(axis=(1, 2))


class AltCovariate(Effect):
    """Covariate-related popularity.  s_i = sum_j x_ij v_j"""

    def __init__(self, covariate: str = "v"):
        self.covariate = covariate
        self.name = f"alt_{covariate}"

    def delta(self, ctx):
        v = ctx.cov[self.covariate].astype(ctx.X.dtype)
        return ctx.sign * v[None, :]

    def statistic(self, X, cov):
        v = cov[self.covariate].astype(X.dtype)
        return (X * v[None, None, :]).sum(axis=(1, 2))


class EgoCovariate(Effect):
    """Covariate-related activity.  s_i = v_i sum_j x_ij"""

    def __init__(self, covariate: str = "v"):
        self.covariate = covariate
        self.name = f"ego_{covariate}"

    def delta(self, ctx):
        v = ctx.cov[self.covariate].astype(ctx.X.dtype)
        vi = v[ctx.actor][:, None]                  # (B, 1)
        return ctx.sign * vi

    def statistic(self, X, cov):
        v = cov[self.covariate].astype(X.dtype)
        return (X * v[None, :, None]).sum(axis=(1, 2))


#: Effects that take no arguments, addressable by name.
SIMPLE_EFFECTS = {
    "density": Density,
    "recip": Reciprocity,
    "transTrip": TransitiveTriplets,
    "cycle3": ThreeCycles,
}

#: Effects parameterised by a covariate name.
COVARIATE_EFFECTS = {
    "sameX": SameCovariate,
    "altX": AltCovariate,
    "egoX": EgoCovariate,
}


def make_effects(spec) -> list:
    """Build an effect list from a compact specification.

    Parameters
    ----------
    spec : sequence
        Each item is either a string naming a simple effect ("density",
        "recip", "transTrip", "cycle3"), or a tuple ``(kind, covariate)``
        for a covariate effect, e.g. ``("sameX", "v")``.

    Examples
    --------
    >>> eff = make_effects(["density", "recip", "transTrip", ("sameX", "v")])
    >>> [e.name for e in eff]
    ['density', 'recip', 'transTrip', 'same_v']
    """
    out = []
    for item in spec:
        if isinstance(item, Effect):
            out.append(item)
        elif isinstance(item, str):
            if item not in SIMPLE_EFFECTS:
                raise KeyError(
                    f"unknown effect {item!r}; simple effects are "
                    f"{sorted(SIMPLE_EFFECTS)}, covariate effects are "
                    f"{sorted(COVARIATE_EFFECTS)} and need a (kind, name) tuple"
                )
            out.append(SIMPLE_EFFECTS[item]())
        else:
            kind, covname = item
            if kind not in COVARIATE_EFFECTS:
                raise KeyError(f"unknown covariate effect {kind!r}")
            out.append(COVARIATE_EFFECTS[kind](covname))
    return out


def statistics(X: np.ndarray, effects: list, cov: dict | None = None) -> np.ndarray:
    """Network-level statistics for a batch of networks.

    Returns
    -------
    (B, p) array, column k holding ``effects[k].statistic`` for each network.
    """
    cov = cov or {}
    X = np.atleast_3d(X) if X.ndim == 3 else X[None, :, :]
    return np.stack([e.statistic(X, cov) for e in effects], axis=1)
