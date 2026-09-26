"""Read a panel of any length with a two-wave estimator.

A SAOM's likelihood factorises over periods, each conditioned on the *observed* start
of that period:

    p(theta | x_0 .. x_W)  ∝  p(theta) · prod_w L_w(beta, rate_w ; x_{w-1} -> x_w)

Our priors are flat on a box, so a two-wave posterior for period w is

    q_w(rate_w, beta)  ∝  L_w · 1_box,

and the product of the per-period posteriors, restricted to the box, is proportional to
the joint posterior over (rate_1 .. rate_{W-1}, beta). One two-wave estimator therefore
reads a two-, three-, four- or W-wave panel: apply it once per period and sample the
product (``benchmarks/multiwave.py``). Nothing is retrained, padded or flagged, and the
per-period rates fall out on their own rather than fixing the wave count of theta.

What this module does is the first half: turn a W-wave summary vector into the W-1
two-wave vectors the estimator expects. For the network-only summaries that is a pure
re-slice -- the x_w block already holds the same statistics of wave w that an x0 block
would, plus a change count -- so no network is touched and nothing is recomputed:

    period w:   x0_<stat> <- x{w-1}_<stat>,   x1_<stat> <- x{w}_<stat>

The co-evolution vector needs one statistic that re-slicing cannot supply; see
``coev_period_views``.
"""

import numpy as np

from saomsim.population import m2_summary_names, transform_m2

__all__ = ["period_view_index", "m2_period_views", "coev_period_views", "n_periods"]


def n_periods(names: list) -> int:
    """Number of periods a summary-name list covers: waves - 1, i.e. the highest wave index."""
    return max(
        int(nm[1:].split("_", 1)[0])
        for nm in names
        if nm.startswith("x") and nm[1:2].isdigit()
    )


def period_view_index(names: list, two_wave_names: list, period: int) -> list[int]:
    """Columns of a W-wave vector that form the two-wave vector for ``period`` (1-based).

    Raises ``KeyError`` naming the statistic if some column of the two-wave vector is not
    a column of the W-wave one -- which is how the co-evolution case announces itself.
    """
    if not 1 <= period <= n_periods(names):
        raise ValueError(f"period {period} outside 1..{n_periods(names)}")
    if period > 1 and "sim_mean" in names:
        # Co-evolution. Every name would resolve, but one block would resolve to the wrong
        # statistic and say nothing about it: a start block holds selection_statistics(X_w,
        # z_w), while the x_w block of a W-wave vector holds selection_statistics(X_w,
        # z_{w-1}). Re-slicing cannot tell them apart, so refuse rather than return it.
        raise KeyError(
            f"period {period} of a co-evolution panel cannot be re-sliced: the start block "
            "needs selection_statistics(X, z) at the same wave, and the W-wave vector only "
            "carries the cross-lagged pair. Use coev_period_views(), which recomputes from "
            "the panel."
        )
    pos = {nm: i for i, nm in enumerate(names)}
    idx = []
    for nm in two_wave_names:
        src = nm
        if nm.startswith("x0_"):
            src = f"x{period - 1}_{nm[3:]}"
        elif nm.startswith("x1_"):
            src = f"x{period}_{nm[3:]}"
        elif nm.startswith("z0_"):
            src = f"z{period - 1}_{nm[3:]}"
        elif nm.startswith("z1_"):
            src = f"z{period}_{nm[3:]}"
        if src not in pos:
            raise KeyError(
                f"period {period} of this panel has no column {src!r} for the two-wave "
                f"summary {nm!r}; it cannot be recovered by re-slicing"
            )
        idx.append(pos[src])
    return idx


def m2_period_views(S: np.ndarray, names: list, model, *, transform: bool = False) -> np.ndarray:
    """(B, d_W) raw network summaries -> (B, W-1, d_2) two-wave views, one per period.

    ``transform=True`` applies ``transform_m2`` to each view with the two-wave names, which
    is the form the estimator conditions on. The transform is per column and depends only
    on that column's statistic, so transforming before or after the re-slice is the same.
    """
    two = m2_summary_names(model, 2)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    views = np.stack(
        [S[:, period_view_index(names, two, w)] for w in range(1, n_periods(names) + 1)], axis=1
    )
    if transform:
        B, P, d = views.shape
        views = transform_m2(views.reshape(B * P, d), two, model).reshape(B, P, d)
    return views


def coev_period_views(Xs, zs, spec, model, bmodel) -> np.ndarray:
    """(B, W-1, d_2) two-wave co-evolution views, recomputed from the panel itself.

    Re-slicing is not enough here. A start block carries ``selection_statistics(X_w, z_w)``
    -- network and behaviour at the *same* wave -- whereas the x_w block of a W-wave vector
    carries ``selection_statistics(X_w, z_{w-1})``, RSiena's cross-lagged convention. The
    missing statistic is a deterministic function of the observed panel, not a simulation,
    so we recompute each period's vector from the waves directly; for a 4-wave classroom
    that is three calls on data already in memory.
    """
    from saomsim.population_coev import coev_summaries

    Xs, zs = list(Xs), list(zs)
    if len(Xs) != len(zs):
        raise ValueError(f"{len(Xs)} network waves but {len(zs)} behaviour waves")
    return np.stack(
        [
            coev_summaries(Xs[w : w + 2], zs[w : w + 2], spec, model, bmodel)
            for w in range(len(Xs) - 1)
        ],
        axis=1,
    )
