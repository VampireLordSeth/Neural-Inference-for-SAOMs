"""``saom-fit``: read a panel of any number of waves with a trained two-wave estimator.

    saom-fit --waves w1.csv w2.csv w3.csv --v attribute.csv --g group.csv \
             --posterior npe_m5c.pt
    saom-fit --waves w1.csv w2.csv w3.csv w4.csv --behaviour z.csv \
             --posterior npe_coev_m5c.pt --out posterior.npz

Each wave is an n x n adjacency matrix of 0s and 1s as a headerless CSV, in time order.
For the network model, ``--v`` is one number per actor and ``--g`` one category per actor.
For the co-evolution model, ``--behaviour`` is an n x waves matrix of integer scores on a
1-5 scale, and no covariates are used.

Any number of waves from two upwards works, because the per-period posteriors multiply to
the joint one (``saomsim/multiwave.py``); one rate comes back per period and the effects
are shared. Nothing is retrained for a longer panel.

Two diagnostics are printed with every fit and are worth reading before the estimates.
The effective sample size says whether the product was sampled well. The **period spread**
asks whether the periods agree about the effects they are assumed to share: the model
asserts one set for the whole panel, and a large spread is a statement about the data, not
a numerical complaint.

The estimator's training range in ``n`` is checked and an extrapolation is announced. It
matters: an amortized estimator answers any query, and outside the population it was
trained on it degrades quietly, the posterior shrinking toward that population rather than
toward your data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Estimator file name -> the n range its training population actually covered. The
# screening references band n in steps of 20 and so cannot distinguish n = 25 from n = 35;
# the range is recorded here instead. See docs/PRIORS_M*.md.
TRAINED_N = {
    "npe_m5.pt": (30, 100),
    "npe_m5b.pt": (30, 100),
    "npe_m5c.pt": (30, 100),
    "npe_m5c_mom.pt": (30, 100),
    "npe_coev_m5.pt": (30, 100),
    "npe_coev_m5b.pt": (30, 100),
    "npe_coev_m5c.pt": (30, 100),
    "npe_m4b.pt": (20, 200),
    "npe_coev_m4b.pt": (20, 200),
    "npe_coev_m4_10m.pt": (20, 200),
}


def check_n(posterior_path, n):
    """Warn if the panel's n falls outside the estimator's training range."""
    rng = TRAINED_N.get(Path(posterior_path).name)
    if rng is None:
        print(f"  n = {n}; training range of {Path(posterior_path).name} not recorded")
        return None
    lo, hi = rng
    if lo <= n <= hi:
        return True
    print(
        f"  ** n = {n} is outside this estimator's training range [{lo}, {hi}]. The "
        f"posterior below is an extrapolation and should not be read as calibrated. **"
    )
    return False


def load_waves(paths):
    """Read adjacency CSVs in time order, checking they are square, 0/1 and the same size."""
    Xs = [np.loadtxt(p, delimiter=",", dtype=np.int8, ndmin=2) for p in paths]
    n = Xs[0].shape[0]
    for p, X in zip(paths, Xs, strict=True):
        if X.shape != (n, n):
            raise SystemExit(f"{p}: expected an {n} x {n} adjacency matrix, got {X.shape}")
        if not set(np.unique(X)).issubset({0, 1}):
            raise SystemExit(f"{p}: entries must be 0 or 1")
        np.fill_diagonal(X, 0)
    return Xs


def csv_views(paths, v_path=None, g_path=None, beh_path=None):
    """Any panel from CSVs -> (views, effect names, n_rate, periods, n).

    Co-evolution when ``beh_path`` is given, network-only otherwise.
    """
    from saomsim.multiwave import m2_period_views
    from saomsim.population import m2_summary_names, real_data_summary

    Xs = load_waves(paths)
    W = len(Xs)
    if W < 2:
        raise SystemExit("a panel needs at least two waves")

    if beh_path is None:
        if not (v_path and g_path):
            raise SystemExit("the network model needs --v and --g")
        v = np.loadtxt(v_path, delimiter=",", ndmin=1)
        g = np.loadtxt(g_path, delimiter=",", ndmin=1)
        if len(v) != len(Xs[0]) or len(g) != len(Xs[0]):
            raise SystemExit(
                f"covariates have {len(v)} and {len(g)} rows but the network has {len(Xs[0])}"
            )
        S, model = real_data_summary(Xs[0], Xs[1:], v, g - g.min())
        names = m2_summary_names(model, W)
        views = m2_period_views(S, names, model, transform=True)[0]
        return views, list(model.labels), 1, W - 1, Xs[0].shape[0]

    from saomsim.behaviour import BehaviourSpec, spec_from_data
    from saomsim.multiwave import coev_period_views
    from saomsim.population_coev import (
        BEH_EFFECTS,
        NET_EFFECTS,
        SEL_EFFECTS,
        beh_model,
        coev_summary_names,
        net_model,
        transform_coev,
    )

    Z = np.loadtxt(beh_path, delimiter=",", ndmin=2)
    if Z.shape[0] != len(Xs[0]):
        raise SystemExit(f"behaviour has {Z.shape[0]} rows but the network has {len(Xs[0])}")
    if Z.shape[1] != W:
        raise SystemExit(f"behaviour has {Z.shape[1]} waves but {W} networks were given")
    sp = spec_from_data(Z, 1, 5)
    spec = BehaviourSpec(1, 5, np.array([sp.zbar]), np.array([sp.sim_mean]))
    model, bmodel = net_model(), beh_model()
    views = coev_period_views(
        [X[None] for X in Xs], [Z[:, w][None] for w in range(W)], spec, model, bmodel
    )[0]
    views = transform_coev(views, coev_summary_names(2), model)
    return views, NET_EFFECTS + SEL_EFFECTS + BEH_EFFECTS, 2, W - 1, Xs[0].shape[0]


def phi_names_for(waves, n_rate, eff_names):
    """Names for phi = (rates of period 1, ..., rates of period P, shared effects).

    Note the rate order: phi groups rates *by period*, so co-evolution reads
    (rate_net_1, rate_beh_1, rate_net_2, ...) rather than grouping them by kind.
    """
    P = waves - 1
    if n_rate == 1:
        rates = [f"rate_{w + 1}" for w in range(P)]
    else:
        rates = [f"{k}_{w + 1}" for w in range(P) for k in ("rate_net", "rate_beh")]
    return rates + list(eff_names)


def report(phi, info, phi_names, eff_names, n_rate, P):
    """Print the joint posterior, the per-period means and the period spread."""
    from saomsim.product import period_spread

    per = info["per_period"]
    if len(phi_names) != phi.shape[1]:
        raise SystemExit(
            f"{len(phi_names)} names for {phi.shape[1]} sampled parameters -- the wave count "
            "used for naming does not match the panel"
        )
    print(f"\n{'parameter':<14}{'mean':>9}{'sd':>8}{'2.5%':>9}{'97.5%':>9}   per-period means")
    for j, nm in enumerate(phi_names):
        col = phi[:, j]
        q = np.percentile(col, [2.5, 97.5])
        if j < P * n_rate:
            note = f"period {j // n_rate + 1} only: {per[j // n_rate][:, j % n_rate].mean():+.3f}"
        else:
            k = j - P * n_rate + n_rate
            note = "  ".join(f"{per[w][:, k].mean():+.3f}" for w in range(P))
        print(f"{nm:<14}{col.mean():9.3f}{col.std(ddof=1):8.3f}{q[0]:9.3f}{q[1]:9.3f}   {note}")

    spread = period_spread(per, n_rate)
    print("\nperiod spread, max |mu_a - mu_b| / sqrt(s_a^2 + s_b^2) over period pairs:")
    for nm, s in zip(eff_names, spread, strict=True):
        print(f"  {nm:<14}{s:6.2f}" + ("   <-- periods disagree" if s > 2 else ""))
    return spread


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--waves", nargs="+", required=True, help="adjacency CSVs in time order")
    ap.add_argument("--posterior", required=True, help="trained two-wave estimator (.pt)")
    ap.add_argument("--v", help="numeric actor covariate CSV (network model)")
    ap.add_argument("--g", help="categorical actor covariate CSV (network model)")
    ap.add_argument("--behaviour", help="n x waves behaviour CSV (co-evolution model)")
    ap.add_argument("--draws", type=int, default=10000, help="posterior draws to return")
    ap.add_argument("--proposal", type=int, default=200000, help="importance proposals")
    ap.add_argument("--min-ess", type=float, default=500, help="fall back to MCMC below this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the draws to this .npz")
    a = ap.parse_args(argv)

    from saomsim.product import combine, load_posterior, require_torch

    try:
        require_torch()
    except ModuleNotFoundError as e:  # pragma: no cover - depends on the install
        raise SystemExit(str(e)) from e

    views, eff_names, n_rate, P, n = csv_views(a.waves, a.v, a.g, a.behaviour)
    post = load_posterior(a.posterior)
    n_eff = len(post.prior.base_dist.low) - n_rate
    if len(eff_names) != n_eff:
        raise SystemExit(
            f"{Path(a.posterior).name} expects {n_rate} rates and {n_eff} effects per period, "
            f"but this panel supplies {len(eff_names)} effect names. Did you mean to pass "
            f"{'--behaviour' if a.behaviour is None else '--v/--g'} instead?"
        )

    print(
        f"{Path(a.waves[0]).stem}, {len(a.waves)} waves -> {P} periods, "
        f"n = {n}, estimator {Path(a.posterior).name}"
    )
    check_n(a.posterior, n)
    phi, info = combine(
        post, views, n_rate, draws=a.draws, proposal=a.proposal, min_ess=a.min_ess, seed=a.seed
    )
    print(
        f"  {info['method']}, ESS {info['ess']:.0f} effective draws"
        + (f", accept {info['accept']:.2f}" if "accept" in info else "")
        + (f", max Rhat {np.nanmax(info['rhat']):.3f}" if "rhat" in info else "")
    )

    phi_names = phi_names_for(len(a.waves), n_rate, eff_names)
    spread = report(phi, info, phi_names, eff_names, n_rate, P)

    if a.out:
        np.savez(
            a.out,
            samples=phi,
            names=np.array(phi_names),
            per_period=info["per_period"],
            ess=info["ess"],
            spread=spread,
        )
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
