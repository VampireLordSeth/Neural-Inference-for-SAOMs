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
    "npe_m8.pt": (20, 150),   # the M6 population at three waves (docs/PRODUCT_VS_NATIVE.md)
    "npe_m7.pt": (20, 150),   # M6 plus evolved starts
    "npe_m6.pt": (20, 150),
    "npe_coev_m6.pt": (20, 150),
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
    # mcse is the Monte Carlo error of the printed mean, sd / sqrt(effective draws). It is
    # there so the nominal draw count cannot be mistaken for the independent one: a column
    # of 10,000 correlated draws with ESS 900 carries a third fewer digits than it looks.
    ess = max(float(info.get("ess", np.nan)), 1.0)
    print(
        f"\n{'parameter':<14}{'mean':>9}{'mcse':>8}{'sd':>8}{'2.5%':>9}{'97.5%':>9}"
        "   per-period means"
    )
    for j, nm in enumerate(phi_names):
        col = phi[:, j]
        q = np.percentile(col, [2.5, 97.5])
        if j < P * n_rate:
            note = f"period {j // n_rate + 1} only: {per[j // n_rate][:, j % n_rate].mean():+.3f}"
        else:
            k = j - P * n_rate + n_rate
            note = "  ".join(f"{per[w][:, k].mean():+.3f}" for w in range(P))
        sd = col.std(ddof=1)
        print(
            f"{nm:<14}{col.mean():9.3f}{sd / np.sqrt(ess):8.3f}{sd:8.3f}"
            f"{q[0]:9.3f}{q[1]:9.3f}   {note}"
        )

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
        f"  {info['method']}, {len(phi)} draws carrying {info['ess']:.0f} effective"
        + (f", accept {info['accept']:.2f}" if "accept" in info else "")
        + (f", max Rhat {np.nanmax(info['rhat']):.3f}" if "rhat" in info else "")
    )
    if info["ess"] < a.min_ess:
        print(
            f"  WARNING: {info['ess']:.0f} effective draws. Intervals from these samples "
            f"carry a Monte Carlo error about {np.sqrt(len(phi) / info['ess']):.1f}x larger "
            "than the draw count suggests; treat the third decimal as noise."
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
            draws=len(phi),
            method=info["method"],
            spread=spread,
        )
        print(
            f"\nwrote {a.out}: {len(phi)} draws, {info['ess']:.0f} effective. Use the "
            "effective count, not the array length, for any Monte Carlo error."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------- goodness of fit


def main_gof(argv=None):
    """``saom-gof``: does the fitted model reproduce statistics it never targeted?

        saom-gof --waves w1.csv w2.csv w3.csv --v v.csv --g g.csv \
                 --posterior npe_m5c.pt

    Agreement with another estimator says the two recover the same parameters. It says
    nothing about whether the *model* reproduces the data, and on one of the panels in
    this project it does not: the canonical effect set cannot match the out-degree
    distribution of a nomination-limited survey, which the check below detects and no
    amount of agreement with RSiena would have revealed (``docs/GOF_RESULTS.md``).

    Each period is simulated from its observed start under draws from the posterior, and
    the observed end network is compared with the simulated ones on the out- and
    in-degree distributions, the triad census and the distribution of geodesic distances
    -- none of which the estimator conditions on. Each vector is judged as a whole by the
    Mahalanobis distance of the observation from the simulated cloud.

    Simulating under posterior *draws* rather than a point estimate matters: conditioning
    on any single theta discards the uncertainty the data leave, giving a predictive
    distribution that is too narrow and a check that over-rejects.
    """
    ap = argparse.ArgumentParser(
        description=main_gof.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--waves", nargs="+", required=True, help="adjacency CSVs in time order")
    ap.add_argument("--posterior", required=True, help="trained two-wave estimator (.pt)")
    ap.add_argument("--v", required=True, help="numeric actor covariate CSV")
    ap.add_argument("--g", required=True, help="categorical actor covariate CSV")
    ap.add_argument("--B", type=int, default=1000, help="simulations per period")
    ap.add_argument("--proposal", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--cap", type=int, default=0,
        help="truncate each simulated actor to this many out-ties before the statistics, "
             "as a nomination-limited questionnaire would"
    )
    a = ap.parse_args(argv)

    from saomsim import simulate_period
    from saomsim.gof import TRIAD_TYPES, auxiliary, mahalanobis_test
    from saomsim.population import cap_outdegree, m2_model
    from saomsim.product import combine, load_posterior, require_torch

    try:
        require_torch()
    except ModuleNotFoundError as e:  # pragma: no cover - depends on the install
        raise SystemExit(str(e)) from e

    labels = {
        "outdegree": [f"out {k}" for k in range(8)] + ["out 8+"],
        "indegree": [f"in {k}" for k in range(8)] + ["in 8+"],
        "triad census": TRIAD_TYPES,
        "geodesic": ["d=1", "d=2", "d=3", "d=4", "d=5", "d>5 or inf"],
    }
    rng = np.random.default_rng(a.seed)
    Xs = load_waves(a.waves)
    views, eff_names, n_rate, P, n = csv_views(a.waves, a.v, a.g, None)
    if n_rate != 1:
        raise SystemExit("this check covers the network model only")

    post = load_posterior(a.posterior)
    print(
        f"{Path(a.waves[0]).stem}, {len(a.waves)} waves -> {P} periods, n = {n}, "
        f"{a.B} simulations per period"
    )
    check_n(a.posterior, n)
    phi, info = combine(post, views, 1, draws=max(4000, a.B), proposal=a.proposal, seed=a.seed)
    print(f"  posterior by {info['method']}, {a.draws} draws, {info['ess']:.0f} effective\n")

    v = np.loadtxt(a.v, delimiter=",", ndmin=1)
    g = np.loadtxt(a.g, delimiter=",", ndmin=1)
    covs = {
        "v": np.repeat(np.asarray(v, float)[None], a.B, 0),
        "g": np.repeat(np.asarray(g - g.min(), float)[None], a.B, 0),
    }
    model = m2_model(covs)
    draws = phi[rng.choice(len(phi), a.B, replace=True)]

    rejected = 0
    for w in range(P):
        X0 = np.repeat(Xs[w][None], a.B, 0).astype(np.int8)
        Xsim = simulate_period(
            X0, draws[:, P:], np.ascontiguousarray(draws[:, w]), model, rng
        )
        if a.cap:
            cap_outdegree(Xsim, np.full(len(Xsim), a.cap), rng)
        sim = auxiliary(Xsim)
        obs = auxiliary(np.asarray(Xs[w + 1])[None].astype(np.int8))
        print(f"== period {w + 1}: wave {w + 1} -> wave {w + 2}")
        print(f"{'statistic':<16}{'Mahalanobis p':>15}   largest deviations")
        for key in ("outdegree", "indegree", "triad census", "geodesic"):
            p_, _, contrib = mahalanobis_test(sim[key], obs[key][0])
            worst = np.argsort(-np.abs(contrib))[:3]
            bits = ", ".join(
                f"{labels[key][j]} {contrib[j]:+.1f}" for j in worst if abs(contrib[j]) > 0.5
            )
            rejected += p_ < 0.05
            print(
                f"{key:<16}{p_:>15.3f}   {bits or 'none > 0.5 sd'}"
                + ("  <-- rejected" if p_ < 0.05 else "")
            )
        print()
    print(
        "p is the fraction of simulated networks at least as far from the simulated mean "
        "as\nthe observed one; deviations are per-entry standardised differences in sd.\n"
        f"{rejected} of {4 * P} tests rejected at 5 %."
    )
    if rejected:
        print(
            "A rejection is a statement about the model, not about the estimator: RSiena "
            "simulates\nthe same process and would say the same. See docs/GOF_RESULTS.md "
            "for a worked case\nwhere the cause was a nomination cap the SAOM does not "
            "enforce, and where truncating\nthe simulations (--cap) did not repair it."
        )
    return 0
