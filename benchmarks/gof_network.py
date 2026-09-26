"""Goodness of fit on network statistics the estimator never targeted.

    python benchmarks/gof_network.py --posterior data/npe_m5c.pt --real s50 --waves 3
    python benchmarks/gof_network.py --posterior data/npe_m5c.pt \
        --waves-csv w1.csv w2.csv w3.csv w4.csv --v v.csv --g g.csv

The posterior predictive checks elsewhere in this repository (``ppc_coev.py``,
``ppc_s50.py``) look at behaviour statistics, or at scalar summaries of one period of
s50. This is the network counterpart in the style of ``sienaGOF`` (Lospinoso and Snijders
2019), and it is the check a reader will ask for: simulate each period from its
*observed* start under draws from the posterior, then compare the observed end network
with the simulated ones on the out- and in-degree distributions, the triad census and
the distribution of geodesic distances.

None of those is a statistic the flow conditions on, and the triad census and geodesics
are not moment conditions of any estimator here, so this is genuinely held out. A
vector-valued statistic is judged as a whole by the Mahalanobis distance of the
observation from the simulated cloud, with a Monte Carlo p-value and no distributional
assumption; the per-entry standardised deviations then say *where* a poor fit fails.

The posterior is the joint one from ``multiwave.py``, so every period is simulated under
draws that respect the constraint that the effects are shared across periods -- which is
the relevant null when the question is whether one parameter vector can reproduce the
whole panel.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.multiwave import (  # noqa: E402
    check_n,
    combine,
    csv_views,
    load_posterior,
    net_views,
)
from saomsim import simulate_period  # noqa: E402
from saomsim.gof import TRIAD_TYPES, auxiliary, mahalanobis_test  # noqa: E402
from saomsim.population import m2_model  # noqa: E402

LABELS = {
    "outdegree": [f"out {k}" for k in range(8)] + ["out 8+"],
    "indegree": [f"in {k}" for k in range(8)] + ["in 8+"],
    "triad census": TRIAD_TYPES,
    "geodesic": ["d=1", "d=2", "d=3", "d=4", "d=5", "d>5 or inf"],
}


def real_panel(real, waves):
    from benchmarks.m2 import glasgow_as_m2, s50_as_m2

    if real == "s50":
        x0, later, v, g, _, _ = s50_as_m2(waves=waves)
    else:
        x0, later, v, g, _, _ = glasgow_as_m2(waves=waves)
    Xs = [x0] + (list(later) if isinstance(later, (list, tuple)) else [later])
    return Xs, v, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_m5c.pt")
    ap.add_argument("--real", default="s50", choices=["s50", "glasgow"])
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--waves-csv", nargs="+")
    ap.add_argument("--v")
    ap.add_argument("--g")
    ap.add_argument("--B", type=int, default=1000, help="simulations per period")
    ap.add_argument("--proposal", type=int, default=200000)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rsiena", help="RSiena fit JSON: run the same check at its point")
    ap.add_argument(
        "--point-control", action="store_true",
        help="also check the posterior mean as a point, to separate the location of the "
             "estimate from the propagation of its uncertainty"
    )
    ap.add_argument("--out", help="write the per-statistic table to this .npz")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.waves_csv:
        from benchmarks.fit import load_waves

        Xs = load_waves(a.waves_csv)
        v = np.loadtxt(a.v, delimiter=",", ndmin=1)
        g = np.loadtxt(a.g, delimiter=",", ndmin=1)
        g = g - g.min()
        views, eff_names, n_rate, P, n = csv_views(a.waves_csv, a.v, a.g, None)
        label = Path(a.waves_csv[0]).stem
    else:
        Xs, v, g = real_panel(a.real, a.waves)
        views, eff_names, n_rate, P, n = net_views(a.real, a.waves)
        label = a.real

    post = load_posterior(a.posterior)
    print(f"network goodness of fit: {label}, {len(Xs)} waves -> {P} periods, "
          f"{a.B} simulations per period, {Path(a.posterior).name}")
    check_n(a.posterior, n)

    phi, info = combine(post, views, n_rate, draws=max(4000, a.B), proposal=a.proposal,
                        seed=a.seed, verbose=False)
    print(f"  joint posterior by {info['method']}, ESS {info['ess']:.0f} effective draws\n")

    covs = {"v": np.repeat(np.asarray(v, float)[None], a.B, 0),
            "g": np.repeat(np.asarray(g, float)[None], a.B, 0)}
    model = m2_model(covs)
    draws = phi[rng.choice(len(phi), a.B, replace=True)]

    if n_rate != 1:
        raise SystemExit("this check is for the network model, which has one rate per period")
    points = {"amortized posterior": (draws[:, P:], draws[:, :P])}
    if a.point_control:
        # The control that makes the comparison with RSiena interpretable. Simulating from
        # the posterior *mean* holds the estimate where it is but throws away parameter
        # uncertainty, exactly as simulating from a point estimate does. If this behaves
        # like RSiena's point rather than like the posterior, then what separates them is
        # the propagation of uncertainty and not the location of the estimate.
        mean = phi.mean(0)
        points["amortized mean, as a point"] = (
            np.repeat(mean[None, P:], a.B, 0),
            np.repeat(mean[None, :P], a.B, 0),
        )
    if a.rsiena:
        # The same specification fitted by RSiena. If both fail a statistic the same way,
        # the specification is what fails, not the inference -- which is the only way to
        # tell a model criticism from an estimator criticism.
        import json

        d = json.loads(Path(a.rsiena).read_text(encoding="utf-8"))
        est = d["estimate"]

        def base(k):
            """Strip an optional 'net:'/'alc:' prefix and any '(covariate)' suffix, so that
            'net:altX', 'altX(alc)' and our own 'altX(v)' all reduce to 'altX'. The R
            scripts here label effects both ways depending on whether the fit was a
            co-evolution one."""
            return k.split(":", 1)[-1].split("(", 1)[0]

        by_base = {}
        for k, val in est.items():
            by_base.setdefault(base(k), []).append((k, val))

        def one(name):
            hits = by_base.get(base(name), [])
            if len(hits) != 1:
                raise SystemExit(
                    f"{Path(a.rsiena).name}: {len(hits)} keys match {name!r} "
                    f"({[h[0] for h in hits]}); cannot pick one"
                )
            return hits[0][1]

        eff = np.array([one(nm) for nm in eff_names])
        rates = np.array([one(f"rate_{w + 1}") for w in range(P)])
        points["RSiena MoM point"] = (
            np.repeat(eff[None], a.B, 0),
            np.repeat(rates[None], a.B, 0),
        )

    rows = {}
    for lab, (beta, rate_cols) in points.items():
        print(f"### {lab}")
        for w in range(P):
            X0 = np.repeat(Xs[w][None], a.B, 0).astype(np.int8)
            Xsim = simulate_period(
                X0, beta, np.ascontiguousarray(rate_cols[:, w]), model, rng, backend=a.backend
            )
            sim = auxiliary(Xsim)
            obs = auxiliary(np.asarray(Xs[w + 1])[None].astype(np.int8))
            print(f"== period {w + 1}: wave {w + 1} -> wave {w + 2}")
            print(f"{'statistic':<16}{'entries':>9}{'Mahalanobis p':>15}   largest deviations")
            for key in ("outdegree", "indegree", "triad census", "geodesic"):
                p, dd, contrib = mahalanobis_test(sim[key], obs[key][0])
                worst = np.argsort(-np.abs(contrib))[:3]
                bits = ", ".join(
                    f"{LABELS[key][j]} {contrib[j]:+.1f}" for j in worst if abs(contrib[j]) > 0.5
                )
                flag = "  <-- rejected" if p < 0.05 else ""
                print(
                    f"{key:<16}{sim[key].shape[1]:>9}{p:>15.3f}   "
                    f"{bits or 'none > 0.5 sd'}{flag}"
                )
                tag = lab.replace(" ", "_").replace(",", "")
                rows[f"{tag}_p{w + 1}_{key}"] = np.array([p, dd])
                rows[f"{tag}_c{w + 1}_{key}"] = contrib
            print()

    print("p is the fraction of simulated networks at least as far from the simulated mean")
    print("as the observed one, by Mahalanobis distance on that statistic; deviations are")
    print("per-entry standardised differences (observed minus simulated mean, in sd).")
    if a.out:
        np.savez(a.out, **rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
