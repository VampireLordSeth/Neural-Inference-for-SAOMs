"""Simulation-based calibration of the product-of-periods posterior.

    python benchmarks/sbc_multiwave.py --posterior data/npe_m5c.pt --N 400 --waves 3

This is the test that decides whether reading a W-wave panel as a product of two-wave
posteriors is correct (``saomsim/multiwave.py``). Panels are drawn from the same
population the two-wave estimator was trained on, but with ``--waves`` waves and one rate
per period; each is split into its periods, the estimator is applied to each, the product
is sampled, and the true theta is ranked within the joint draws. If the factorisation is
right and the flow is calibrated, the ranks are uniform.

It is a stronger test than it looks, because it also probes the one assumption the
factorisation quietly makes about *data*: period w is conditioned on x_{w-1}, which for
w > 1 is an **evolved** network, not a draw from the population's start distribution. The
estimator never saw an evolved start in training.

That shift turned out to matter, mildly. ``--per-period`` skips the product and
calibrates each period's two-wave posterior on its own, against (rate_w, beta), which
separates the shift from anything the product does. On 1,000 panels period 1 passes 7 of
8, while period 2 rejects rate (KS p 0.008) and density (0.014), both with mean rank
0.528 — the truth sits above the posterior median too often, so on an evolved start the
estimator reads rate and density low. The factorisation is not what costs; the training
population is. See ``docs/MULTIWAVE.md``.

Reported per parameter: the KS test against uniform, the mean rank (0.5 if calibrated),
and 90 %/95 % central credible-interval coverage.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.m2 import m2_prior  # noqa: E402
from benchmarks.multiwave import combine, load_posterior  # noqa: E402
from saomsim.multiwave import m2_period_views  # noqa: E402
from saomsim.population import generate_m2, m2_summary_names, rate_names  # noqa: E402


def per_period_sbc(post, views_all, theta, P, n_rate, model, a, kstest):
    """Calibrate each period's two-wave posterior separately.

    The product posterior can only be as good as its factors. Period 1 starts from a
    network drawn the way the training population draws them; period w > 1 starts from a
    network that has already evolved, which the estimator never saw. Ranking each period's
    own posterior against (rate_w, beta) separates that shift from anything the product
    itself does, and the two columns are directly comparable because they use the same
    estimator on the same panels.
    """
    import torch


    eff = list(model.labels)
    pnames = ["rate"] + eff
    print(f"\nper-period calibration, {a.N} panels, {P} periods (no product)\n", flush=True)
    out = {}
    for w in range(P):
        ranks, in90 = [], []
        for s in range(0, a.N, 200):
            chunk = views_all[s : s + 200, w]
            smp = np.stack(
                [
                    post.sample(
                        (a.draws,),
                        x=torch.as_tensor(chunk[i : i + 1], dtype=torch.float32),
                        show_progress_bars=False,
                    ).numpy()
                    for i in range(len(chunk))
                ]
            )
            for i in range(len(chunk)):
                # theta is (rate_1..rate_P, effects); this period's rate plus the effects
                true = np.concatenate([theta[s + i, w : w + 1], theta[s + i, P:]])
                ranks.append((smp[i] < true).mean(0))
                lo, hi = np.percentile(smp[i], [5, 95], axis=0)
                in90.append((true >= lo) & (true <= hi))
        ranks, in90 = np.asarray(ranks), np.asarray(in90)
        out[w] = (ranks, in90)
        print(f"== period {w + 1} ({'population start' if w == 0 else 'evolved start'})")
        print(f"{'parameter':<12}{'KS p':>8}{'mean rank':>11}{'90% cov':>10}")
        for j, nm in enumerate(pnames):
            p = kstest(ranks[:, j], "uniform").pvalue
            print(
                f"{nm:<12}{p:8.3f}{ranks[:, j].mean():11.3f}{in90[:, j].mean():10.3f}"
                + ("  <-- rejected" if p < 0.05 else "")
            )
        print()
    np.savez(
        a.out.replace(".npz", "_per_period.npz"),
        names=np.array(pnames),
        **{f"ranks_{w}": out[w][0] for w in out},
        **{f"in90_{w}": out[w][1] for w in out},
    )
    print(f"wrote {a.out.replace('.npz', '_per_period.npz')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_m5c.pt")
    ap.add_argument("--N", type=int, default=400, help="panels")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--draws", type=int, default=2000, help="joint draws per panel")
    ap.add_argument("--proposal", type=int, default=60000)
    ap.add_argument("--n-min", type=int, default=30)
    ap.add_argument("--n-max", type=int, default=100)
    ap.add_argument("--rate-max", type=float, default=12.0)
    ap.add_argument("--start", default="survey", choices=["m2", "sparse", "survey"])
    ap.add_argument("--box", default="sparse", choices=["default", "sparse"])
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default="data/sbc_multiwave.npz")
    ap.add_argument(
        "--per-period", action="store_true",
        help="skip the product and calibrate each period's two-wave posterior on its own, "
             "which isolates the one distribution shift the product relies on: period w > 1 "
             "conditions on an evolved network, never a population start"
    )
    a = ap.parse_args()

    from scipy.stats import kstest

    P = a.waves - 1
    prior = m2_prior(a.waves, rate=(1.0, a.rate_max), box=a.box)
    rng = np.random.default_rng(a.seed)
    print(
        f"SBC of the product posterior: {a.N} panels, {a.waves} waves ({P} periods), "
        f"n in [{a.n_min}, {a.n_max}], start={a.start}, box={a.box}",
        flush=True,
    )

    t0 = time.perf_counter()
    ts = generate_m2(
        prior,
        a.N,
        rng,
        chunk=a.chunk,
        backend="numpy",
        keep_networks=False,
        waves=a.waves,
        n_range=(a.n_min, a.n_max),
        start=a.start,
    )
    print(f"  simulated in {time.perf_counter() - t0:.0f}s", flush=True)

    from saomsim.population import m2_model, sample_covariates

    names = list(ts.summary_names)
    model = m2_model(sample_covariates(1, a.n_min, np.random.default_rng(0)))
    assert names == m2_summary_names(model, a.waves), "summary layout does not match"

    views_all = m2_period_views(ts.summary, names, model, transform=True)
    theta = np.asarray(ts.theta)
    post = load_posterior(a.posterior)
    n_rate = 1

    if a.per_period:
        return per_period_sbc(post, views_all, theta, P, n_rate, model, a, kstest)

    ranks, ess, inside90, inside95 = [], [], [], []
    t0 = time.perf_counter()
    for i in range(a.N):
        phi, info = combine(
            post,
            views_all[i],
            n_rate,
            draws=a.draws,
            proposal=a.proposal,
            seed=a.seed + i,
            verbose=False,
        )
        ranks.append((phi < theta[i]).mean(0))
        ess.append(info["ess_frac"])
        lo90, hi90 = np.percentile(phi, [5, 95], axis=0)
        lo95, hi95 = np.percentile(phi, [2.5, 97.5], axis=0)
        inside90.append((theta[i] >= lo90) & (theta[i] <= hi90))
        inside95.append((theta[i] >= lo95) & (theta[i] <= hi95))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(
                f"  {i + 1}/{a.N}  {el:.0f}s  eta {el / (i + 1) * (a.N - i - 1):.0f}s  "
                f"median ESS {np.median(ess) * 100:.0f}%",
                flush=True,
            )

    ranks = np.asarray(ranks)
    inside90, inside95 = np.asarray(inside90), np.asarray(inside95)
    pnames = rate_names(a.waves) + list(model.labels)
    se = np.sqrt(0.9 * 0.1 / a.N)

    print(f"\nmedian ESS {np.median(ess) * 100:.1f}%, min {np.min(ess) * 100:.1f}%")
    print(f"\n{'parameter':<12}{'KS p':>8}{'mean rank':>11}{'90% cov':>10}{'95% cov':>10}")
    for j, nm in enumerate(pnames):
        p = kstest(ranks[:, j], "uniform").pvalue
        flag = "  <-- rejected" if p < 0.05 else ""
        print(
            f"{nm:<12}{p:8.3f}{ranks[:, j].mean():11.3f}"
            f"{inside90[:, j].mean():10.3f}{inside95[:, j].mean():10.3f}{flag}"
        )
    print(f"\nbinomial se on 90% coverage at N={a.N}: {se:.3f}")

    np.savez(a.out, ranks=ranks, names=np.array(pnames), ess=np.array(ess),
             inside90=inside90, inside95=inside95)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
