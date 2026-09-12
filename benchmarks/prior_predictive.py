"""Prior predictive check for the s50 prior (docs/PRIORS.md).

Draws N parameters from the prior, simulates one period from s501 for each,
and reports (a) the quantiles of every summary statistic under the prior
predictive, (b) where the observed s502 statistics fall in it, and (c) how
much of the prior predictive is degenerate. Nothing is filtered.

    python benchmarks/prior_predictive.py [--N 20000] [--backend torch] [--save path.npz]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.s50 import load_s50, s50_prior  # noqa: E402
from saomsim.prior import generate_training_set, prior_predictive_report, summaries  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    x0, x1, model = load_s50()
    prior = s50_prior(model)
    print("prior:\n" + prior.table() + "\n")

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    ts = generate_training_set(prior, x0, model, args.N, rng, backend=args.backend)
    dt = time.perf_counter() - t0
    rate = args.N / dt
    print(
        f"{args.N} panels from s501 in {dt:.1f}s ({rate:,.0f} panels/s, backend={args.backend})\n"
    )

    observed = summaries(x0[None], x1[None], model)[0]
    print(prior_predictive_report(ts, observed))

    # which parameters produce degenerate outcomes?
    dens = ts.summary[:, ts.summary_names.index("tie_fraction")]
    for label, mask in [("empty", dens == 0), ("denser than 0.5", dens > 0.5)]:
        if mask.any():
            print(f"\nmean theta given {label} (n={mask.sum()}):")
            for name, m_in, m_all in zip(
                ts.theta_names, ts.theta[mask].mean(0), ts.theta.mean(0), strict=True
            ):
                print(f"  {name:<12} {m_in:>7.2f}   (prior mean {m_all:>6.2f})")

    if args.save:
        ts.save(args.save)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
