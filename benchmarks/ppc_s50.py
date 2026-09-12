"""Posterior predictive check for the s50 estimator on statistics NOT in the embedding.

    python benchmarks/ppc_s50.py --posterior data/npe_s50_posterior.npz [--B 2000] [--backend torch]

Draws B parameter vectors from the saved posterior, simulates one period each
from s501, and compares the observed s502 on graph-space statistics the flow
never saw: dyad census, degree extremes and dispersion, clustering, reachability,
geodesics. Reports the posterior predictive quantile of each observed value;
values near 0 or 1 flag structure the model (or the summaries) misses.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.s50 import load_s50  # noqa: E402
from saomsim import simulate_period  # noqa: E402


def graph_stats(X: np.ndarray) -> dict:
    """Held-out statistics for a batch ``(B, n, n)`` of 0/1 networks."""
    Xf = X.astype(np.float64)
    B, n, _ = Xf.shape
    XT = Xf.transpose(0, 2, 1)
    mutual = (Xf * XT).sum(axis=(1, 2)) / 2
    asym = (Xf * (1 - XT)).sum(axis=(1, 2))
    null = n * (n - 1) / 2 - mutual - asym
    out = Xf.sum(axis=2)
    ind = Xf.sum(axis=1)
    # transitivity: transitive triplets / two-paths (i->h->j, i != j)
    XX = Xf @ Xf
    two_paths = XX.sum(axis=(1, 2)) - np.einsum("bii->b", XX)
    trans = (Xf * XX).sum(axis=(1, 2)) / np.maximum(two_paths, 1)
    # reachability and geodesics via boolean matrix powers (n = 50: cheap)
    reach = np.eye(n, dtype=bool)[None].repeat(B, 0)
    dist = np.full((B, n, n), np.inf)
    dist[:, np.arange(n), np.arange(n)] = 0
    frontier = reach.copy()
    A = X.astype(bool)
    for d in range(1, n):
        nxt = np.einsum("bij,bjk->bik", frontier.astype(np.int32), A.astype(np.int32)) > 0
        new = nxt & ~reach
        if not new.any():
            break
        dist[new] = d
        reach |= new
        frontier = new
    offdiag = ~np.eye(n, dtype=bool)[None]
    reachable = reach & offdiag
    frac_reach = reachable.sum(axis=(1, 2)) / (n * (n - 1))
    mean_geo = np.array(
        [dist[b][reachable[b]].mean() if reachable[b].any() else np.nan for b in range(B)]
    )
    # degree assortativity (out-degree of sender vs in-degree of receiver over ties)
    assort = np.empty(B)
    for b in range(B):
        s, r = np.nonzero(X[b])
        assort[b] = np.corrcoef(out[b, s], ind[b, r])[0, 1] if len(s) > 2 else np.nan
    return {
        "mutual_dyads": mutual,
        "asymmetric_dyads": asym,
        "null_dyads": null,
        "max_outdegree": out.max(axis=1),
        "max_indegree": ind.max(axis=1),
        "indegree_gini": np.array([gini(v) for v in ind]),
        "out_isolates": (out == 0).sum(axis=1),
        "transitivity": trans,
        "frac_reachable": frac_reach,
        "mean_geodesic": mean_geo,
        "degree_assortativity": assort,
    }


def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    n = v.size
    if v.sum() == 0:
        return 0.0
    return (2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_s50_posterior.npz")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x0, x1, model = load_s50()
    z = np.load(args.posterior, allow_pickle=False)
    samples, names = z["samples"], [str(n) for n in z["names"]]
    assert names == ["rate"] + model.labels
    rng = np.random.default_rng(args.seed)
    theta = samples[rng.choice(len(samples), args.B, replace=False)]
    X0 = np.repeat(x0[None], args.B, axis=0)
    X1 = simulate_period(X0, theta[:, 1:], theta[:, 0], model, rng, backend=args.backend)

    sim = graph_stats(X1)
    obs = graph_stats(x1[None])
    print(f"posterior predictive check: {args.B} draws from {args.posterior}, start s501\n")
    print(f"{'statistic':<22}{'obs':>9}{'pp mean':>9}{'pp sd':>8}{'q05':>8}{'q95':>8}{'F(obs)':>8}")
    flags = []
    for k, v in sim.items():
        v = v[np.isfinite(v)]
        o = float(obs[k][0])
        F = (v < o).mean() + 0.5 * (v == o).mean()
        q = np.quantile(v, [0.05, 0.95])
        print(f"{k:<22}{o:>9.3f}{v.mean():>9.3f}{v.std():>8.3f}{q[0]:>8.3f}{q[1]:>8.3f}{F:>8.3f}")
        if F < 0.025 or F > 0.975:
            flags.append(k)
    print(f"\nobserved outside the central 95% of the posterior predictive: {flags or 'none'}")


if __name__ == "__main__":
    main()
