"""Auxiliary network statistics for goodness of fit, in the style of ``sienaGOF``.

Lospinoso and Snijders (2019) judge a fitted SAOM by simulating from it and comparing
the observed network with the simulated ones on statistics the estimator did *not*
target -- the out- and in-degree distributions, the triad census and the distribution of
geodesic distances. A vector-valued statistic is compared as a whole, by the Mahalanobis
distance of the observation from the simulated cloud, which accounts for the strong
correlations between its entries.

Everything here is batched over a leading simulation axis. The triad census is the
awkward one: rather than derive the 64-to-16 mapping by hand, ``triad_lookup`` builds it
by asking ``networkx`` to classify each of the 64 labelled three-node digraphs once, so
the class labels agree with the standard convention by construction rather than by my
reading of it. The lookup is cached, and the counting after that is pure numpy.
"""

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

TRIAD_TYPES = [
    "003", "012", "102", "021D", "021U", "021C", "111D", "111U",
    "030T", "030C", "201", "120D", "120U", "120C", "210", "300",
]

_LOOKUP = None


def triad_lookup():
    """(4, 4, 4) int8 array: dyad codes of (i,j), (i,k), (j,k) -> index into TRIAD_TYPES.

    A dyad code is 0 for a null dyad, 1 for i->j only, 2 for j->i only, 3 for mutual.
    Built once from ``networkx.triadic_census`` so the labels are not mine to get wrong.
    """
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    import networkx as nx

    pos = {t: i for i, t in enumerate(TRIAD_TYPES)}
    table = np.full((4, 4, 4), -1, dtype=np.int8)
    edges = {0: [], 1: [(0, 1)], 2: [(1, 0)], 3: [(0, 1), (1, 0)]}
    for a in range(4):
        for b in range(4):
            for c in range(4):
                g = nx.DiGraph()
                g.add_nodes_from([0, 1, 2])
                g.add_edges_from(edges[a])
                g.add_edges_from([(u + (0 if u == 0 else 1), v + (0 if v == 0 else 1))
                                  for u, v in edges[b]])  # pair (0,2)
                g.add_edges_from([(u + 1, v + 1) for u, v in edges[c]])  # pair (1,2)
                census = nx.triadic_census(g)
                hot = [t for t, n in census.items() if n == 1]
                if len(hot) != 1:
                    raise RuntimeError(f"ambiguous triad for codes {(a, b, c)}: {census}")
                table[a, b, c] = pos[hot[0]]
    _LOOKUP = table
    return table


def dyad_codes(X: np.ndarray) -> np.ndarray:
    """(B, n, n) 0/1 -> (B, n, n) codes, read as the code of the (row, col) dyad."""
    A = np.asarray(X, dtype=np.int8)
    return (A + 2 * A.transpose(0, 2, 1)).astype(np.int8)


@lru_cache(maxsize=8)
def _triples(n: int) -> tuple:
    """Index arrays for every i < j < k. Cached: the same n recurs across periods,
    simulations and chunks, and for n = 129 there are 357,760 of them."""
    idx = np.fromiter(
        (v for t in combinations(range(n), 3) for v in t), dtype=np.int32, count=comb(n, 3) * 3
    ).reshape(-1, 3)
    return idx[:, 0].copy(), idx[:, 1].copy(), idx[:, 2].copy()


def triad_census(X: np.ndarray, chunk: int = 32) -> np.ndarray:
    """(B, n, n) 0/1 -> (B, 16) counts over unordered triples, in TRIAD_TYPES order.

    Chunked over the batch because the gathered code arrays are one int8 per triple per
    network, which at n = 129 is 0.36 MB per network and would otherwise be held for the
    whole batch at once.
    """
    A = np.asarray(X)
    B, n, _ = A.shape
    if n < 3:
        return np.zeros((B, 16), dtype=np.int64)
    table = triad_lookup()
    ii, jj, kk = _triples(n)
    out = np.zeros((B, 16), dtype=np.int64)
    for s in range(0, B, chunk):
        C = dyad_codes(A[s : s + chunk])
        t = table[C[:, ii, jj], C[:, ii, kk], C[:, jj, kk]]
        for b in range(t.shape[0]):
            out[s + b] = np.bincount(t[b], minlength=16)
    return out


def degree_distribution(X: np.ndarray, axis: int, cap: int = 8) -> np.ndarray:
    """(B, cap + 1) counts of actors by degree, the last cell lumping ``>= cap``."""
    d = np.asarray(X, dtype=np.int64).sum(axis=axis)
    d = np.minimum(d, cap)
    return np.stack([np.bincount(row, minlength=cap + 1) for row in d])


def geodesic_distribution(X: np.ndarray, cap: int = 5) -> np.ndarray:
    """(B, cap + 1) counts of ordered pairs at distance 1..cap and unreachable.

    Breadth-first by boolean matrix powers; ``cap + 1`` collects pairs at distance
    greater than ``cap`` together with pairs that are not connected at all, which is how
    ``sienaGOF`` reports it (an infinite distance is just a large one).
    """
    A = np.asarray(X).astype(bool)
    B, n, _ = A.shape
    eye = np.eye(n, dtype=bool)
    reach = np.repeat(eye[None], B, 0)
    counts = np.zeros((B, cap + 1), dtype=np.int64)
    frontier = reach.copy()
    for d in range(1, cap + 1):
        nxt = np.einsum("bij,bjk->bik", frontier.astype(np.int8), A.astype(np.int8)) > 0
        new = nxt & ~reach
        counts[:, d - 1] = (new & ~eye[None]).sum(axis=(1, 2))
        reach |= new
        frontier = new
        if not new.any():
            break
    counts[:, cap] = n * (n - 1) - counts[:, :cap].sum(axis=1)
    return counts


def auxiliary(X: np.ndarray, deg_cap: int = 8, geo_cap: int = 5) -> dict:
    """The four auxiliary statistics ``sienaGOF`` is usually run with."""
    return {
        "outdegree": degree_distribution(X, axis=2, cap=deg_cap),
        "indegree": degree_distribution(X, axis=1, cap=deg_cap),
        "triad census": triad_census(X),
        "geodesic": geodesic_distribution(X, cap=geo_cap),
    }


def mahalanobis_test(sim: np.ndarray, obs: np.ndarray) -> tuple:
    """Monte Carlo test of a vector-valued statistic. Returns (p, d_obs, contributions).

    ``sim`` is (B, k) simulated values, ``obs`` is (k,). The covariance of the simulated
    cloud is inverted by pseudo-inverse, as ``sienaGOF`` does, because the entries of a
    census or a degree distribution sum to a constant and the covariance is therefore
    singular by construction. The p-value is the fraction of simulated points at least as
    far from the centre as the observation, so it needs no distributional assumption.

    ``contributions`` are per-entry standardised deviations, for saying *where* a
    rejected fit fails rather than only that it does.
    """
    S = np.asarray(sim, dtype=float)
    o = np.asarray(obs, dtype=float).ravel()
    mu = S.mean(axis=0)
    Vi = np.linalg.pinv(np.cov(S, rowvar=False))
    dif = S - mu
    d_sim = np.einsum("ij,jk,ik->i", dif, Vi, dif)
    do = o - mu
    d_obs = float(do @ Vi @ do)
    p = float(((d_sim >= d_obs).sum() + 1) / (len(d_sim) + 1))
    sd = S.std(axis=0, ddof=1)
    contrib = np.where(sd > 0, do / np.maximum(sd, 1e-12), 0.0)
    return p, d_obs, contrib
