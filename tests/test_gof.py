"""Tests for the auxiliary goodness-of-fit statistics (saomsim.gof)."""

import numpy as np
import pytest

from saomsim.gof import (
    TRIAD_TYPES,
    auxiliary,
    degree_distribution,
    dyad_codes,
    geodesic_distribution,
    mahalanobis_test,
    triad_census,
    triad_lookup,
)

nx = pytest.importorskip("networkx")


def _random_digraphs(B, n, p, seed):
    rng = np.random.default_rng(seed)
    X = (rng.random((B, n, n)) < p).astype(np.int8)
    for b in range(B):
        np.fill_diagonal(X[b], 0)
    return X


def _nx_census(X):
    g = nx.from_numpy_array(X, create_using=nx.DiGraph)
    c = nx.triadic_census(g)
    return np.array([c[t] for t in TRIAD_TYPES])


@pytest.mark.parametrize("n,p,seed", [(7, 0.4, 0), (11, 0.25, 1), (16, 0.3, 2), (9, 0.6, 3)])
def test_triad_census_matches_networkx(n, p, seed):
    """The whole point of building the lookup from networkx is that this holds exactly."""
    X = _random_digraphs(4, n, p, seed)
    got = triad_census(X)
    for b in range(4):
        np.testing.assert_array_equal(got[b], _nx_census(X[b]))


def test_triad_census_on_edge_cases():
    n = 6
    empty = np.zeros((1, n, n), dtype=np.int8)
    full = (1 - np.eye(n, dtype=np.int8))[None]
    assert triad_census(empty)[0, TRIAD_TYPES.index("003")] == 20  # C(6,3)
    assert triad_census(full)[0, TRIAD_TYPES.index("300")] == 20
    np.testing.assert_array_equal(triad_census(empty)[0], _nx_census(empty[0]))
    np.testing.assert_array_equal(triad_census(full)[0], _nx_census(full[0]))


def test_triad_census_sums_to_the_number_of_triples():
    X = _random_digraphs(3, 13, 0.3, 7)
    assert (triad_census(X).sum(axis=1) == 13 * 12 * 11 // 6).all()


def test_triad_census_is_chunk_invariant():
    X = _random_digraphs(9, 10, 0.3, 8)
    np.testing.assert_array_equal(triad_census(X, chunk=2), triad_census(X, chunk=64))


def test_triad_lookup_covers_every_dyad_code_combination():
    table = triad_lookup()
    assert table.shape == (4, 4, 4)
    assert (table >= 0).all() and (table < 16).all()


def test_dyad_codes():
    X = np.array([[[0, 1, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.int8)
    C = dyad_codes(X)
    assert C[0, 0, 1] == 3  # mutual
    assert C[0, 1, 2] == 2  # 2 -> 1 only, read from (1,2)
    assert C[0, 0, 2] == 0  # null


def test_degree_distribution_counts_actors_and_caps_the_tail():
    # actor 0 sends to everyone (out-degree 5), the rest send nothing
    n = 6
    X = np.zeros((1, n, n), dtype=np.int8)
    X[0, 0, 1:] = 1
    out = degree_distribution(X, axis=2, cap=8)
    assert out.sum() == n
    assert out[0, 5] == 1 and out[0, 0] == 5
    capped = degree_distribution(X, axis=2, cap=3)
    assert capped[0, 3] == 1  # 5 lumped into the 3+ cell
    ind = degree_distribution(X, axis=1, cap=8)
    assert ind[0, 1] == 5 and ind[0, 0] == 1


def test_geodesic_distribution_on_a_directed_path():
    """0 -> 1 -> 2 -> 3: distances 1 (x3), 2 (x2), 3 (x1); the other 6 pairs unreachable."""
    n = 4
    X = np.zeros((1, n, n), dtype=np.int8)
    for i in range(n - 1):
        X[0, i, i + 1] = 1
    g = geodesic_distribution(X, cap=5)[0]
    assert g.sum() == n * (n - 1)
    assert list(g[:3]) == [3, 2, 1]
    assert g[5] == 6  # unreachable pairs land in the last cell


def test_geodesic_distribution_on_a_complete_graph():
    n = 5
    X = (1 - np.eye(n, dtype=np.int8))[None]
    g = geodesic_distribution(X, cap=5)[0]
    assert g[0] == n * (n - 1) and g[1:].sum() == 0


def test_mahalanobis_p_is_uniform_when_the_observation_comes_from_the_cloud():
    """If the observation is itself a draw from the simulated distribution, the test
    should not reject more often than its nominal level."""
    rng = np.random.default_rng(0)
    k, B = 5, 600
    A = rng.normal(size=(k, k))
    cov = A @ A.T + np.eye(k)
    ps = []
    for _ in range(200):
        draws = rng.multivariate_normal(np.zeros(k), cov, size=B + 1)
        p, _, _ = mahalanobis_test(draws[:B], draws[B])
        ps.append(p)
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.12


def test_mahalanobis_flags_a_shifted_observation_and_says_where():
    rng = np.random.default_rng(1)
    sim = rng.normal(size=(500, 4))
    obs = np.array([0.0, 6.0, 0.0, 0.0])
    p, d, contrib = mahalanobis_test(sim, obs)
    assert p < 0.01 and d > 0
    assert np.argmax(np.abs(contrib)) == 1
    assert contrib[1] > 4


def test_mahalanobis_survives_a_singular_covariance():
    """Census entries sum to a constant, so the covariance is always singular; the
    pseudo-inverse has to absorb that rather than raising."""
    rng = np.random.default_rng(2)
    part = rng.multinomial(20, [0.3, 0.3, 0.4], size=300).astype(float)
    assert np.linalg.matrix_rank(np.cov(part, rowvar=False)) < 3
    p, d, _ = mahalanobis_test(part, part.mean(0))
    assert 0.0 < p <= 1.0 and np.isfinite(d)


def test_auxiliary_returns_the_four_statistics_with_consistent_totals():
    X = _random_digraphs(5, 12, 0.25, 9)
    aux = auxiliary(X)
    assert set(aux) == {"outdegree", "indegree", "triad census", "geodesic"}
    assert (aux["outdegree"].sum(axis=1) == 12).all()
    assert (aux["indegree"].sum(axis=1) == 12).all()
    assert (aux["triad census"].sum(axis=1) == 220).all()
    assert (aux["geodesic"].sum(axis=1) == 132).all()


def test_embedded_triad_table_matches_a_fresh_networkx_build():
    """The table in the module is a generated copy. If it ever drifts from what networkx
    says, every triad census silently changes, so check the copy against its source."""
    from saomsim.gof import build_triad_lookup

    np.testing.assert_array_equal(triad_lookup(), build_triad_lookup())
