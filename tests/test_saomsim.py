"""Test suite for saomsim.

The load-bearing tests are the ``*_match_reference`` ones: every vectorized
kernel is checked against the loop implementation in ``saomsim.reference`` on
random networks at several densities. Everything else is behavioural.
"""

import numpy as np
import pytest

from saomsim import (
    KINDS,
    Effect,
    Model,
    estimate,
    moments,
    random_network,
    rate_statistic,
    simulate_panel,
    simulate_period,
    statistics,
)
from saomsim import reference as ref
from saomsim.backend import batched_row_products, categorical_sample, softmax, toggle

DENSITIES = [0.05, 0.3, 0.7]
ALL_EFFECTS = [
    "density",
    "recip",
    "transTrip",
    "cycle3",
    ("sameX", "grp"),
    ("altX", "val"),
    ("egoX", "val"),
]


def effect_id(e):
    return e if isinstance(e, str) else e[0]


def make_covs(n, rng):
    return {"grp": rng.integers(0, 3, size=n).astype(float), "val": rng.normal(size=n)}


def make_model(effects=ALL_EFFECTS, n=10, rng=None):
    rng = rng or np.random.default_rng(1)
    covs = make_covs(n, rng)
    return Model(effects, covs), covs


def seed_from(*parts):
    return abs(hash(tuple(str(p) for p in parts))) % 2**32


# ------------------------------------------------------------------ backend


@pytest.mark.parametrize("density", DENSITIES)
def test_row_products_match_loops(density):
    rng = np.random.default_rng(10)
    B, n = 6, 9
    X = random_network(B, n, density, rng).astype(float)
    actor = rng.integers(0, n, size=B)
    rows = batched_row_products(X, actor)
    for b in range(B):
        i = actor[b]
        x = X[b]
        assert np.array_equal(rows.out_row[b], x[i, :])
        assert np.array_equal(rows.in_col[b], x[:, i])
        for j in range(n):
            assert rows.two_path[b, j] == sum(x[i, h] * x[h, j] for h in range(n))
            assert rows.shared_out[b, j] == sum(x[i, h] * x[j, h] for h in range(n))
            assert rows.back_path[b, j] == sum(x[j, h] * x[h, i] for h in range(n))


def test_row_products_respect_needs():
    rng = np.random.default_rng(11)
    X = random_network(3, 5, 0.4, rng)
    actor = np.array([0, 1, 2])
    rows = batched_row_products(X, actor, frozenset({"back_path"}))
    assert rows.two_path is None and rows.shared_out is None
    assert rows.back_path is not None
    full = batched_row_products(X, actor)
    assert np.array_equal(rows.back_path, full.back_path)


def test_softmax_rows_sum_to_one_and_are_shift_invariant():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(5, 7)) * 30
    p = softmax(f)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p, softmax(f + 1000.0))
    assert np.all(np.isfinite(p))


def test_categorical_sample_is_inverse_cdf():
    p = np.array([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    u = np.array([0.65, 0.999, 0.001])
    assert categorical_sample(p, u).tolist() == [1, 0, 2]
    # u exactly on a cdf edge goes to the lower cell; u == 1 never overflows
    assert categorical_sample(p[:1], np.array([0.2]))[0] == 0
    assert categorical_sample(p[:1], np.array([1.0]))[0] == 2


def test_categorical_sample_frequencies():
    rng = np.random.default_rng(3)
    p = np.tile([[0.1, 0.6, 0.3]], (200_000, 1))
    draws = categorical_sample(p, rng.random(200_000))
    freq = np.bincount(draws, minlength=3) / draws.size
    assert np.allclose(freq, [0.1, 0.6, 0.3], atol=0.01)


def test_toggle_flips_only_active_offdiagonal():
    X = np.zeros((3, 4, 4))
    actor = np.array([0, 1, 2])
    target = np.array([1, 1, 3])  # chain 1 picks itself: no change
    active = np.array([True, True, False])  # chain 2 inactive
    toggle(X, actor, target, active)
    assert X[0, 0, 1] == 1 and X.sum() == 1
    toggle(X, actor, target, active)
    assert X.sum() == 0


# ---------------------------------------------------------------- effects


def test_effect_validation():
    with pytest.raises(ValueError):
        Effect("nonsense")
    with pytest.raises(ValueError):
        Effect("sameX")
    with pytest.raises(ValueError):
        Effect("density", "grp")
    with pytest.raises(KeyError):
        Model([("sameX", "missing")], {"grp": np.zeros(4)})
    assert set(KINDS) == {"density", "recip", "transTrip", "cycle3", "sameX", "altX", "egoX"}


def test_model_labels_and_needs():
    m, _ = make_model()
    assert m.labels == [
        "density",
        "recip",
        "transTrip",
        "cycle3",
        "sameX(grp)",
        "altX(val)",
        "egoX(val)",
    ]
    assert m.needs == {"two_path", "shared_out", "back_path"}
    assert Model(["density", "recip"]).needs == frozenset()
    assert Model(["cycle3"]).needs == {"back_path"}


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=effect_id)
@pytest.mark.parametrize("density", DENSITIES)
def test_change_statistics_match_reference(effect, density):
    rng = np.random.default_rng(seed_from(effect, density))
    B, n = 8, 11
    m, covs = make_model([effect], n, rng)
    X = random_network(B, n, density, rng)
    actor = rng.integers(0, n, size=B)
    delta = m.change_statistics(batched_row_products(X, actor), actor)
    assert delta.shape == (B, n, 1)
    for b in range(B):
        expect = ref.change_statistics_row(X[b], actor[b], [effect], covs)
        assert np.allclose(delta[b], expect), f"chain {b}, actor {actor[b]}"


@pytest.mark.parametrize("density", DENSITIES)
def test_change_statistics_all_effects_jointly_match_reference(density):
    rng = np.random.default_rng(int(density * 1000))
    B, n = 5, 12
    m, covs = make_model(ALL_EFFECTS, n, rng)
    X = random_network(B, n, density, rng)
    actor = rng.integers(0, n, size=B)
    delta = m.change_statistics(batched_row_products(X, actor), actor)
    for b in range(B):
        expect = ref.change_statistics_row(X[b], actor[b], ALL_EFFECTS, covs)
        assert np.allclose(delta[b], expect)


def test_no_change_option_is_zero():
    rng = np.random.default_rng(4)
    B, n = 20, 8
    m, _ = make_model(ALL_EFFECTS, n, rng)
    X = random_network(B, n, 0.5, rng)
    actor = rng.integers(0, n, size=B)
    rows = batched_row_products(X, actor)
    delta = m.change_statistics(rows, actor)
    assert np.all(delta[np.arange(B), actor, :] == 0)
    f = m.objective(rows, actor, rng.normal(size=m.K))
    assert np.all(f[np.arange(B), actor] == 0)


def test_change_statistic_antisymmetric_under_toggle():
    """delta on x equals -delta on x with the tie toggled."""
    rng = np.random.default_rng(5)
    n = 9
    m, covs = make_model(ALL_EFFECTS, n, rng)
    x = random_network(1, n, 0.3, rng)[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x2 = x.copy()
            x2[i, j] = 1 - x2[i, j]
            for e in m.effects:
                d1 = ref.change_statistic(x, i, j, e, covs)
                d2 = ref.change_statistic(x2, i, j, e, covs)
                assert d1 == -d2


@pytest.mark.parametrize("effect", ALL_EFFECTS, ids=effect_id)
@pytest.mark.parametrize("density", DENSITIES)
def test_statistics_match_reference(effect, density):
    rng = np.random.default_rng(seed_from("stats", effect, density))
    B, n = 6, 10
    m, covs = make_model([effect], n, rng)
    X = random_network(B, n, density, rng)
    S = statistics(X, m)
    assert S.shape == (B, 1)
    for b in range(B):
        assert np.allclose(S[b], ref.statistics(X[b], [effect], covs))


def test_statistics_are_sum_of_actor_statistics():
    rng = np.random.default_rng(6)
    n = 8
    m, covs = make_model(ALL_EFFECTS, n, rng)
    x = random_network(1, n, 0.4, rng)[0]
    S = statistics(x[None], m)[0]
    for k, e in enumerate(m.effects):
        assert np.isclose(S[k], sum(ref.actor_statistic(x, i, e, covs) for i in range(n)))


def test_recip_and_cycle3_counting_convention():
    """recip counts each mutual dyad twice, cycle3 each 3-cycle three times."""
    x = np.zeros((1, 3, 3), dtype=np.int8)
    x[0, 0, 1] = x[0, 1, 0] = 1
    assert statistics(x, Model(["recip"]))[0, 0] == 2
    c = np.zeros((1, 3, 3), dtype=np.int8)
    c[0, 0, 1] = c[0, 1, 2] = c[0, 2, 0] = 1
    assert statistics(c, Model(["cycle3"]))[0, 0] == 3
    assert statistics(c, Model(["transTrip"]))[0, 0] == 0
    t = np.zeros((1, 3, 3), dtype=np.int8)
    t[0, 0, 1] = t[0, 1, 2] = t[0, 0, 2] = 1
    assert statistics(t, Model(["transTrip"]))[0, 0] == 1


def test_per_chain_covariates():
    rng = np.random.default_rng(7)
    B, n = 4, 7
    grp = rng.integers(0, 2, size=(B, n)).astype(float)
    m = Model([("sameX", "grp")], {"grp": grp})
    X = random_network(B, n, 0.4, rng)
    actor = rng.integers(0, n, size=B)
    delta = m.change_statistics(batched_row_products(X, actor), actor)
    S = statistics(X, m)
    for b in range(B):
        covs_b = {"grp": grp[b]}
        assert np.allclose(
            delta[b], ref.change_statistics_row(X[b], actor[b], [("sameX", "grp")], covs_b)
        )
        assert np.allclose(S[b], ref.statistics(X[b], [("sameX", "grp")], covs_b))
    with pytest.raises(ValueError):
        m.statistics(random_network(B + 1, n, 0.4, rng))


def test_objective_matches_reference_with_per_chain_theta():
    rng = np.random.default_rng(8)
    B, n = 5, 8
    m, covs = make_model(ALL_EFFECTS, n, rng)
    X = random_network(B, n, 0.3, rng)
    actor = rng.integers(0, n, size=B)
    theta = rng.normal(size=(B, m.K))
    rows = batched_row_products(X, actor)
    f = m.objective(rows, actor, theta)
    for b in range(B):
        assert np.allclose(f[b], ref.objective(X[b], actor[b], theta[b], ALL_EFFECTS, covs))
    f_shared = m.objective(rows, actor, theta[0])
    assert np.allclose(f_shared[0], f[0])


# --------------------------------------------------------------- simulate


def test_random_network_shape_dtype_diagonal():
    X = random_network(4, 6, 0.5, np.random.default_rng(0))
    assert X.shape == (4, 6, 6) and X.dtype == np.int8
    assert np.all(X[:, np.arange(6), np.arange(6)] == 0)
    assert set(np.unique(X)) <= {0, 1}


def test_simulate_period_shape_dtype_diagonal():
    rng = np.random.default_rng(0)
    X0 = random_network(10, 12, 0.2, rng)
    X1 = simulate_period(X0, [-1.0, 0.5], 2.0, Model(["density", "recip"]), rng)
    assert X1.shape == X0.shape and X1.dtype == np.int8
    assert np.all(X1[:, np.arange(12), np.arange(12)] == 0)
    assert set(np.unique(X1)) <= {0, 1}


def test_simulate_does_not_modify_input():
    rng = np.random.default_rng(0)
    X0 = random_network(5, 8, 0.2, rng)
    before = X0.copy()
    simulate_period(X0, [-1.0], 2.0, Model(["density"]), rng)
    assert np.array_equal(X0, before)


def test_reproducible_from_seed():
    m = Model(["density", "recip", "transTrip"])
    X0 = random_network(50, 15, 0.1, np.random.default_rng(0))
    a = simulate_period(X0, [-2.0, 1.0, 0.3], 3.0, m, np.random.default_rng(123))
    b = simulate_period(X0, [-2.0, 1.0, 0.3], 3.0, m, np.random.default_rng(123))
    c = simulate_period(X0, [-2.0, 1.0, 0.3], 3.0, m, np.random.default_rng(124))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_chain_unaffected_by_other_chains_parameters():
    """With a shared generator, chain b's draws depend on its position in the
    batch but not on what the other chains do with theirs."""
    m = Model(["density", "recip"])
    X0 = random_network(4, 10, 0.2, np.random.default_rng(0))
    theta_a = np.array([[-2.0, 1.0]] * 4)
    theta_b = theta_a.copy()
    theta_b[3] = [0.0, 0.0]
    a = simulate_period(X0, theta_a, 2.0, m, np.random.default_rng(9))
    b = simulate_period(X0, theta_b, 2.0, m, np.random.default_rng(9))
    assert np.array_equal(a[:3], b[:3])


def test_zero_rate_leaves_network_unchanged():
    rng = np.random.default_rng(0)
    X0 = random_network(5, 8, 0.3, rng)
    X1 = simulate_period(X0, [-1.0], 0.0, Model(["density"]), rng)
    assert np.array_equal(X0, X1)


def test_number_of_ministeps_is_poisson_n_rate():
    rng = np.random.default_rng(0)
    B, n, rate = 4000, 10, 2.0
    X0 = random_network(B, n, 0.1, rng)
    _, steps = simulate_period(X0, [-1.0], rate, Model(["density"]), rng, return_n_steps=True)
    assert abs(steps.mean() - n * rate) < 0.5
    assert abs(steps.var() - n * rate) < 2.0


def test_rate_increases_tie_changes():
    rng = np.random.default_rng(0)
    m = Model(["density"])
    X0 = random_network(300, 12, 0.3, rng)
    changes = []
    for rate in [0.5, 1.0, 2.0, 4.0]:
        X1 = simulate_period(X0, [-0.85], rate, m, rng)  # logit(0.3): density stays put
        changes.append(rate_statistic(X0, X1).mean())
    assert all(np.diff(changes) > 0)


def test_density_parameter_controls_density():
    rng = np.random.default_rng(0)
    m = Model(["density"])
    X0 = random_network(400, 12, 0.3, rng)
    dens = []
    for beta in [-3.0, -1.0, 1.0]:
        X1 = simulate_period(X0, [beta], 10.0, m, rng)
        dens.append(X1.mean() * 12 / 11)
    assert all(np.diff(dens) > 0)


def exact_stationary_distribution(n, model, theta):
    """Stationary law of the ministep chain on all 2^(n(n-1)) digraphs, by brute force."""
    import itertools

    offdiag = [(i, j) for i in range(n) for j in range(n) if i != j]
    states = []
    for bits in itertools.product([0, 1], repeat=len(offdiag)):
        x = np.zeros((n, n), dtype=np.int8)
        for (i, j), b in zip(offdiag, bits, strict=True):
            x[i, j] = b
        states.append(x)
    index = {s.tobytes(): k for k, s in enumerate(states)}
    S = len(states)
    P = np.zeros((S, S))
    for a, x in enumerate(states):
        for i in range(n):
            f = ref.objective(x, i, theta, model.effects, model.covariates)
            p = np.exp(f - f.max())
            p /= p.sum()
            for j in range(n):
                if j == i:
                    P[a, a] += p[j] / n
                else:
                    x2 = x.copy()
                    x2[i, j] = 1 - x2[i, j]
                    P[a, index[x2.tobytes()]] += p[j] / n
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    return states, index, pi


def test_simulation_matches_exact_stationary_distribution():
    """The simulator's long-run law equals the exact stationary distribution of
    the ministep Markov chain on n = 3 (64 states). This checks the *dynamics*,
    not just the statistics: actor choice, softmax over the neighbourhood,
    the no-change option and the toggle all have to be right."""
    n = 3
    theta = np.array([-0.4, 1.0, 0.8])
    model = Model(["density", "recip", "cycle3"])
    states, index, pi = exact_stationary_distribution(n, model, theta)
    rng = np.random.default_rng(0)
    B = 40_000
    X0 = random_network(B, n, 0.5, rng)
    X1 = simulate_period(X0, theta, 40.0, model, rng)  # ~120 ministeps: well mixed
    counts = np.zeros(len(states))
    for x in X1:
        counts[index[x.tobytes()]] += 1
    freq = counts / B
    assert pi.min() > 0 and abs(pi.sum() - 1) < 1e-10
    assert np.abs(freq - pi).max() < 0.006, np.abs(freq - pi).max()
    # and the exact law is genuinely non-uniform, so this is not a vacuous check
    assert pi.max() / pi.min() > 5


def test_reciprocity_increases_mutual_dyads():
    rng = np.random.default_rng(0)
    m = Model(["density", "recip"])
    X0 = random_network(400, 15, 0.1, rng)
    mutual = []
    for beta in [-1.0, 0.0, 1.0, 2.0]:
        X1 = simulate_period(X0, [-2.0, beta], 4.0, m, rng)
        mutual.append(statistics(X1, Model(["recip"])).mean() / 2)
    assert all(np.diff(mutual) > 0)


def test_homophily_increases_within_group_share():
    rng = np.random.default_rng(0)
    n = 16
    grp = np.repeat([0.0, 1.0], n // 2)
    m = Model(["density", ("sameX", "grp")], {"grp": grp})
    same_model = Model([("sameX", "grp")], {"grp": grp})
    X0 = random_network(400, n, 0.1, rng)
    share = []
    for beta in [0.0, 0.5, 1.0, 2.0]:
        X1 = simulate_period(X0, [-2.0, beta], 4.0, m, rng)
        share.append(statistics(X1, same_model).sum() / X1.sum())
    assert all(np.diff(share) > 0)
    assert abs(share[0] - (n / 2 - 1) / (n - 1)) < 0.03  # no homophily: chance level


def test_transitivity_increases_transitive_triplets():
    rng = np.random.default_rng(0)
    m = Model(["density", "transTrip"])
    tt_model = Model(["transTrip"])
    X0 = random_network(300, 15, 0.1, rng)
    tt = []
    for beta in [0.0, 0.3, 0.6]:
        X1 = simulate_period(X0, [-2.0, beta], 4.0, m, rng)
        per_tie = statistics(X1, tt_model)[:, 0] / np.maximum(X1.sum(axis=(1, 2)), 1)
        tt.append(per_tie.mean())
    assert all(np.diff(tt) > 0)


def test_degenerate_outcomes_are_returned_not_filtered():
    rng = np.random.default_rng(0)
    m = Model(["density"])
    X0 = random_network(50, 8, 0.5, rng)
    empty = simulate_period(X0, [-15.0], 20.0, m, rng)
    full = simulate_period(X0, [15.0], 20.0, m, rng)
    assert empty.shape == X0.shape and empty.sum() == 0
    assert full.shape == X0.shape and full.sum() == 50 * 8 * 7


def test_parameter_validation():
    rng = np.random.default_rng(0)
    X0 = random_network(3, 5, 0.2, rng)
    m = Model(["density", "recip"])
    with pytest.raises(ValueError):
        simulate_period(X0, [-1.0], 1.0, m, rng)  # K mismatch
    with pytest.raises(ValueError):
        simulate_period(X0, np.zeros((2, 2)), 1.0, m, rng)  # B mismatch
    with pytest.raises(ValueError):
        simulate_period(X0, [-1.0, 0.0], -1.0, m, rng)
    with pytest.raises(ValueError):
        simulate_period(X0[0], [-1.0, 0.0], 1.0, m, rng)


def test_per_chain_rate():
    rng = np.random.default_rng(0)
    B, n = 400, 10
    X0 = random_network(B, n, 0.3, rng)
    rate = np.where(np.arange(B) < B // 2, 0.2, 3.0)
    X1 = simulate_period(X0, [-0.85], rate, Model(["density"]), rng)
    ch = rate_statistic(X0, X1)
    assert ch[: B // 2].mean() < ch[B // 2 :].mean()


# ------------------------------------------------------------------ panels


def test_simulate_panel_shapes_and_first_wave():
    rng = np.random.default_rng(0)
    X0 = random_network(6, 9, 0.2, rng)
    P = simulate_panel(X0, [-1.5, 0.5], 1.0, Model(["density", "recip"]), rng, waves=4)
    assert P.shape == (4, 6, 9, 9) and P.dtype == np.int8
    assert np.array_equal(P[0], X0)
    assert np.all(P[:, :, np.arange(9), np.arange(9)] == 0)


def test_simulate_panel_is_sequential_periods():
    m = Model(["density", "recip"])
    X0 = random_network(6, 9, 0.2, np.random.default_rng(0))
    P = simulate_panel(X0, [-1.5, 0.5], 1.0, m, np.random.default_rng(42), waves=3)
    rng = np.random.default_rng(42)
    X1 = simulate_period(X0, [-1.5, 0.5], 1.0, m, rng)
    X2 = simulate_period(X1, [-1.5, 0.5], 1.0, m, rng)
    assert np.array_equal(P[1], X1) and np.array_equal(P[2], X2)


def test_simulate_panel_per_period_parameters():
    m = Model(["density"])
    X0 = random_network(200, 10, 0.3, np.random.default_rng(0))
    theta = np.array([[-4.0], [4.0]])
    rate = np.array([30.0, 30.0])  # 300 ministeps per period: enough to cross 90 ties
    P = simulate_panel(X0, theta, rate, m, np.random.default_rng(1), waves=3)
    assert P[1].mean() < 0.1 and P[2].mean() > 0.8
    with pytest.raises(ValueError):
        simulate_panel(X0, np.array([[-4.0]] * 3), 1.0, m, np.random.default_rng(1), waves=3)
    with pytest.raises(ValueError):
        simulate_panel(X0, [-1.0], 1.0, m, np.random.default_rng(1), waves=1)


# ---------------------------------------------------------------- estimate


def test_moments_layout():
    rng = np.random.default_rng(0)
    m = Model(["density", "recip"])
    X0 = random_network(7, 8, 0.2, rng)
    X1 = simulate_period(X0, [-1.0, 0.5], 1.0, m, rng)
    M = moments(X0, X1, m)
    assert M.shape == (7, 3)
    assert np.array_equal(M[:, 0], rate_statistic(X0, X1))
    assert np.array_equal(M[:, 1:], statistics(X1, m))


def test_estimate_jacobian_has_expected_signs():
    """More rate -> more changes; larger density parameter -> more ties; etc."""
    rng = np.random.default_rng(0)
    m = Model(["density", "recip"])
    X0 = random_network(40, 12, 0.15, rng)
    X1 = simulate_period(X0, [-1.5, 1.0], 2.0, m, rng)
    res = estimate(X0, X1, m, rng, n_sim=10, max_iter=1, fd_step=0.2)
    D = res.jacobian
    assert D[0, 0] > 0  # d(changes)/d(rate)
    assert D[1, 1] > 0  # d(ties)/d(density)
    assert D[2, 2] > 0  # d(recip stat)/d(recip)
    assert res.se.shape == (3,) and np.all(res.se > 0)


def test_estimate_table_formats():
    rng = np.random.default_rng(0)
    m = Model(["density"])
    X0 = random_network(20, 8, 0.2, rng)
    X1 = simulate_period(X0, [-1.0], 1.0, m, rng)
    res = estimate(X0, X1, m, rng, n_sim=5, max_iter=2)
    txt = res.table(truth=[1.0, -1.0])
    assert "rate" in txt and "density" in txt and "converged=" in txt
    assert res.names == ["rate", "density"]
    assert len(res.history) >= 1


@pytest.mark.slow
def test_estimate_recovers_parameters():
    rng = np.random.default_rng(2024)
    n, P = 20, 150
    grp = (np.arange(n) % 2).astype(float)
    m = Model(["density", "recip", ("sameX", "grp")], {"grp": grp})
    truth = np.array([2.5, -2.0, 1.2, 0.6])
    X0 = random_network(P, n, 0.12, rng)
    X1 = simulate_period(X0, truth[1:], truth[0], m, rng)
    res = estimate(X0, X1, m, rng, n_sim=20, max_iter=25, fd_step=0.15)
    z = (res.theta - truth) / res.se
    assert res.converged, res.table(truth)
    assert np.all(np.abs(z) < 3.0), res.table(truth)
