"""
Validation suite for saomsim.

The central tests compare every vectorised change statistic, entry by entry,
against a brute-force reference that toggles the tie and recomputes the actor
statistic from its definition. This is the M0 gate: an incorrect simulator
produces a confidently wrong posterior with no diagnostic signature.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saomsim import (  # noqa: E402
    estimate_mom,
    make_effects,
    observed_distance,
    random_network,
    simulate_panel,
    simulate_period,
    statistics,
)
from saomsim.backend import softmax  # noqa: E402
from saomsim.effects import StepContext  # noqa: E402
from saomsim.reference import (  # noqa: E402
    network_statistic,
    reference_choice_probabilities,
    reference_delta,
)

EFFECT_NAMES = ["density", "recip", "transTrip", "cycle3", "sameX", "altX", "egoX"]


def _effect_for(name, covname="v"):
    if name in ("sameX", "altX", "egoX"):
        return make_effects([(name, covname)])[0]
    return make_effects([name])[0]


# ---------------------------------------------------------------------------
# Change statistics vs brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EFFECT_NAMES)
@pytest.mark.parametrize("n,density", [(8, 0.2), (12, 0.35), (15, 0.6)])
def test_change_statistics_match_reference(name, n, density):
    rng = np.random.default_rng(hash((name, n)) % 2**32)
    B = 6
    X = random_network(n, density, rng, B=B)
    v = rng.integers(0, 3, size=n).astype(float)
    cov = {"v": v}
    eff = _effect_for(name)

    for actor_i in range(n):
        actor = np.full(B, actor_i)
        ctx = StepContext(X, actor, cov)
        got = eff.delta(ctx)
        got[np.arange(B), actor] = 0.0          # no-change option

        for b in range(B):
            want = reference_delta(X[b], actor_i, name, v)
            np.testing.assert_allclose(
                got[b], want, atol=1e-9,
                err_msg=f"{name}: chain {b}, actor {actor_i}",
            )


@pytest.mark.parametrize("name", EFFECT_NAMES)
def test_change_statistics_with_random_actors(name):
    """Same check, but with a different focal actor per chain (the real path)."""
    rng = np.random.default_rng(7)
    n, B = 11, 16
    X = random_network(n, 0.3, rng, B=B)
    v = rng.integers(0, 4, size=n).astype(float)
    eff = _effect_for(name)

    actor = rng.integers(0, n, size=B)
    ctx = StepContext(X, actor, {"v": v})
    got = eff.delta(ctx)
    got[np.arange(B), actor] = 0.0

    for b in range(B):
        want = reference_delta(X[b], int(actor[b]), name, v)
        np.testing.assert_allclose(got[b], want, atol=1e-9)


# ---------------------------------------------------------------------------
# Network statistics vs brute force
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", EFFECT_NAMES)
@pytest.mark.parametrize("n", [7, 13])
def test_network_statistics_match_reference(name, n):
    rng = np.random.default_rng(hash((name, n, "net")) % 2**32)
    X = random_network(n, 0.3, rng, B=4)
    v = rng.integers(0, 3, size=n).astype(float)
    eff = _effect_for(name)

    got = eff.statistic(X, {"v": v})
    for b in range(4):
        want = network_statistic(X[b], name, v)
        assert got[b] == pytest.approx(want, abs=1e-9)


def test_statistics_stacking_order():
    rng = np.random.default_rng(3)
    n = 9
    X = random_network(n, 0.25, rng, B=5)
    v = rng.integers(0, 2, size=n).astype(float)
    effects = make_effects(["density", "recip", ("sameX", "v")])
    S = statistics(X, effects, {"v": v})
    assert S.shape == (5, 3)
    np.testing.assert_allclose(S[:, 0], X.sum(axis=(1, 2)))


# ---------------------------------------------------------------------------
# Choice probabilities
# ---------------------------------------------------------------------------

def test_choice_probabilities_match_reference():
    rng = np.random.default_rng(11)
    n, B = 10, 5
    X = random_network(n, 0.3, rng, B=B)
    v = rng.integers(0, 3, size=n).astype(float)
    names = ["density", "recip", "transTrip", "sameX"]
    theta = np.array([-1.0, 0.8, 0.25, 0.5])
    effects = [_effect_for(nm) for nm in names]

    actor = rng.integers(0, n, size=B)
    ctx = StepContext(X, actor, {"v": v})
    utility = np.zeros((B, n))
    for th, eff in zip(theta, effects):
        utility += th * eff.delta(ctx)
    utility[np.arange(B), actor] = 0.0
    got = softmax(utility, axis=1)

    for b in range(B):
        want = reference_choice_probabilities(X[b], int(actor[b]), theta, names, v)
        np.testing.assert_allclose(got[b], want, atol=1e-9)


def test_no_change_option_has_zero_utility():
    """The do-nothing option must always sit at utility exactly zero."""
    rng = np.random.default_rng(5)
    n, B = 8, 3
    X = random_network(n, 0.4, rng, B=B)
    effects = make_effects(["density", "recip", "transTrip"])
    actor = rng.integers(0, n, size=B)
    ctx = StepContext(X, actor, {})
    utility = sum(th * e.delta(ctx) for th, e in zip([-2.0, 1.0, 0.5], effects))
    utility[np.arange(B), actor] = 0.0
    assert np.all(utility[np.arange(B), actor] == 0.0)


# ---------------------------------------------------------------------------
# Simulator invariants
# ---------------------------------------------------------------------------

def test_conditional_simulation_step_count():
    rng = np.random.default_rng(1)
    X0 = random_network(20, 0.15, rng, B=32)
    effects = make_effects(["density", "recip"])
    res = simulate_period(X0, [-1.0, 0.5], effects, n_steps=25, rng=rng)
    assert np.all(res.n_steps == 25)
    assert np.all(res.n_changes <= 25)


def test_simulation_preserves_binary_and_zero_diagonal():
    rng = np.random.default_rng(2)
    n = 18
    X0 = random_network(n, 0.2, rng, B=16)
    effects = make_effects(["density", "recip", "transTrip"])
    res = simulate_period(X0, [-1.2, 0.9, 0.2], effects, n_steps=60, rng=rng)
    assert set(np.unique(res.X)).issubset({0.0, 1.0})
    idx = np.arange(n)
    assert np.all(res.X[:, idx, idx] == 0.0)


def test_simulation_is_reproducible_from_seed():
    effects = make_effects(["density", "recip", "transTrip"])
    out = []
    for _ in range(2):
        rng = np.random.default_rng(12345)
        X0 = random_network(15, 0.2, rng, B=8)
        res = simulate_period(X0, [-1.0, 0.7, 0.3], effects, n_steps=40, rng=rng)
        out.append(res.X)
    np.testing.assert_array_equal(out[0], out[1])


def test_different_seeds_give_different_draws():
    effects = make_effects(["density", "recip"])
    rng_a = np.random.default_rng(1)
    X0 = random_network(15, 0.2, np.random.default_rng(99), B=8)
    a = simulate_period(X0, [-1.0, 0.5], effects, n_steps=50, rng=rng_a).X
    rng_b = np.random.default_rng(2)
    b = simulate_period(X0, [-1.0, 0.5], effects, n_steps=50, rng=rng_b).X
    assert not np.array_equal(a, b)


def test_unconditional_step_count_is_poisson_mean():
    rng = np.random.default_rng(4)
    n, lam, B = 12, 3.0, 4000
    X0 = random_network(n, 0.2, rng, B=B)
    effects = make_effects(["density"])
    res = simulate_period(X0, [-1.0], effects, lam=lam, rng=rng)
    # mean number of micro-steps should be n * lam
    assert res.n_steps.mean() == pytest.approx(n * lam, rel=0.03)


def test_zero_theta_gives_uniform_choice():
    """With theta = 0 every option is equally likely, so tie density drifts to 0.5."""
    rng = np.random.default_rng(6)
    n = 12
    X0 = random_network(n, 0.5, rng, B=200)
    effects = make_effects(["density"])
    res = simulate_period(X0, [0.0], effects, n_steps=400, rng=rng)
    dens = res.X.sum(axis=(1, 2)).mean() / (n * (n - 1))
    assert 0.40 < dens < 0.60


def test_strong_negative_density_empties_network():
    rng = np.random.default_rng(8)
    n = 12
    X0 = random_network(n, 0.5, rng, B=40)
    effects = make_effects(["density"])
    res = simulate_period(X0, [-6.0], effects, n_steps=800, rng=rng)
    assert res.X.sum(axis=(1, 2)).mean() < 2.0


def test_reciprocity_increases_mutual_dyads():
    rng = np.random.default_rng(9)
    n = 14
    X0 = random_network(n, 0.2, rng, B=60)
    effects = make_effects(["density", "recip"])
    low = simulate_period(X0, [-1.5, 0.0], effects, n_steps=500, rng=rng).X
    high = simulate_period(X0, [-1.5, 3.0], effects, n_steps=500, rng=rng).X
    mut_low = (low * np.swapaxes(low, 1, 2)).sum(axis=(1, 2)).mean()
    mut_high = (high * np.swapaxes(high, 1, 2)).sum(axis=(1, 2)).mean()
    assert mut_high > mut_low


def test_homophily_concentrates_ties_within_groups():
    rng = np.random.default_rng(10)
    n = 16
    v = np.repeat([0.0, 1.0], n // 2)
    X0 = random_network(n, 0.2, rng, B=60)
    effects = make_effects(["density", ("sameX", "v")])
    same = (v[:, None] == v[None, :]).astype(float)

    neutral = simulate_period(X0, [-1.5, 0.0], effects, {"v": v},
                              n_steps=600, rng=rng).X
    homoph = simulate_period(X0, [-1.5, 2.5], effects, {"v": v},
                             n_steps=600, rng=rng).X

    frac = lambda X: ((X * same).sum(axis=(1, 2)) /
                      np.maximum(X.sum(axis=(1, 2)), 1)).mean()
    assert frac(homoph) > frac(neutral) + 0.10


def test_observed_distance():
    X1 = np.zeros((1, 5, 5))
    X2 = np.zeros((1, 5, 5))
    X2[0, 0, 1] = 1
    X2[0, 2, 3] = 1
    assert observed_distance(X1, X2)[0] == 2


def test_panel_shape_and_first_wave():
    rng = np.random.default_rng(13)
    X0 = random_network(10, 0.2, rng, B=5)
    effects = make_effects(["density", "recip"])
    panel = simulate_panel(X0, [-1.0, 0.5], effects, n_waves=4,
                           n_steps=20, rng=rng)
    assert panel.shape == (5, 4, 10, 10)
    np.testing.assert_array_equal(panel[:, 0], X0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_both_or_neither_regime():
    rng = np.random.default_rng(14)
    X0 = random_network(8, 0.2, rng, B=2)
    effects = make_effects(["density"])
    with pytest.raises(ValueError, match="exactly one"):
        simulate_period(X0, [-1.0], effects, rng=rng)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_period(X0, [-1.0], effects, n_steps=5, lam=1.0, rng=rng)


def test_rejects_theta_effect_mismatch():
    rng = np.random.default_rng(15)
    X0 = random_network(8, 0.2, rng, B=2)
    effects = make_effects(["density", "recip"])
    with pytest.raises(ValueError, match="theta has"):
        simulate_period(X0, [-1.0], effects, n_steps=5, rng=rng)


def test_rejects_nonzero_diagonal():
    rng = np.random.default_rng(16)
    X0 = random_network(8, 0.2, rng, B=2)
    X0[0, 3, 3] = 1.0
    effects = make_effects(["density"])
    with pytest.raises(ValueError, match="zero diagonal"):
        simulate_period(X0, [-1.0], effects, n_steps=5, rng=rng)


def test_unknown_effect_name_is_helpful():
    with pytest.raises(KeyError, match="unknown effect"):
        make_effects(["transitivity"])


# ---------------------------------------------------------------------------
# End-to-end recovery (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mom_recovers_known_parameters():
    """Simulate from known theta, estimate, check the truth is within ~2.5 s.e."""
    rng = np.random.default_rng(2024)
    n = 30
    theta_true = np.array([-1.8, 1.5])
    effects = make_effects(["density", "recip"])

    X1 = random_network(n, 0.10, rng, B=1)
    X2 = simulate_period(X1, theta_true, effects, n_steps=120, rng=rng).X[0]

    res = estimate_mom(X1[0], X2, effects, rng=rng,
                       n_sim_phase1=150, n_iter_per_subphase=40,
                       n_sim_phase3=600)

    assert res.converged(threshold=0.25), res.summary()
    z = np.abs(res.theta - theta_true) / np.maximum(res.se, 1e-6)
    assert np.all(z < 2.5), res.summary()
