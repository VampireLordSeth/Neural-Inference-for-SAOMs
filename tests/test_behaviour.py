"""Tests for network-behaviour co-evolution (saomsim.behaviour)."""

import numpy as np
import pytest

from conftest import to_np
from saomsim import Model, random_network
from saomsim import reference as ref
from saomsim.backend import NUMPY
from saomsim.behaviour import (
    BEHAVIOUR_KINDS,
    SELECTION_KINDS,
    BehaviourModel,
    BehaviourSpec,
    behaviour_rate_statistic,
    random_behaviour,
    simulate_coevolution,
    spec_from_data,
)

ALL_BEH = list(BEHAVIOUR_KINDS)
ALL_SEL = list(SELECTION_KINDS)


def case(B=6, n=9, seed=0):
    rng = np.random.default_rng(seed)
    X = random_network(B, n, 0.3, rng)
    z = random_behaviour(B, n, 1, 5, rng)
    spec = BehaviourSpec(
        1, 5, zbar=rng.uniform(2, 4, size=B), sim_mean=rng.uniform(0.5, 0.8, size=B)
    )
    return rng, X, z, spec


def test_spec_from_data_matches_rsiena_constants():
    Z = np.loadtxt("benchmarks/s50a.csv", delimiter=",")
    spec = spec_from_data(Z, 1, 5)
    assert spec.zbar == pytest.approx(3.11333, abs=1e-5)
    assert spec.sim_mean == pytest.approx(0.67439, abs=1e-5)
    assert spec.z_range == 4
    one = spec_from_data(Z[:, 0], 1, 5)
    assert one.zbar == pytest.approx(Z[:, 0].mean())


@pytest.mark.parametrize("effect", ALL_BEH)
def test_move_contributions_match_reference(backend, effect):
    rng, X, z, spec = case()
    B, n = z.shape
    bm = BehaviourModel(effects=[effect], selection=[])
    actor = rng.integers(0, n, size=B)
    c = to_np(
        backend,
        bm.move_contributions(
            backend.network(X), backend.array(z), backend.int_array(actor), spec, backend
        ),
    )
    assert c.shape == (B, 3, 1)
    for b in range(B):
        for d_idx, d in enumerate((-1, 0, 1)):
            want = ref.behaviour_change(
                X[b], z[b], actor[b], d, effect, spec.zbar[b], spec.sim_mean[b], spec.z_range
            )
            assert c[b, d_idx, 0] == pytest.approx(want, abs=1e-9), (effect, b, d)


@pytest.mark.parametrize("effect", ALL_SEL)
def test_selection_contributions_match_reference(backend, effect):
    rng, X, z, spec = case(seed=1)
    B, n = z.shape
    bm = BehaviourModel(effects=[], selection=[effect])
    actor = rng.integers(0, n, size=B)
    c = to_np(
        backend,
        bm.selection_contributions(backend.array(z), backend.int_array(actor), spec, backend),
    )
    assert c.shape == (B, n, 1)
    for b in range(B):
        for j in range(n):
            want = ref.selection_creation(
                z[b], actor[b], j, effect, spec.zbar[b], spec.sim_mean[b], spec.z_range
            )
            assert c[b, j, 0] == pytest.approx(want, abs=1e-9)


def test_target_statistics_match_reference_sums():
    rng, X, z, spec = case(seed=2)
    B, n = z.shape
    bm = BehaviourModel(effects=ALL_BEH, selection=ALL_SEL)
    z_end = random_behaviour(B, n, 1, 5, rng)
    X_end = random_network(B, n, 0.3, rng)
    S = bm.statistics(X, z_end, spec)
    for k, e in enumerate(ALL_BEH):
        for b in range(B):
            want = sum(
                ref.behaviour_actor_statistic(
                    X[b], z_end[b], i, e, spec.zbar[b], spec.sim_mean[b], spec.z_range
                )
                for i in range(n)
            )
            assert S[b, k] == pytest.approx(want, abs=1e-9)
    T = bm.selection_statistics(X_end, z, spec)
    for k, e in enumerate(ALL_SEL):
        for b in range(B):
            want = sum(
                X_end[b, i, j]
                * ref.selection_creation(
                    z[b], i, j, e, spec.zbar[b], spec.sim_mean[b], spec.z_range
                )
                for i in range(n)
                for j in range(n)
                if i != j
            )
            assert T[b, k] == pytest.approx(want, abs=1e-9)
    assert np.array_equal(behaviour_rate_statistic(z, z_end), np.abs(z - z_end).sum(axis=1))


def test_behaviour_objective_excludes_off_scale_moves(backend):
    rng, X, z, spec = case(seed=3)
    B, n = z.shape
    z[:, 0] = 1
    z[:, 1] = 5
    bm = BehaviourModel(effects=["linear"], selection=[])
    th = backend.array(np.zeros((B, 1)))
    f_lo = to_np(
        backend,
        bm.behaviour_objective(
            backend.network(X),
            backend.array(z),
            backend.int_array(np.zeros(B, int)),
            th,
            spec,
            backend,
        ),
    )
    f_hi = to_np(
        backend,
        bm.behaviour_objective(
            backend.network(X),
            backend.array(z),
            backend.int_array(np.ones(B, int)),
            th,
            spec,
            backend,
        ),
    )
    assert np.all(f_lo[:, 0] < -1e20) and np.all(f_lo[:, 1:] == 0)
    assert np.all(f_hi[:, 2] < -1e20) and np.all(f_hi[:, :2] == 0)


def coev_case(B=64, n=15, seed=4):
    rng = np.random.default_rng(seed)
    X0 = random_network(B, n, 0.15, rng)
    z0 = random_behaviour(B, n, 1, 5, rng)
    spec = BehaviourSpec(1, 5, zbar=z0.mean(axis=1), sim_mean=np.full(B, 0.7))
    model = Model(["density", "recip"])
    bm = BehaviourModel(effects=["linear", "quad", "avAlt"], selection=["egoZ", "altZ", "simZ"])
    return rng, X0, z0, spec, model, bm


def test_simulate_coevolution_shapes_bounds_reproducible(backend):
    rng, X0, z0, spec, model, bm = coev_case()
    args = (X0, z0, [-1.5, 1.0], [0.0, 0.0, 1.0], [0.0, -0.3, 1.0], 2.0, 1.0, model, bm, spec)
    X1, z1, info = simulate_coevolution(
        *args, np.random.default_rng(1), backend=backend, return_info=True
    )
    assert X1.shape == X0.shape and X1.dtype == np.int8 and z1.shape == z0.shape
    assert z1.min() >= 1 and z1.max() <= 5
    assert np.all(X1[:, np.arange(15), np.arange(15)] == 0)
    assert np.all(info["n_net"] + info["n_beh"] == info["n_events"])
    assert 0.55 < info["n_net"].sum() / info["n_events"].sum() < 0.78  # rate 2 vs 1 -> 2/3 network
    X1b, z1b = simulate_coevolution(*args, np.random.default_rng(1), backend=backend)
    assert np.array_equal(X1, X1b) and np.array_equal(z1, z1b)
    assert np.array_equal(X0, X0) and np.array_equal(z0, np.clip(z0, 1, 5))  # inputs untouched
    Xz, zz = simulate_coevolution(
        X0, z0, [-1.5, 1.0], [0, 0, 0], [0, 0, 0], 0.0, 0.0, model, bm, spec, rng, backend=backend
    )
    assert np.array_equal(Xz, X0) and np.array_equal(zz, z0)


def test_behaviour_rate_controls_behaviour_changes(backend):
    rng, X0, z0, spec, model, bm = coev_case(B=200)
    changes = []
    for rb in (0.5, 2.0):
        _, z1 = simulate_coevolution(
            X0,
            z0,
            [-1.5, 1.0],
            [0, 0, 0],
            [0, 0, 0],
            1.0,
            rb,
            model,
            bm,
            spec,
            rng,
            backend=backend,
        )
        changes.append(behaviour_rate_statistic(z0, z1).mean())
    assert changes[1] > changes[0]


def test_influence_increases_friend_similarity(backend):
    """Positive avAlt pulls actors toward their alters: friends end up more similar."""
    rng, X0, z0, spec, model, bm = coev_case(B=300, n=20)
    z0 = rng.integers(1, 6, size=z0.shape)  # spread-out start
    spec = BehaviourSpec(1, 5, zbar=z0.mean(axis=1), sim_mean=np.full(300, 0.7))

    def friend_sim(X, z):
        S = 1 - np.abs(z[:, :, None] - z[:, None, :]) / 4
        return (X * S).sum(axis=(1, 2)) / np.maximum(X.sum(axis=(1, 2)), 1)

    out = {}
    for beta in (0.0, 2.5):
        _, z1 = simulate_coevolution(
            X0,
            z0,
            [-1.5, 1.0],
            [0, 0, 0],
            [0.0, 0.0, beta],
            0.0,
            6.0,
            model,
            bm,
            spec,
            rng,
            backend=backend,
        )
        out[beta] = friend_sim(X0, z1).mean()
    assert out[2.5] > out[0.0] + 0.05


def test_selection_increases_friend_similarity(backend):
    """Positive simZ makes actors choose similar alters: friends end up more similar."""
    rng, X0, z0, spec, model, bm = coev_case(B=300, n=20)
    z0 = rng.integers(1, 6, size=z0.shape)
    spec = BehaviourSpec(1, 5, zbar=z0.mean(axis=1), sim_mean=np.full(300, 0.7))

    def friend_sim(X, z):
        S = 1 - np.abs(z[:, :, None] - z[:, None, :]) / 4
        return (X * S).sum(axis=(1, 2)) / np.maximum(X.sum(axis=(1, 2)), 1)

    out = {}
    for beta in (0.0, 3.0):
        X1, _ = simulate_coevolution(
            X0,
            z0,
            [-1.5, 1.0],
            [0.0, 0.0, beta],
            [0, 0, 0],
            6.0,
            0.0,
            model,
            bm,
            spec,
            rng,
            backend=backend,
        )
        out[beta] = friend_sim(X1, z0).mean()
    assert out[3.0] > out[0.0] + 0.05


def test_backends_agree_on_coevolution_distribution(backend):
    if backend is NUMPY:
        pytest.skip("comparison target")
    rng, X0, z0, spec, model, bm = coev_case(B=1500, n=15)
    args = (X0, z0, [-1.5, 1.0], [0.2, 0.1, 1.0], [0.1, -0.3, 1.0], 2.0, 1.5, model, bm, spec)
    Xa, za = simulate_coevolution(*args, np.random.default_rng(5))
    Xb, zb = simulate_coevolution(*args, np.random.default_rng(6), backend=backend)
    Sa = np.column_stack(
        [Xa.sum(axis=(1, 2)), bm.statistics(X0, za, spec), bm.selection_statistics(Xa, z0, spec)]
    )
    Sb = np.column_stack(
        [Xb.sum(axis=(1, 2)), bm.statistics(X0, zb, spec), bm.selection_statistics(Xb, z0, spec)]
    )
    se = np.sqrt(Sa.var(axis=0, ddof=1) / 1500 + Sb.var(axis=0, ddof=1) / 1500)
    z = (Sa.mean(axis=0) - Sb.mean(axis=0)) / se
    assert np.all(np.abs(z) < 4.0), z
