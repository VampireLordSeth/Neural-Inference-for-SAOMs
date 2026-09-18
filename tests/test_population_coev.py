"""Tests for the co-evolution population (saomsim.population_coev)."""

import numpy as np
import pytest

from saomsim.behaviour import BehaviourSpec
from saomsim.population import N_MAX, unpack
from saomsim.population_coev import (
    BEH_EFFECTS,
    NET_EFFECTS,
    SEL_EFFECTS,
    CoevTrainingSet,
    beh_model,
    coev_prior,
    coev_summaries,
    coev_summary_names,
    coev_theta_names,
    generate_coev,
    net_model,
    real_coev_summary,
    transform_coev,
)


def test_theta_names_and_prior():
    assert coev_theta_names(2) == ["rate_net", "rate_beh"] + NET_EFFECTS + SEL_EFFECTS + BEH_EFFECTS
    assert coev_theta_names(3)[:4] == ["rate_net_1", "rate_net_2", "rate_beh_1", "rate_beh_2"]
    p = coev_prior(3)
    assert p.dim == 14 and p.low[2] == 0.3 and p.high[p.names.index("simZ")] == 4.0


def test_generate_coev_layout_storage_and_summary_consistency(tmp_path, backend):
    prior = coev_prior(3)
    rng = np.random.default_rng(1)
    ts = generate_coev(prior, 40, rng, waves=3, n_range=(20, 26), chunk=10, backend=backend)
    names = coev_summary_names(3)
    assert ts.theta.shape == (40, 14) and ts.summary.shape == (40, len(names)) == (40, 62)
    assert ts.X.shape == (40, 3, N_MAX, N_MAX // 8) and ts.z.shape == (40, 3, N_MAX)
    assert set(ts.z_max) <= {3, 4, 5}
    # every behaviour value on its scale; padding is -1
    for i in range(40):
        n = int(ts.n[i])
        zz = ts.z[i, :, :n]
        assert zz.min() >= 1 and zz.max() <= ts.z_max[i]
        assert np.all(ts.z[i, :, n:] == -1)
    # stored waves reproduce the stored summaries, with the stored constants
    idx = np.arange(10)  # the first chunk: one n and one z_max
    n = int(ts.n[0])
    Xs = [unpack(ts.X[idx, w], n) for w in range(3)]
    zs = [ts.z[idx, w, :n].astype(float) for w in range(3)]
    zb = ts.summary[idx, names.index("zbar")]
    sm = ts.summary[idx, names.index("sim_mean")]
    spec = BehaviourSpec(1, int(ts.z_max[idx[0]]), zb, sm)
    S = coev_summaries(Xs, zs, spec, net_model(), beh_model())
    assert np.allclose(S, ts.summary[idx])
    assert np.all(ts.summary[idx, names.index("z1_changes")] == np.abs(zs[1] - zs[0]).sum(axis=1))
    T = transform_coev(ts.summary, names, net_model())
    assert np.all(np.isfinite(T)) and np.allclose(T[:, names.index("n")], np.log(ts.n))
    ts.save(tmp_path / "c.npz")
    back = CoevTrainingSet.load(tmp_path / "c.npz")
    assert np.array_equal(back.X, ts.X) and np.array_equal(back.z, ts.z) and back.meta["waves"] == 3
    with pytest.raises(ValueError):
        generate_coev(coev_prior(2), 4, rng, waves=3, chunk=4)


def test_generate_coev_reproducible():
    prior = coev_prior(2)
    a = generate_coev(prior, 12, np.random.default_rng(3), chunk=6, keep_networks=False)
    b = generate_coev(prior, 12, np.random.default_rng(3), chunk=6, keep_networks=False)
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.summary, b.summary)


def test_generate_coev_sparse_regime_and_wide_rates():
    """start="sparse" at n > N_MAX (summary-only) gives sparser starts than the m2
    regime and records the regime; rate_net can be widened and lands in the meta."""
    prior = coev_prior(3, rate_net=(1.0, 20.0))
    assert prior.high[0] == 20.0 and prior.high[1] == 20.0 and prior.low[2] == 0.3
    names = coev_summary_names(3)
    k = names.index("x0_tie_fraction")
    kw = dict(waves=3, n_range=(120, 120), chunk=64, keep_networks=False)
    sparse = generate_coev(prior, 128, np.random.default_rng(5), start="sparse", **kw)
    dense = generate_coev(prior, 128, np.random.default_rng(5), start="m2", **kw)
    assert sparse.meta["start_regime"] == "sparse" and dense.meta["start_regime"] == "m2"
    assert sparse.meta["prior_high"][0] == 20.0
    assert np.median(sparse.summary[:, k]) < 0.75 * np.median(dense.summary[:, k])
    with pytest.raises(ValueError):
        generate_coev(prior, 8, np.random.default_rng(0), waves=3, n_range=(120, 120), chunk=8)


def test_homophilous_starts_carry_behaviour_alignment():
    """The homophilous regime burns the start in under the selection effects with z0
    frozen, so x0_simZ spreads out with the drawn simZ; the sparse regime's starts are
    built independently of behaviour."""
    prior = coev_prior(3)
    names = coev_summary_names(3)
    k = names.index("x0_simZ")
    kw = dict(waves=3, n_range=(40, 40), chunk=64, keep_networks=False)
    hom = generate_coev(prior, 256, np.random.default_rng(7), start="homophilous", **kw)
    spa = generate_coev(prior, 256, np.random.default_rng(7), start="sparse", **kw)
    assert hom.meta["start_regime"] == "homophilous"
    # sparse starts scatter x0_simZ around zero by chance alone; homophilous ones spread
    # it with the drawn selection effects (2.4x at n=40, 256 panels)
    assert hom.summary[:, k].std() > 1.8 * spa.summary[:, k].std()
    with pytest.raises(ValueError):
        generate_coev(prior, 8, np.random.default_rng(0), start="nope", **kw)


def test_real_coev_summary_uses_rsiena_constants():
    Xs = [np.loadtxt(f"benchmarks/s50{w}.csv", delimiter=",", dtype=np.int8) for w in (1, 2, 3)]
    Z = np.loadtxt("benchmarks/s50a.csv", delimiter=",")
    S, spec = real_coev_summary(Xs, [Z[:, w] for w in range(3)], 1, 5)
    names = coev_summary_names(3)
    assert S.shape == (1, 62)
    assert S[0, names.index("zbar")] == pytest.approx(3.11333, abs=1e-5)
    assert S[0, names.index("sim_mean")] == pytest.approx(0.67439, abs=1e-5)
    assert S[0, names.index("x1_changes")] == 115 and S[0, names.index("z1_changes")] == 27
    assert S[0, names.index("z2_changes")] == 33
