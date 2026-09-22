"""Tests for the M2 population (start networks, covariates, summaries, packing)."""

import numpy as np
import pytest

from saomsim.population import (
    M2_EFFECTS,
    N_MAX,
    M2TrainingSet,
    cap_outdegree,
    covariate_shape,
    generate_m2,
    m2_model,
    m2_summaries,
    m2_summary_names,
    real_data_summary,
    sample_covariates,
    sample_start_networks,
    transform_m2,
    unpack,
)
from saomsim.prior import BoxPrior, summaries


def prior_for(model):
    return BoxPrior.for_model(
        model,
        rate=(1, 12),
        **{
            "density": (-4, 0),
            "recip": (-1, 4),
            "transTrip": (-0.5, 1.5),
            "cycle3": (-1.5, 0.5),
            "altX(v)": (-1, 1),
            "egoX(v)": (-1, 1),
            "sameX(g)": (-1, 2),
        },
    )


def test_covariates_shapes_and_centring():
    rng = np.random.default_rng(0)
    covs = sample_covariates(200, 30, rng)
    assert covs["v"].shape == covs["g"].shape == (200, 30)
    assert np.allclose(covs["v"].mean(axis=1), 0, atol=1e-12)
    ncat = np.array([len(np.unique(r)) for r in covs["g"]])
    assert set(ncat) <= {2, 3, 4} and len(set(ncat)) > 1
    # roughly half Likert: those rows take at most 5 distinct values
    distinct_v = np.array([len(np.unique(r)) for r in covs["v"]])
    assert 0.3 < np.mean(distinct_v <= 5) < 0.7
    shape = covariate_shape(covs)
    assert shape.shape == (200, 3) and np.all(shape[:, 0] > 0) and np.all(shape[:, 2] >= 0)


def test_start_networks_mix_er_and_burnin(backend):
    rng = np.random.default_rng(1)
    B, n = 64, 25
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    X0, info = sample_start_networks(B, n, rng, prior_for(model), model, backend=backend)
    assert X0.shape == (B, n, n) and X0.dtype == np.int8
    assert np.all(X0[:, np.arange(n), np.arange(n)] == 0)
    assert 0.2 < info["burnin"].mean() < 0.8
    # burnt-in networks carry structure ER ones do not: more reciprocity on average
    recip = summaries(X0, X0, model)[:, 1 + model.labels.index("recip")]
    dens = X0.sum(axis=(1, 2))
    expected_er = dens**2 / (n * (n - 1))  # E[mutual pairs*2] under independence
    excess = (recip - expected_er)[info["burnin"]].mean() - (recip - expected_er)[
        ~info["burnin"]
    ].mean()
    assert excess > 0


def test_sparse_start_regime_mean_degree_and_cap():
    rng = np.random.default_rng(3)
    B, n = 400, 120
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    X0, info = sample_start_networks(B, n, rng, prior_for(model), model, start="sparse")
    assert X0.shape == (B, n, n) and X0.dtype == np.int8
    assert np.all(X0[:, np.arange(n), np.arange(n)] == 0)
    sparse, cap, burn = info["sparse"], info["cap"], info["burnin"]
    assert 0.35 < sparse.mean() < 0.65
    capped = cap > 0
    assert capped.sum() > 0 and not np.any(capped & ~sparse)
    # the cap holds for every actor of a capped start
    outdeg = X0.sum(axis=2)
    assert np.all(outdeg[capped] <= cap[capped, None])
    # uncapped, un-burnt sparse ER starts have mean degree in the drawn range;
    # the m2 regime at this n is several times denser
    er_sparse = sparse & ~burn & ~capped
    md = outdeg.mean(axis=1)
    assert np.all(md[er_sparse] < 12) and md[er_sparse].mean() > 1.5
    assert md[~sparse & ~burn].mean() > 2 * md[er_sparse].mean()
    # the m2 regime is unchanged by default
    rng2 = np.random.default_rng(0)
    model2 = m2_model(sample_covariates(8, 20, rng2))
    _, info2 = sample_start_networks(8, 20, rng2, prior_for(model2), model2)
    assert not info2["sparse"].any() and np.all(info2["cap"] < 0)


def test_cap_outdegree_keeps_exactly_cap_ties():
    rng = np.random.default_rng(4)
    X = (rng.random((3, 30, 30)) < 0.5).astype(np.int8)
    X[:, np.arange(30), np.arange(30)] = 0
    before = X.copy()
    cap = np.array([4, 7, 100])
    cap_outdegree(X, cap, rng)
    assert np.all(X <= before)  # only removals
    assert np.all(X[0].sum(1) == 4) and np.all(X[1].sum(1) == 7)
    assert np.array_equal(X[2], before[2])


def test_summaries_layout_and_transform():
    rng = np.random.default_rng(2)
    B, n = 10, 20
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    X0, _ = sample_start_networks(B, n, rng, prior_for(model), model)
    X1 = X0.copy()
    X1[:, 0, 1] = 1 - X1[:, 0, 1]
    S = m2_summaries(X0, X1, model, covs)
    names = m2_summary_names(model)
    assert S.shape == (B, len(names)) == (B, 29)
    assert np.all(S[:, names.index("n")] == n)
    assert np.all(S[:, names.index("x1_changes")] == 1)
    T = transform_m2(S, names, model)
    assert np.allclose(T[:, names.index("n")], np.log(n))
    assert np.allclose(T[:, names.index("x1_changes")], np.log1p(1))
    assert np.allclose(T[:, names.index("x0_tie_fraction")], S[:, names.index("x0_tie_fraction")])
    assert np.all(np.isfinite(T))


def test_generate_m2_and_roundtrip(tmp_path, backend):
    rng = np.random.default_rng(3)
    model = m2_model(sample_covariates(1, 20, rng))
    prior = prior_for(model)
    ts = generate_m2(prior, 50, rng, n_range=(20, 30), chunk=16, backend=backend)
    assert ts.theta.shape == (50, 8) and ts.summary.shape == (50, 29) and ts.n.shape == (50,)
    assert set(ts.n) <= set(range(20, 31)) and len(set(ts.n)) >= 2
    assert ts.X0.shape == ts.X1.shape == (50, N_MAX, N_MAX // 8)
    assert ts.v.shape == ts.g.shape == (50, N_MAX)
    # padding: covariates beyond n are 0 / -1; networks beyond n are empty
    for i in (0, 20, 49):
        n = int(ts.n[i])
        assert np.all(ts.g[i, n:] == -1) and np.all(ts.v[i, n:] == 0)
        X1 = unpack(ts.X1[i : i + 1], n)[0]
        assert X1.shape == (n, n)
        full = np.unpackbits(ts.X1[i : i + 1], axis=-1, count=N_MAX)[0]
        assert full[n:, :].sum() == 0 and full[:, n:].sum() == 0
    # the stored networks reproduce the stored summaries for one chunk
    chunk = ts.n == ts.n[0]
    idx = np.nonzero(chunk)[0][:16]
    n = int(ts.n[0])
    covs = {"v": ts.v[idx, :n].astype(float), "g": ts.g[idx, :n].astype(float)}
    S = m2_summaries(unpack(ts.X0[idx], n), unpack(ts.X1[idx], n), m2_model(covs), covs)
    assert np.allclose(S, ts.summary[idx])
    path = tmp_path / "m2.npz"
    ts.save(path)
    back = M2TrainingSet.load(path)
    assert np.array_equal(back.theta, ts.theta) and np.array_equal(back.X0, ts.X0)
    assert back.summary_names == ts.summary_names and back.meta == ts.meta
    with pytest.raises(ValueError):
        generate_m2(prior, 4, rng, n_range=(20, N_MAX + 1))


def test_generate_m2_reproducible():
    model = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    prior = prior_for(model)
    a = generate_m2(
        prior, 24, np.random.default_rng(7), n_range=(20, 25), chunk=8, keep_networks=False
    )
    b = generate_m2(
        prior, 24, np.random.default_rng(7), n_range=(20, 25), chunk=8, keep_networks=False
    )
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.summary, b.summary)
    assert a.X0 is None


def test_real_data_summary_matches_generated_layout():
    rng = np.random.default_rng(4)
    n = 12
    x0 = (rng.random((n, n)) < 0.2).astype(np.int8)
    x1 = (rng.random((n, n)) < 0.2).astype(np.int8)
    np.fill_diagonal(x0, 0)
    np.fill_diagonal(x1, 0)
    v = rng.normal(size=n)
    v -= v.mean()
    g = rng.integers(0, 3, size=n)
    S, model = real_data_summary(x0, x1, v, g)
    assert S.shape == (1, 29) and model.labels == m2_model(sample_covariates(1, n, rng)).labels
    assert S[0, 0] == n and [e if isinstance(e, str) else e for e in M2_EFFECTS]


def test_load_m2_summaries_concatenates_shards(tmp_path):
    from saomsim.population import load_m2_summaries

    model = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    prior = prior_for(model)
    for i in range(3):
        ts = generate_m2(
            prior, 10, np.random.default_rng(i), n_range=(20, 22), chunk=5, keep_networks=(i == 0)
        )
        ts.save(tmp_path / f"shard_{i}.npz")
    big = load_m2_summaries(str(tmp_path / "shard_*.npz"))
    assert big.theta.shape == (30, 8) and big.summary.shape == (30, 29) and big.n.shape == (30,)
    assert big.X0 is None and big.meta["N"] == 30 and len(big.meta["files"]) == 3
    one = load_m2_summaries(tmp_path / "shard_1.npz")
    assert np.array_equal(one.theta, big.theta[10:20])


def test_three_wave_generation_layout_and_storage(tmp_path):
    from saomsim.population import m2_summary_names, rate_names

    rng = np.random.default_rng(11)
    model = m2_model(sample_covariates(1, 20, rng))
    p2 = prior_for(model)
    assert rate_names(2) == ["rate"] and rate_names(3) == ["rate_1", "rate_2"]
    p3 = BoxPrior(
        ("rate_1", "rate_2") + p2.names[1:], np.r_[1, 1, p2.low[1:]], np.r_[12, 12, p2.high[1:]]
    )
    ts = generate_m2(p3, 20, rng, n_range=(20, 24), chunk=10, waves=3)
    names = m2_summary_names(model, 3)
    assert ts.theta.shape == (20, 9) and ts.summary.shape == (20, 42) and ts.summary_names == names
    assert ts.X0.shape == ts.X1.shape == (20, N_MAX, N_MAX // 8) and ts.Xw.shape == (
        20,
        1,
        N_MAX,
        N_MAX // 8,
    )
    assert ts.meta["waves"] == 3
    # the x2 block is the change from wave 2 to wave 3 plus the statistics of wave 3
    n = int(ts.n[0])
    idx = np.nonzero(ts.n == n)[0]
    covs = {"v": ts.v[idx, :n].astype(float), "g": ts.g[idx, :n].astype(float)}
    X0, X1, X2 = unpack(ts.X0[idx], n), unpack(ts.X1[idx], n), unpack(ts.Xw[idx, 0], n)
    S = m2_summaries(X0, [X1, X2], m2_model(covs), covs)
    assert np.allclose(S, ts.summary[idx])
    assert np.all(ts.summary[idx, names.index("x2_changes")] == (X1 != X2).sum(axis=(1, 2)))
    T = transform_m2(ts.summary, names, model)
    assert np.all(np.isfinite(T)) and T.shape == ts.summary.shape
    with pytest.raises(ValueError):
        generate_m2(p2, 4, rng, n_range=(20, 21), waves=3)  # prior lacks rate_1/rate_2
    with pytest.raises(ValueError):
        generate_m2(p2, 4, rng, n_range=(20, 21), waves=1)
    ts.save(tmp_path / "w3.npz")
    back = M2TrainingSet.load(tmp_path / "w3.npz")
    assert np.array_equal(back.Xw, ts.Xw) and back.meta["waves"] == 3
    # two-wave files load with Xw None
    ts2 = generate_m2(p2, 4, rng, n_range=(20, 21), chunk=4)
    ts2.save(tmp_path / "w2.npz")
    assert M2TrainingSet.load(tmp_path / "w2.npz").Xw is None


def test_survey_start_regime_every_start_sparse():
    rng = np.random.default_rng(5)
    B, n = 300, 60
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    X0, info = sample_start_networks(B, n, rng, prior_for(model), model, start="survey")
    assert info["sparse"].all()
    capped, burn = info["cap"] > 0, info["burnin"]
    assert 0.35 < capped.mean() < 0.65
    outdeg = X0.sum(axis=2)
    assert np.all(outdeg[capped] <= info["cap"][capped, None])
    # un-burnt, uncapped starts sit at the drawn mean degree, k ~ U(0.5, 6)
    er = ~burn & ~capped
    md = outdeg[er].mean(axis=1)
    assert md.min() < 1.5 and md.max() < 8 and 2 < md.mean() < 4.5
