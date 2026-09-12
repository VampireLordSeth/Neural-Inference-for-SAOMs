"""Tests for the prior and training-set generation."""

import numpy as np
import pytest

from saomsim import Model, random_network, simulate_period
from saomsim.prior import (
    EXTRA_SUMMARIES,
    BoxPrior,
    TrainingSet,
    generate_training_set,
    prior_predictive_report,
    summaries,
    summary_names,
)


def small_case():
    rng = np.random.default_rng(0)
    n = 8
    grp = (np.arange(n) % 2).astype(float)
    model = Model(["density", "recip", ("sameX", "grp")], {"grp": grp})
    prior = BoxPrior.for_model(
        model, rate=(1, 4), **{"density": (-3, 0), "recip": (0, 2), "sameX(grp)": (-1, 1)}
    )
    x0 = random_network(1, n, 0.25, rng)[0]
    return model, prior, x0


def test_box_prior_construction_and_validation():
    model, prior, _ = small_case()
    assert prior.names == ("rate", "density", "recip", "sameX(grp)") and prior.dim == 4
    assert "U(-3, 0)" in prior.table()
    with pytest.raises(KeyError):
        BoxPrior.for_model(model, density=(-3, 0), recip=(0, 2))  # missing sameX
    with pytest.raises(KeyError):
        BoxPrior.for_model(
            model, density=(-3, 0), recip=(0, 2), **{"sameX(grp)": (0, 1), "cycle3": (0, 1)}
        )
    with pytest.raises(ValueError):
        BoxPrior(("a",), [1.0], [1.0])


def test_box_prior_sample_and_log_prob():
    _, prior, _ = small_case()
    rng = np.random.default_rng(1)
    th = prior.sample(5000, rng)
    assert th.shape == (5000, 4)
    assert np.all(th >= prior.low) and np.all(th <= prior.high)
    # roughly uniform: each coordinate's mean near the box centre
    assert np.allclose(th.mean(0), (prior.low + prior.high) / 2, atol=0.1)
    lp = prior.log_prob(th)
    assert np.allclose(lp, -np.log(prior.high - prior.low).sum())
    assert prior.log_prob([[0.0, -1.0, 1.0, 0.0]])[0] == -np.inf  # rate below 1
    assert prior.contains(th).all()


def test_summaries_layout_and_values():
    model, _, x0 = small_case()
    rng = np.random.default_rng(2)
    X0 = np.repeat(x0[None], 5, axis=0)
    X1 = simulate_period(X0, [-1.0, 0.5, 0.3], 2.0, model, rng)
    S = summaries(X0, X1, model)
    names = summary_names(model)
    assert S.shape == (5, len(names)) and names[:4] == ["changes", "density", "recip", "sameX(grp)"]
    assert names[4:] == list(EXTRA_SUMMARIES)
    n = x0.shape[0]
    assert np.allclose(S[:, names.index("tie_fraction")], X1.sum(axis=(1, 2)) / (n * (n - 1)))
    assert np.allclose(S[:, names.index("mutual_dyads")], S[:, names.index("recip")] / 2)
    assert summaries(X0, X1, model, extra=False).shape == (5, 4)


def test_generate_training_set_keeps_every_draw(backend):
    model, prior, x0 = small_case()
    rng = np.random.default_rng(3)
    ts = generate_training_set(
        prior, x0, model, 37, rng, chunk=10, backend=backend, keep_networks=True
    )
    assert ts.theta.shape == (37, 4) and ts.summary.shape == (37, len(summary_names(model)))
    assert ts.X1.shape == (37, 8, 8) and ts.X1.dtype == np.int8
    assert prior.contains(ts.theta).all()
    assert ts.meta["N"] == 37 and ts.meta["start"] == "fixed empirical"
    # the summaries are those of the stored networks
    X0 = np.repeat(x0[None], 37, axis=0)
    assert np.allclose(ts.summary, summaries(X0, ts.X1, model))


def test_generate_training_set_reproducible_and_theta_override():
    model, prior, x0 = small_case()
    a = generate_training_set(prior, x0, model, 20, np.random.default_rng(4), chunk=7)
    b = generate_training_set(prior, x0, model, 20, np.random.default_rng(4), chunk=7)
    assert np.array_equal(a.theta, b.theta) and np.array_equal(a.summary, b.summary)
    theta = np.tile([[2.0, -1.5, 1.0, 0.5]], (6, 1))
    c = generate_training_set(prior, x0, model, 6, np.random.default_rng(5), theta=theta)
    assert np.array_equal(c.theta, theta)
    with pytest.raises(ValueError):
        generate_training_set(prior, x0, model, 6, np.random.default_rng(5), theta=theta[:3])
    with pytest.raises(ValueError):
        generate_training_set(prior, x0[None], model, 6, np.random.default_rng(5))
    other = Model(["density"])
    with pytest.raises(ValueError):
        generate_training_set(prior, x0, other, 6, np.random.default_rng(5))


def test_training_set_roundtrip(tmp_path):
    model, prior, x0 = small_case()
    ts = generate_training_set(prior, x0, model, 12, np.random.default_rng(6), keep_networks=True)
    path = tmp_path / "ts.npz"
    ts.save(path)
    back = TrainingSet.load(path)
    assert np.array_equal(back.theta, ts.theta) and np.array_equal(back.summary, ts.summary)
    assert np.array_equal(back.X1, ts.X1)
    assert back.theta_names == ts.theta_names and back.summary_names == ts.summary_names
    assert back.meta == ts.meta
    ts2 = generate_training_set(prior, x0, model, 3, np.random.default_rng(7))
    ts2.save(path)
    assert TrainingSet.load(path).X1 is None


def test_prior_predictive_report_mentions_degeneracy():
    model, prior, x0 = small_case()
    ts = generate_training_set(prior, x0, model, 50, np.random.default_rng(8))
    obs = ts.summary[0]
    txt = prior_predictive_report(ts, obs)
    assert "F(obs)" in txt and "prior predictive:" in txt and "tie_fraction" in txt


def test_sort_by_rate_preserves_order_and_association():
    model, prior, x0 = small_case()
    theta = prior.sample(30, np.random.default_rng(9))
    ts = generate_training_set(
        prior, x0, model, 30, np.random.default_rng(10), theta=theta, chunk=8, keep_networks=True
    )
    assert np.array_equal(ts.theta, theta)  # original order kept
    X0 = np.repeat(x0[None], 30, axis=0)
    assert np.allclose(ts.summary, summaries(X0, ts.X1, model))  # rows still belong together


def test_transform_summaries():
    from saomsim.prior import transform_summaries

    model = Model(["density", ("altX", "v")], {"v": np.zeros(4)})
    names = summary_names(model)
    S = np.zeros((2, len(names)))
    S[0, names.index("density")] = np.e - 1
    S[0, names.index("altX(v)")] = -np.sinh(1.0)
    S[0, names.index("tie_fraction")] = 0.25
    T = transform_summaries(S, names, model)
    assert np.isclose(T[0, names.index("density")], 1.0)
    assert np.isclose(T[0, names.index("altX(v)")], -1.0)
    assert T[0, names.index("tie_fraction")] == 0.25
    assert np.all(T[1] == 0)
