"""Tests for reading a W-wave panel with a two-wave estimator (saomsim.multiwave)."""

import numpy as np
import pytest

from saomsim.behaviour import BehaviourSpec
from saomsim.multiwave import (
    coev_period_views,
    m2_period_views,
    n_periods,
    period_view_index,
)
from saomsim.population import (
    m2_model,
    m2_summaries,
    m2_summary_names,
    sample_covariates,
)
from saomsim.population_coev import (
    beh_model,
    coev_summaries,
    coev_summary_names,
    net_model,
)


def _model(n=12, seed=0):
    return m2_model(sample_covariates(1, n, np.random.default_rng(seed)))


def _panel(B, n, waves, seed=0):
    rng = np.random.default_rng(seed)
    Xs = []
    for _ in range(waves):
        X = (rng.random((B, n, n)) < 0.2).astype(np.int8)
        for b in range(B):
            np.fill_diagonal(X[b], 0)
        Xs.append(X)
    return Xs


@pytest.mark.parametrize("waves", [2, 3, 4, 5])
def test_n_periods_is_waves_minus_one(waves):
    assert n_periods(m2_summary_names(_model(), waves)) == waves - 1
    assert n_periods(coev_summary_names(waves)) == waves - 1


def test_period_one_view_is_the_leading_block():
    """A two-wave vector is the head of a longer one, so period 1 re-slices to a prefix."""
    model = _model()
    three, two = m2_summary_names(model, 3), m2_summary_names(model, 2)
    assert period_view_index(three, two, 1) == list(range(len(two)))


def test_period_view_names_shift_by_one_wave():
    model = _model()
    three, two = m2_summary_names(model, 3), m2_summary_names(model, 2)
    got = [three[i] for i in period_view_index(three, two, 2)]
    assert got == [
        nm.replace("x0_", "x1_").replace("x1_", "x2_") if nm.startswith("x1_")
        else nm.replace("x0_", "x1_")
        for nm in two
    ]


def test_period_out_of_range_is_rejected():
    model = _model()
    three, two = m2_summary_names(model, 3), m2_summary_names(model, 2)
    for bad in (0, 3):
        with pytest.raises(ValueError):
            period_view_index(three, two, bad)


@pytest.mark.parametrize("waves", [3, 4])
def test_m2_period_view_equals_the_summary_of_that_wave_pair(waves):
    """The point of the re-slice: period w's view must be what you would get by
    summarising waves (w-1, w) as a two-wave panel from scratch."""
    n, B = 14, 3
    rng = np.random.default_rng(5)
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    Xs = _panel(B, n, waves, seed=1)

    S = m2_summaries(Xs[0], Xs[1:], model, covs)
    views = m2_period_views(S, m2_summary_names(model, waves), model)

    for w in range(1, waves):
        direct = m2_summaries(Xs[w - 1], Xs[w], model, covs)
        np.testing.assert_allclose(views[:, w - 1], direct, rtol=0, atol=0)


def test_m2_period_views_transform_matches_transforming_afterwards():
    from saomsim.population import transform_m2

    n, B, waves = 14, 2, 3
    rng = np.random.default_rng(6)
    covs = sample_covariates(B, n, rng)
    model = m2_model(covs)
    Xs = _panel(B, n, waves, seed=2)
    S = m2_summaries(Xs[0], Xs[1:], model, covs)
    names = m2_summary_names(model, waves)

    raw = m2_period_views(S, names, model, transform=False)
    hot = m2_period_views(S, names, model, transform=True)
    two = m2_summary_names(model, 2)
    for w in range(waves - 1):
        np.testing.assert_allclose(hot[:, w], transform_m2(raw[:, w], two, model), rtol=1e-6)


def test_coev_reslice_refuses_beyond_the_first_period():
    """The start block needs selection_statistics(X_w, z_w); a W-wave vector only carries
    the cross-lagged pair, and every name would still resolve, so it must refuse."""
    three, two = coev_summary_names(3), coev_summary_names(2)
    assert len(period_view_index(three, two, 1)) == len(two)
    with pytest.raises(KeyError, match="cannot be re-sliced"):
        period_view_index(three, two, 2)


@pytest.mark.parametrize("waves", [3, 4])
def test_coev_period_views_equal_the_pairwise_summaries(waves):
    n, B = 12, 2
    rng = np.random.default_rng(7)
    Xs = _panel(B, n, waves, seed=3)
    zs = [rng.integers(1, 6, size=(B, n)).astype(float) for _ in range(waves)]
    spec = BehaviourSpec(1, 5, np.full(B, 3.0), np.full(B, 0.5))
    model, bmodel = net_model(), beh_model()

    views = coev_period_views(Xs, zs, spec, model, bmodel)
    assert views.shape == (B, waves - 1, len(coev_summary_names(2)))
    for w in range(waves - 1):
        direct = coev_summaries(Xs[w : w + 2], zs[w : w + 2], spec, model, bmodel)
        np.testing.assert_allclose(views[:, w], direct, rtol=0, atol=0)


def test_coev_period_views_checks_wave_counts_match():
    Xs = _panel(1, 10, 3, seed=4)
    zs = [np.ones((1, 10))] * 2
    spec = BehaviourSpec(1, 5, np.full(1, 3.0), np.full(1, 0.5))
    with pytest.raises(ValueError, match="behaviour waves"):
        coev_period_views(Xs, zs, spec, net_model(), beh_model())


# --------------------------------------------------------------- product sampler
torch = pytest.importorskip("torch")
bmw = pytest.importorskip("benchmarks.multiwave")


def test_selectors_pick_the_right_rate_and_the_shared_effects():
    P, n_rate, n_eff = 3, 2, 4
    A = bmw.selectors(P, n_rate, n_eff)
    phi = np.arange(P * n_rate + n_eff, dtype=float)
    for w in range(P):
        theta = A[w] @ phi
        assert list(theta[:n_rate]) == list(phi[w * n_rate : (w + 1) * n_rate])
        assert list(theta[n_rate:]) == list(phi[P * n_rate :])


def test_gaussian_product_halves_the_variance_of_a_shared_effect():
    """Two periods that agree, with rate independent of the effects: the shared effect
    should gain precision by a factor of the number of periods, each rate should not."""
    rng = np.random.default_rng(0)
    sd_rate, sd_beta, N = 0.7, 1.3, 400_000
    per = np.stack(
        [
            np.column_stack(
                [
                    rng.normal(5.0, sd_rate, N),
                    rng.normal(-2.0, sd_beta, N),
                    rng.normal(1.0, sd_beta, N),
                ]
            )
            for _ in range(2)
        ]
    )
    mu, cov, _ = bmw.gaussian_product(per, n_rate=1)
    sd = np.sqrt(np.diag(cov))
    assert mu == pytest.approx([5.0, 5.0, -2.0, 1.0], abs=0.02)
    assert sd[0] == pytest.approx(sd_rate, rel=0.03)
    assert sd[1] == pytest.approx(sd_rate, rel=0.03)
    assert sd[2] == pytest.approx(sd_beta / np.sqrt(2), rel=0.03)
    assert sd[3] == pytest.approx(sd_beta / np.sqrt(2), rel=0.03)


def test_gaussian_product_is_dominated_by_the_sharper_period():
    """A period that says almost nothing should barely move the answer."""
    rng = np.random.default_rng(1)
    N = 400_000
    sharp = np.column_stack([rng.normal(4.0, 0.5, N), rng.normal(1.0, 0.1, N)])
    vague = np.column_stack([rng.normal(4.0, 0.5, N), rng.normal(-3.0, 10.0, N)])
    mu, cov, _ = bmw.gaussian_product(np.stack([sharp, vague]), n_rate=1)
    assert mu[2] == pytest.approx(1.0, abs=0.05)
    assert np.sqrt(cov[2, 2]) == pytest.approx(0.1, rel=0.05)


def test_split_rhat_is_one_for_iid_draws_and_large_for_offset_chains():
    rng = np.random.default_rng(2)
    good = rng.normal(size=(2000, 8, 3))
    assert np.nanmax(bmw.split_rhat(good)) < 1.05
    bad = good + np.arange(8)[None, :, None] * 3.0
    assert np.nanmin(bmw.split_rhat(bad)) > 1.5


def test_t_logpdf_matches_scipy():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    d, df = 3, 6.0
    B = rng.normal(size=(d, d))
    cov = B @ B.T + np.eye(d)
    mu = rng.normal(size=d)
    x = rng.normal(size=(50, d)) + mu
    want = sp.multivariate_t.logpdf(x, loc=mu, shape=cov, df=df)
    np.testing.assert_allclose(bmw.t_logpdf(x, mu, cov, df), want, rtol=1e-8)


def test_phi_names_group_rates_by_period():
    assert bmw.phi_names_for("net", 4, 1, ["a", "b"]) == ["rate_1", "rate_2", "rate_3", "a", "b"]
    assert bmw.phi_names_for("coev", 3, 2, ["a"]) == [
        "rate_net_1", "rate_beh_1", "rate_net_2", "rate_beh_2", "a",
    ]


def test_rsiena_compare_maps_names_and_reports_the_unmatched(tmp_path):
    import json

    p = tmp_path / "fit.json"
    p.write_text(
        json.dumps(
            {
                "estimate": {"net:rate_1": 5.0, "net:density": -2.0, "net:altX": 0.1,
                             "net:simX": 1.2, "alc:quad": -0.3},
                "se": {"net:rate_1": 0.5, "net:density": 0.2, "net:altX": 0.05,
                       "net:simX": 0.3, "alc:quad": 0.1},
            }
        ),
        encoding="utf-8",
    )
    got, missing = bmw.rsiena_compare(
        p, ["rate_1", "density", "altX(v)", "simZ", "quad", "recip"]
    )
    assert got["rate_1"] == (5.0, 0.5)
    assert got["altX(v)"] == (0.1, 0.05)   # the (v) suffix is ours, not RSiena's
    assert got["simZ"] == (1.2, 0.3)       # selection effects are simX/egoX/altX there
    assert got["quad"] == (-0.3, 0.1)      # behaviour effects take the behaviour prefix
    assert [n for n, _ in missing] == ["recip"]
