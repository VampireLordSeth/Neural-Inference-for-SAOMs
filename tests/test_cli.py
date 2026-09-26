"""Tests for the installed command-line entry points (saomsim.cli).

These cover the paths a new user hits first: malformed CSVs, covariates of the wrong
length, a behaviour matrix whose wave count disagrees with the networks, and an estimator
whose training range does not cover the panel. Every one of those should produce a
sentence that says what is wrong, not a traceback from three frames down.

Nothing here loads a trained estimator, so the tests need neither torch nor sbi.
"""

import numpy as np
import pytest

from saomsim.cli import TRAINED_N, check_n, csv_views, load_waves, phi_names_for, report


def write_panel(tmp_path, n=12, waves=3, seed=0, behaviour=False):
    rng = np.random.default_rng(seed)
    paths = []
    for w in range(waves):
        X = (rng.random((n, n)) < 0.25).astype(int)
        np.fill_diagonal(X, 0)
        p = tmp_path / f"w{w + 1}.csv"
        np.savetxt(p, X, delimiter=",", fmt="%d")
        paths.append(str(p))
    v = tmp_path / "v.csv"
    g = tmp_path / "g.csv"
    np.savetxt(v, rng.normal(size=n), delimiter=",")
    np.savetxt(g, rng.integers(0, 2, size=n), delimiter=",", fmt="%d")
    out = [paths, str(v), str(g)]
    if behaviour:
        z = tmp_path / "z.csv"
        np.savetxt(z, rng.integers(1, 6, size=(n, waves)), delimiter=",", fmt="%d")
        out.append(str(z))
    return out


# ------------------------------------------------------------------ load_waves


def test_load_waves_reads_and_zeroes_the_diagonal(tmp_path):
    n = 9
    X = np.ones((n, n), dtype=int)
    p = tmp_path / "a.csv"
    np.savetxt(p, X, delimiter=",", fmt="%d")
    (Y,) = load_waves([str(p)])
    assert Y.shape == (n, n)
    assert np.diag(Y).sum() == 0
    assert Y.sum() == n * (n - 1)


def test_load_waves_rejects_a_non_square_matrix(tmp_path):
    p = tmp_path / "a.csv"
    np.savetxt(p, np.zeros((4, 5), dtype=int), delimiter=",", fmt="%d")
    with pytest.raises(SystemExit, match="adjacency matrix"):
        load_waves([str(p)])


def test_load_waves_rejects_values_that_are_not_zero_or_one(tmp_path):
    p = tmp_path / "a.csv"
    X = np.zeros((4, 4), dtype=int)
    X[0, 1] = 2
    np.savetxt(p, X, delimiter=",", fmt="%d")
    with pytest.raises(SystemExit, match="0 or 1"):
        load_waves([str(p)])


def test_load_waves_rejects_waves_of_different_sizes(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    np.savetxt(a, np.zeros((5, 5), dtype=int), delimiter=",", fmt="%d")
    np.savetxt(b, np.zeros((6, 6), dtype=int), delimiter=",", fmt="%d")
    with pytest.raises(SystemExit, match="5 x 5"):
        load_waves([str(a), str(b)])


# ------------------------------------------------------------------ csv_views


@pytest.mark.parametrize("waves", [2, 3, 5])
def test_csv_views_network_gives_one_view_per_period(tmp_path, waves):
    paths, v, g = write_panel(tmp_path, n=11, waves=waves)
    views, eff, n_rate, P, n = csv_views(paths, v, g)
    assert P == waves - 1
    assert views.shape == (P, 29)  # the two-wave network summary vector
    assert n_rate == 1 and n == 11
    assert len(eff) == 7


@pytest.mark.parametrize("waves", [2, 4])
def test_csv_views_coevolution_gives_one_view_per_period(tmp_path, waves):
    paths, v, g, z = write_panel(tmp_path, n=10, waves=waves, behaviour=True)
    views, eff, n_rate, P, n = csv_views(paths, beh_path=z)
    assert P == waves - 1
    assert views.shape == (P, 41)  # the two-wave co-evolution summary vector
    assert n_rate == 2 and n == 10
    assert len(eff) == 10


def test_csv_views_needs_covariates_for_the_network_model(tmp_path):
    paths, _, _ = write_panel(tmp_path)
    with pytest.raises(SystemExit, match="--v and --g"):
        csv_views(paths)


def test_csv_views_rejects_a_single_wave(tmp_path):
    paths, v, g = write_panel(tmp_path, waves=1)
    with pytest.raises(SystemExit, match="at least two waves"):
        csv_views(paths, v, g)


def test_csv_views_rejects_covariates_of_the_wrong_length(tmp_path):
    paths, v, g = write_panel(tmp_path, n=12)
    short = tmp_path / "short.csv"
    np.savetxt(short, np.zeros(7), delimiter=",")
    with pytest.raises(SystemExit, match="but the network has 12"):
        csv_views(paths, str(short), g)


def test_csv_views_rejects_a_behaviour_with_the_wrong_wave_count(tmp_path):
    paths, v, g, z = write_panel(tmp_path, n=10, waves=3, behaviour=True)
    two = tmp_path / "z2.csv"
    np.savetxt(two, np.ones((10, 2), dtype=int), delimiter=",", fmt="%d")
    with pytest.raises(SystemExit, match="2 waves but 3 networks"):
        csv_views(paths, beh_path=str(two))


def test_csv_views_rejects_a_behaviour_with_the_wrong_number_of_actors(tmp_path):
    paths, v, g, z = write_panel(tmp_path, n=10, waves=3, behaviour=True)
    wrong = tmp_path / "zbad.csv"
    np.savetxt(wrong, np.ones((6, 3), dtype=int), delimiter=",", fmt="%d")
    with pytest.raises(SystemExit, match="6 rows but the network has 10"):
        csv_views(paths, beh_path=str(wrong))


# ------------------------------------------------------------------ check_n


def test_check_n_accepts_a_size_inside_the_training_range(capsys):
    assert check_n("data/npe_m5c.pt", 50) is True
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("n", [25, 129])
def test_check_n_announces_an_extrapolation(capsys, n):
    assert check_n("data/npe_m5c.pt", n) is False
    out = capsys.readouterr().out
    assert "outside this estimator's training range [30, 100]" in out
    assert "extrapolation" in out


def test_check_n_says_so_when_the_range_is_unknown(capsys):
    assert check_n("somebody_elses_model.pt", 40) is None
    assert "not recorded" in capsys.readouterr().out


def test_every_recorded_range_is_ordered_and_plausible():
    for name, (lo, hi) in TRAINED_N.items():
        assert 0 < lo < hi <= 500, name


# ------------------------------------------------------------------ naming and report


def test_phi_names_group_rates_by_period_not_by_kind():
    assert phi_names_for(4, 1, ["a", "b"]) == ["rate_1", "rate_2", "rate_3", "a", "b"]
    assert phi_names_for(3, 2, ["a"]) == [
        "rate_net_1", "rate_beh_1", "rate_net_2", "rate_beh_2", "a",
    ]


def test_report_refuses_a_name_list_that_does_not_match_the_draws():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(50, 4))
    info = {"per_period": rng.normal(size=(2, 30, 3))}
    with pytest.raises(SystemExit, match="does not match the panel"):
        report(phi, info, ["only", "three", "names"], ["a", "b"], 1, 2)


def test_report_prints_a_row_per_parameter_and_returns_the_spread(capsys):
    rng = np.random.default_rng(1)
    P, n_rate, n_eff = 2, 1, 2
    per = rng.normal(size=(P, 400, n_rate + n_eff))
    phi = rng.normal(size=(500, P * n_rate + n_eff))
    names = phi_names_for(P + 1, n_rate, ["alpha", "beta"])
    spread = report(phi, {"per_period": per}, names, ["alpha", "beta"], n_rate, P)
    out = capsys.readouterr().out
    for nm in names:
        assert nm in out
    assert "period spread" in out
    assert len(spread) == n_eff
    assert (spread >= 0).all()


def test_report_flags_periods_that_disagree(capsys):
    """A shared effect the periods disagree about is a finding, and has to be visible."""
    P, n_rate = 2, 1
    per = np.zeros((P, 400, 2))
    rng = np.random.default_rng(2)
    per[0, :, 1] = rng.normal(0.0, 0.01, 400)
    per[1, :, 1] = rng.normal(5.0, 0.01, 400)  # same effect, wildly different periods
    per[:, :, 0] = rng.normal(5.0, 1.0, (P, 400))
    phi = rng.normal(size=(100, P + 1))
    spread = report(phi, {"per_period": per}, phi_names_for(P + 1, n_rate, ["shared"]),
                    ["shared"], n_rate, P)
    assert spread[0] > 2
    assert "periods disagree" in capsys.readouterr().out
