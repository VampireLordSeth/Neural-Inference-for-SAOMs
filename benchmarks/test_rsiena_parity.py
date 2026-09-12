"""RSiena parity: saomsim target statistics on the s50 data.

``rsiena_compare.R`` exports RSiena's target statistics for s501 -> s502 to
``rsiena_targets.json`` together with the networks and covariates it used.
This test loads the same inputs into saomsim and requires exact agreement.
It needs no R; regenerate the JSON with the R script when RSiena changes.

Conventions this pins down (RSiena 1.6.6):

* rate target       = number of tie changes between waves
* effect targets    = actor-summed statistic of wave 2 ...
* ... except cycle3 = number of 3-cycles (actor sum / 3)
* recip             = actor sum, so each mutual dyad counts twice
* altX / egoX       = use the centred covariate (coCovar centres by default)
* sameX             = centring-invariant
"""

import json
from pathlib import Path

import numpy as np
import pytest

from saomsim import Model, rate_statistic, statistics

HERE = Path(__file__).parent
FILES = ["s501.csv", "s502.csv", "s50_covariates.csv", "rsiena_targets.json"]

# saomsim effect spec -> label used in the JSON
EFFECTS = [
    ("density", "density"),
    ("recip", "recip"),
    ("transTrip", "transTrip"),
    ("cycle3", "cycle3"),
    (("altX", "alc"), "altX(alc)"),
    (("egoX", "alc"), "egoX(alc)"),
    (("sameX", "smk"), "sameX(smk)"),
]


@pytest.fixture(scope="module")
def s50():
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        pytest.skip(f"benchmark inputs missing ({missing}); run benchmarks/rsiena_compare.R")
    x1 = np.loadtxt(HERE / "s501.csv", delimiter=",", dtype=np.int8)
    x2 = np.loadtxt(HERE / "s502.csv", delimiter=",", dtype=np.int8)
    cov = np.genfromtxt(HERE / "s50_covariates.csv", delimiter=",", names=True)
    with open(HERE / "rsiena_targets.json", encoding="utf-8") as fh:
        targets = json.load(fh)
    return x1, x2, cov, targets


def test_inputs_are_sane(s50):
    x1, x2, cov, targets = s50
    assert x1.shape == x2.shape == (50, 50)
    assert set(np.unique(x1)) <= {0, 1} and set(np.unique(x2)) <= {0, 1}
    assert np.all(np.diag(x1) == 0) and np.all(np.diag(x2) == 0)
    assert np.isclose(cov["alc_centred"].mean(), 0.0)
    assert np.isclose(cov["smk_centred"].mean(), 0.0)
    assert targets["hamming"] == int((x1 != x2).sum())


def test_rate_target_matches_rsiena(s50):
    x1, x2, _, targets = s50
    assert rate_statistic(x1[None], x2[None])[0] == targets["targets"]["Rate"]


@pytest.mark.parametrize(("effect", "label"), EFFECTS, ids=[lbl for _, lbl in EFFECTS])
def test_effect_target_matches_rsiena(s50, effect, label):
    x1, x2, cov, targets = s50
    model = Model([effect], {"alc": cov["alc_centred"], "smk": cov["smk_raw"]})
    ours = statistics(x2[None], model)[0, 0]
    assert ours == pytest.approx(targets["targets"][label], abs=1e-9), (
        f"{label}: saomsim {ours} vs RSiena {targets['targets'][label]}"
    )


def test_all_effects_jointly_match_rsiena(s50):
    x1, x2, cov, targets = s50
    model = Model([e for e, _ in EFFECTS], {"alc": cov["alc_centred"], "smk": cov["smk_raw"]})
    ours = statistics(x2[None], model)[0]
    expect = np.array([targets["targets"][lbl] for _, lbl in EFFECTS])
    np.testing.assert_allclose(ours, expect, atol=1e-9)


def test_sameX_is_centring_invariant(s50):
    _, x2, cov, targets = s50
    raw = statistics(x2[None], Model([("sameX", "smk")], {"smk": cov["smk_raw"]}))[0, 0]
    cen = statistics(x2[None], Model([("sameX", "smk")], {"smk": cov["smk_centred"]}))[0, 0]
    assert raw == cen == targets["targets"]["sameX(smk)"]


def test_altX_requires_centred_covariate(s50):
    """Documents why the centred vector is exported: the raw one does not match."""
    _, x2, cov, targets = s50
    raw = statistics(x2[None], Model([("altX", "alc")], {"alc": cov["alc_raw"]}))[0, 0]
    assert raw != pytest.approx(targets["targets"]["altX(alc)"])
