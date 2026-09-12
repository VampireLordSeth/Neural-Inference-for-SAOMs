"""RSiena dynamics parity: distribution of simulated statistics at fixed theta.

``rsiena_simulate.R`` runs RSiena's simulator from s501 at a fixed parameter
vector 2000 times and stores the statistics of every simulated network in
``rsiena_sims.csv``. Here saomsim simulates from the same start with the same
parameters and the two samples are compared in mean and spread.

Matching *targets* (test_rsiena_parity.py) says the effect definitions agree.
Matching *these* says the ministep process agrees: rate/Poisson step count,
uniform actor choice, multinomial logit over the neighbourhood including the
no-change option, and the change statistics as used inside the objective.

Runs on every backend (conftest.py fixture). Reference run (RSiena 1.6.6 vs
saomsim numpy, 2000 vs 4000 sims): all |z| < 1.4,
sd ratios 0.96-1.00.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from saomsim import Model, moments, simulate_period

HERE = Path(__file__).parent
FILES = ["s501.csv", "s50_covariates.csv", "rsiena_sims.csv", "rsiena_sims.json"]
EFFECTS = {
    "density": "density",
    "recip": "recip",
    "transTrip": "transTrip",
    "cycle3": "cycle3",
    "altX(alc)": ("altX", "alc"),
    "egoX(alc)": ("egoX", "alc"),
    "sameX(smk)": ("sameX", "smk"),
}


def run_saomsim(B, backend, seed=0):
    x1 = np.loadtxt(HERE / "s501.csv", delimiter=",", dtype=np.int8)
    cov = np.genfromtxt(HERE / "s50_covariates.csv", delimiter=",", names=True)
    with open(HERE / "rsiena_sims.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    labels = meta["labels"]
    assert labels[0] == "Rate" and labels[1:] == list(EFFECTS)
    covariates = {"alc": cov["alc_centred"], "smk": cov["smk_raw"]}
    model = Model([EFFECTS[lbl] for lbl in labels[1:]], covariates)
    theta = np.array([meta["theta"][lbl] for lbl in labels])
    X0 = np.repeat(x1[None], B, axis=0)
    X1 = simulate_period(
        X0, theta[1:], theta[0], model, np.random.default_rng(seed), backend=backend
    )
    return labels, moments(X0, X1, model)


@pytest.fixture
def samples(backend):
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        pytest.skip(f"benchmark inputs missing ({missing}); run benchmarks/rsiena_simulate.R")
    R = np.loadtxt(HERE / "rsiena_sims.csv", delimiter=",", skiprows=1)
    labels, S = run_saomsim(B=1500, backend=backend)
    assert R.shape[1] == S.shape[1] == len(labels)
    return labels, R, S


def zscores(R, S):
    se = np.sqrt(R.var(axis=0, ddof=1) / len(R) + S.var(axis=0, ddof=1) / len(S))
    return (R.mean(axis=0) - S.mean(axis=0)) / se


def test_means_agree_with_rsiena(samples):
    labels, R, S = samples
    z = zscores(R, S)
    report = "\n".join(
        f"{lbl:<12} R {R[:, k].mean():9.2f}  saomsim {S[:, k].mean():9.2f}  z {z[k]:6.2f}"
        for k, lbl in enumerate(labels)
    )
    assert np.all(np.abs(z) < 4.0), report


def test_spreads_agree_with_rsiena(samples):
    labels, R, S = samples
    ratio = S.std(axis=0, ddof=1) / R.std(axis=0, ddof=1)
    report = "\n".join(f"{lbl:<12} sd ratio {ratio[k]:.3f}" for k, lbl in enumerate(labels))
    assert np.all((ratio > 0.8) & (ratio < 1.25)), report


def test_rate_statistic_is_poisson_like(samples):
    """Sanity on the one statistic with a known shape: tie changes are close to
    Poisson(n * rate) minus back-and-forth flips, so var/mean is just under 1."""
    labels, R, S = samples
    for X in (R, S):
        ratio = X[:, 0].var(ddof=1) / X[:, 0].mean()
        assert 0.6 < ratio < 1.2
