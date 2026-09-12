"""The s50 case: data, model and prior for the first amortized estimator.

Start network is the observed wave 1 (``s501``), fixed. The model is the one
validated against RSiena in this directory. The prior box is documented and
justified in docs/PRIORS.md; change it there first, then here.

    from benchmarks.s50 import load_s50, s50_prior
    x0, x1, model = load_s50()
    prior = s50_prior(model)
"""

from pathlib import Path

import numpy as np

from saomsim import Model
from saomsim.prior import BoxPrior

HERE = Path(__file__).parent

EFFECTS = [
    "density",
    "recip",
    "transTrip",
    "cycle3",
    ("altX", "alc"),
    ("egoX", "alc"),
    ("sameX", "smk"),
]

PRIOR_RANGES = {
    "rate": (1.0, 12.0),
    "density": (-4.0, 0.0),
    "recip": (-1.0, 4.0),
    "transTrip": (-0.5, 1.5),
    "cycle3": (-1.5, 0.5),
    "altX(alc)": (-1.0, 1.0),
    "egoX(alc)": (-1.0, 1.0),
    "sameX(smk)": (-1.0, 2.0),
}


def load_s50():
    x0 = np.loadtxt(HERE / "s501.csv", delimiter=",", dtype=np.int8)
    x1 = np.loadtxt(HERE / "s502.csv", delimiter=",", dtype=np.int8)
    cov = np.genfromtxt(HERE / "s50_covariates.csv", delimiter=",", names=True)
    model = Model(EFFECTS, {"alc": cov["alc_centred"], "smk": cov["smk_raw"]})
    return x0, x1, model


def s50_prior(model: Model) -> BoxPrior:
    ranges = dict(PRIOR_RANGES)
    rate = ranges.pop("rate")
    return BoxPrior.for_model(model, rate=rate, **ranges)
