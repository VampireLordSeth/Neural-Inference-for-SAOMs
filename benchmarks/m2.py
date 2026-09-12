"""The M2 case: population prior (docs/PRIORS_M2.md) and the s50 data as a real-start test.

from benchmarks.m2 import m2_prior, s50_as_m2
"""

import numpy as np

from benchmarks.s50 import PRIOR_RANGES, load_s50
from saomsim.population import m2_model, real_data_summary, sample_covariates
from saomsim.prior import BoxPrior

# same ranges as M1, keyed by the M2 covariate names
M2_RANGES = {k.replace("(alc)", "(v)").replace("(smk)", "(g)"): v for k, v in PRIOR_RANGES.items()}


def m2_prior() -> BoxPrior:
    probe = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    ranges = dict(M2_RANGES)
    rate = ranges.pop("rate")
    return BoxPrior.for_model(probe, rate=rate, **ranges)


def s50_as_m2():
    """(x0, x1, v, g, summary_row, model): s50 with alc -> v (centred), smk -> g (0-based)."""
    x0, x1, _ = load_s50()
    from pathlib import Path

    cov = np.genfromtxt(Path(__file__).parent / "s50_covariates.csv", delimiter=",", names=True)
    v = cov["alc_centred"]
    g = cov["smk_raw"] - cov["smk_raw"].min()
    S, model = real_data_summary(x0, x1, v, g)
    return x0, x1, v, g, S, model
