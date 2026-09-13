"""The M2 case: population prior (docs/PRIORS_M2.md) and the s50 data as a real-start test.

from benchmarks.m2 import m2_prior, s50_as_m2
"""

import numpy as np

from benchmarks.s50 import PRIOR_RANGES, load_s50
from saomsim.population import m2_model, rate_names, real_data_summary, sample_covariates
from saomsim.prior import BoxPrior

# same ranges as M1, keyed by the M2 covariate names
M2_RANGES = {k.replace("(alc)", "(v)").replace("(smk)", "(g)"): v for k, v in PRIOR_RANGES.items()}


def m2_prior(waves: int = 2) -> BoxPrior:
    """Box prior with one rate range per period (all U(1, 12)) and the M2 effect ranges."""
    probe = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    ranges = dict(M2_RANGES)
    rate = ranges.pop("rate")
    names = rate_names(waves) + probe.labels
    low = [rate[0]] * (waves - 1) + [ranges[lbl][0] for lbl in probe.labels]
    high = [rate[1]] * (waves - 1) + [ranges[lbl][1] for lbl in probe.labels]
    return BoxPrior(tuple(names), np.array(low), np.array(high))


def s50_as_m2(waves: int = 2):
    """(x0, later, v, g, summary_row, model): s50 with alc -> v (centred), smk -> g (0-based).
    ``later`` is s502 for two waves, or [s502, s503] for three."""
    from pathlib import Path

    x0, x1, _ = load_s50()
    later = x1
    if waves == 3:
        x2 = np.loadtxt(Path(__file__).parent / "s503.csv", delimiter=",", dtype=np.int8)
        later = [x1, x2]
    elif waves != 2:
        raise ValueError("s50 has three waves")
    cov = np.genfromtxt(Path(__file__).parent / "s50_covariates.csv", delimiter=",", names=True)
    v = cov["alc_centred"]
    g = cov["smk_raw"] - cov["smk_raw"].min()
    S, model = real_data_summary(x0, later, v, g)
    return x0, later, v, g, S, model
