"""The M2 case: population prior (docs/PRIORS_M2.md) and the s50 data as a real-start test.

from benchmarks.m2 import m2_prior, s50_as_m2
"""

import numpy as np

from benchmarks.s50 import PRIOR_RANGES, load_s50
from saomsim.population import m2_model, rate_names, real_data_summary, sample_covariates
from saomsim.prior import BoxPrior

# same ranges as M1, keyed by the M2 covariate names
M2_RANGES = {k.replace("(alc)", "(v)").replace("(smk)", "(g)"): v for k, v in PRIOR_RANGES.items()}


def m2_prior(waves: int = 2, rate=None, box: str = "default") -> BoxPrior:
    """Box prior with one rate range per period (default U(1, 12)) and the M2 effect ranges.

    ``box="sparse"`` widens density, transTrip and cycle3 (``SPARSE_RANGES`` in
    ``saomsim.population_coev``, docs/PRIORS_M5.md)."""
    if box not in ("default", "sparse"):
        raise ValueError(f"unknown box {box!r}")
    probe = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    ranges = dict(M2_RANGES)
    if box == "sparse":
        from saomsim.population_coev import SPARSE_RANGES

        ranges.update(SPARSE_RANGES)
    rate = tuple(rate) if rate is not None else ranges.pop("rate")
    ranges.pop("rate", None)
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


def glasgow_as_m2(waves: int = 3):
    """Glasgow Teenage Friends and Lifestyle Study, 129 pupils present at all waves
    (benchmarks/glasgow/prepare_and_fit.R): v = alcohol wave 1 (centred), g = sex."""
    from pathlib import Path

    d = Path(__file__).parent / "glasgow"
    X = [np.loadtxt(d / f"glasgow_net{w}.csv", delimiter=",", dtype=np.int8) for w in (1, 2, 3)]
    cov = np.genfromtxt(d / "glasgow_covariates.csv", delimiter=",", names=True)
    v = cov["alc1_centred"]
    g = cov["sex"] - cov["sex"].min()
    later = X[1] if waves == 2 else X[1:]
    S, model = real_data_summary(X[0], later, v, g)
    return X[0], later, v, g, S, model
