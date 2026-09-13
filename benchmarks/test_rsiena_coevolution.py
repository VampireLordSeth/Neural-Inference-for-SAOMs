"""RSiena parity for co-evolution statistics (M3b gate, part 1): every per-period target
statistic RSiena computes for the s50 network x alcohol model is reproduced exactly.

Pins the conventions in ``saomsim.behaviour``: grand-mean centring, simMean over
period-start waves, behaviour targets with end behaviour / start network, selection
targets with end network / start behaviour, behaviour rate = sum |dz|.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from saomsim.behaviour import BehaviourModel, behaviour_rate_statistic, spec_from_data

HERE = Path(__file__).parent
FILES = ["s501.csv", "s502.csv", "s503.csv", "s50a.csv", "rsiena_coevolution_targets.json"]


@pytest.fixture(scope="module")
def data():
    missing = [f for f in FILES if not (HERE / f).exists()]
    if missing:
        pytest.skip(f"missing {missing}; run benchmarks/rsiena_coevolution.R")
    X = [np.loadtxt(HERE / f, delimiter=",", dtype=np.int8) for f in FILES[:3]]
    Z = np.loadtxt(HERE / "s50a.csv", delimiter=",")
    with open(HERE / "rsiena_coevolution_targets.json", encoding="utf-8") as fh:
        J = json.load(fh)
    return X, Z, J


def test_constants(data):
    _, Z, J = data
    spec = spec_from_data(Z, 1, 5)
    assert spec.zbar == pytest.approx(J["constants"]["grand_mean"], abs=1e-9)
    assert spec.sim_mean == pytest.approx(J["constants"]["simMean"], abs=1e-9)
    assert spec.z_range == J["constants"]["range"]


@pytest.mark.parametrize("period", [0, 1])
def test_targets_match_rsiena(data, period):
    X, Z, J = data
    spec = spec_from_data(Z, 1, 5)
    bm = BehaviourModel(
        effects=("linear", "quad", "avSim", "avAlt"), selection=("altZ", "egoZ", "simZ")
    )
    t = J["targets_by_period"][period]
    beh = bm.statistics(X[period][None], Z[:, period + 1][None], spec)[0]
    sel = bm.selection_statistics(X[period + 1][None], Z[:, period][None], spec)[0]
    ours = dict(zip(["alc:linear", "alc:quad", "alc:avSim", "alc:avAlt"], beh, strict=True))
    ours.update(dict(zip(["net:altX", "net:egoX", "net:simX"], sel, strict=True)))
    for k, v in ours.items():
        assert v == pytest.approx(t[k], abs=1e-9), k
    # RSiena's behaviour rate targets (printed by rsiena_coevolution.R; the JSON collapses
    # the two 'alc:Rate' rows into one key): 27 changes in period 1, 33 in period 2
    rate = behaviour_rate_statistic(Z[:, period][None], Z[:, period + 1][None])[0]
    assert rate == [27.0, 33.0][period]
