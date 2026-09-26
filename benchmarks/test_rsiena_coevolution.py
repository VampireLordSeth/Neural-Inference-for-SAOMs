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

from conftest import requires_s50
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
@requires_s50
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


DYN_FILES = ["s501.csv", "s50a.csv", "rsiena_coevolution_sims.csv", "rsiena_coevolution_sims.json"]


@requires_s50
def test_joint_dynamics_match_rsiena(backend):
    """M3b gate, part 2: the joint network-behaviour simulator reproduces RSiena's
    simulated statistic distribution at fixed theta (s501 + alcohol wave 1 start).
    Reference run: 2000 RSiena vs 4000 saomsim draws, all |z| <= 2.0, sd ratios 0.99-1.03."""
    from saomsim import Model, rate_statistic, statistics
    from saomsim.behaviour import simulate_coevolution

    missing = [f for f in DYN_FILES if not (HERE / f).exists()]
    if missing:
        pytest.skip(f"missing {missing}; run benchmarks/rsiena_coevolution_simulate.R")
    x0 = np.loadtxt(HERE / "s501.csv", delimiter=",", dtype=np.int8)
    Z = np.loadtxt(HERE / "s50a.csv", delimiter=",")
    with open(HERE / "rsiena_coevolution_sims.json", encoding="utf-8") as fh:
        J = json.load(fh)
    R = np.loadtxt(HERE / "rsiena_coevolution_sims.csv", delimiter=",", skiprows=1)
    th = J["theta"]
    spec = spec_from_data(Z[:, :2], 1, 5)
    assert spec.sim_mean == pytest.approx(J["constants"]["simMean"], abs=1e-9)
    model = Model(["density", "recip", "transTrip", "cycle3"])
    bm = BehaviourModel(effects=("linear", "quad", "avAlt"), selection=("altZ", "egoZ", "simZ"))
    B = 1500
    X0 = np.repeat(x0[None], B, 0)
    z0 = np.repeat(Z[:, 0][None], B, 0).astype(int)
    X1, z1 = simulate_coevolution(
        X0,
        z0,
        [th["net:density"], th["net:recip"], th["net:transTrip"], th["net:cycle3"]],
        [th["net:altX"], th["net:egoX"], th["net:simX"]],
        [th["alc:linear"], th["alc:quad"], th["alc:avAlt"]],
        th["net:Rate"],
        th["alc:Rate"],
        model,
        bm,
        spec,
        np.random.default_rng(0),
        backend=backend,
    )
    S = np.column_stack(
        [
            rate_statistic(X0, X1),
            statistics(X1, model),
            bm.selection_statistics(X1, z0, spec),
            behaviour_rate_statistic(z0, z1),
            bm.statistics(X0, z1, spec),
        ]
    )
    assert S.shape[1] == R.shape[1] == len(J["labels"])
    se = np.sqrt(R.var(axis=0, ddof=1) / len(R) + S.var(axis=0, ddof=1) / B)
    z = (R.mean(axis=0) - S.mean(axis=0)) / se
    ratio = S.std(axis=0, ddof=1) / R.std(axis=0, ddof=1)
    report = "\n".join(
        f"{lbl:<14} z {z[k]:6.2f}  sd ratio {ratio[k]:.3f}" for k, lbl in enumerate(J["labels"])
    )
    assert np.all(np.abs(z) < 4.0), report
    assert np.all((ratio > 0.85) & (ratio < 1.18)), report
