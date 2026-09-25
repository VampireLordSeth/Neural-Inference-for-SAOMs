"""Posterior predictive check of the s50 co-evolution fits on the behaviour statistics.

    python benchmarks/ppc_coev.py [--posterior data/npe_coev3_10m_c_posterior_s50.npz] [--B 2000]

Simulates the two periods s501 -> s502 -> s503 with alcohol from the observed start
under (a) the amortized posterior (one draw per simulation), (b) its mean, (c) RSiena's
method-of-moments point and (d) RSiena's maximum-likelihood point, and compares the
observed behaviour statistics per period — the MoM targets (linear, quad, avAlt), the
change count, the alcohol mean/sd and the friend-similarity sum — with each predictive
distribution. Written 2026-09-17 to adjudicate the influence/quad pair, on which the
three estimators disagree (docs/M3_RESULTS.md). Note that MoM matches the target
statistics *by construction*; the informative comparisons are on the other statistics
and between the amortized posterior and ML.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saomsim.behaviour import (
    BehaviourSpec,  # noqa: E402
    behaviour_rate_statistic,
    simulate_coevolution,
    spec_from_data,
)
from saomsim.population_coev import (  # noqa: E402
    BEH_EFFECTS,
    NET_EFFECTS,
    SEL_EFFECTS,
    beh_model,
    coev_theta_names,
    net_model,
)

HERE = Path(__file__).parent
NAMES = coev_theta_names(3)


def rsiena_point(path, beh="alc"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lab = {
        "rate_net_1": "net:rate_1",
        "rate_net_2": "net:rate_2",
        "rate_beh_1": f"{beh}:rate_1",
        "rate_beh_2": f"{beh}:rate_2",
        "egoZ": "net:egoX",
        "altZ": "net:altX",
        "simZ": "net:simX",
    }
    lab.update({e: f"net:{e}" for e in NET_EFFECTS})
    lab.update({e: f"{beh}:{e}" for e in BEH_EFFECTS})
    return np.array([d["estimate"][lab[nm]] for nm in NAMES])


def beh_stats(X_start, z_start, z_end, spec, bmodel):
    """Per-period behaviour statistics (B, 7): the three MoM targets, the change count,
    end-of-period mean and sd of z, and the friend-similarity sum at the end."""
    t = bmodel.statistics(X_start, z_end, spec)  # linear, quad, avAlt targets
    ch = behaviour_rate_statistic(z_start, z_end)
    z = np.asarray(z_end, dtype=float)
    rng_ = spec.z_max - spec.z_min
    S = 1.0 - np.abs(z[:, :, None] - z[:, None, :]) / rng_
    sim = (np.asarray(X_start, dtype=float) * S).sum(axis=(1, 2))
    return np.column_stack([t, ch, z.mean(1), z.std(1, ddof=1), sim])


STAT_NAMES = ["linear", "quad", "avAlt", "changes", "z_mean", "z_sd", "friend_sim"]


def simulate(theta, X0, z0, spec, rng, backend, B):
    """theta (B, 14) -> statistics per period, list of (B, 7)."""
    model, bmodel = net_model(), beh_model()
    R = 2
    rn, rb, eff = theta[:, :R], theta[:, R : 2 * R], theta[:, 2 * R :]
    k1, k2 = len(NET_EFFECTS), len(NET_EFFECTS) + len(SEL_EFFECTS)
    X = np.broadcast_to(X0, (B,) + X0.shape).copy()
    z = np.broadcast_to(z0, (B,) + z0.shape).copy()
    out = []
    for w in range(R):
        Xn, zn = simulate_coevolution(
            X,
            z,
            eff[:, :k1],
            eff[:, k1:k2],
            eff[:, k2:],
            rn[:, w],
            rb[:, w],
            model,
            bmodel,
            spec,
            rng,
            backend=backend,
        )
        out.append(beh_stats(X, z, zn, spec, bmodel))
        X, z = Xn, zn
    return out


# panel -> (network CSVs, behaviour CSV, RSiena MoM json, ML json, behaviour label in RSiena)
REAL = {
    "s50": (
        [HERE / f"s50{w}.csv" for w in (1, 2, 3)],
        HERE / "s50a.csv",
        HERE / "rsiena_coevolution_estimate.json",
        HERE / "rsiena_coevolution_maxlike.json",
        "alc",
    ),
    "glasgow": (
        [HERE / "glasgow" / f"glasgow_net{w}.csv" for w in (1, 2, 3)],
        HERE / "glasgow" / "glasgow_alcohol.csv",
        HERE / "glasgow" / "rsiena_coevolution.json",
        HERE / "glasgow" / "rsiena_coevolution_maxlike.json",
        "alcB",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_coev3_10m_c_posterior_s50.npz")
    ap.add_argument("--real", default="s50", choices=list(REAL))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    net_paths, z_path, mom_path, ml_path, beh_lab = REAL[a.real]
    Xs = [np.loadtxt(p, delimiter=",", dtype=np.int8) for p in net_paths]
    Z = np.loadtxt(z_path, delimiter=",")
    sp = spec_from_data(Z, 1, 5)
    spec = BehaviourSpec(1, 5, np.full(a.B, sp.zbar), np.full(a.B, sp.sim_mean))
    bmodel = beh_model()
    obs = [
        beh_stats(Xs[w][None], Z[:, w][None], Z[:, w + 1][None], spec_1(sp), bmodel)[0]
        for w in range(2)
    ]

    post = np.load(a.posterior)
    assert list(post["names"]) == NAMES
    smp = post["samples"]
    points = {
        "amortized draws": smp[rng.choice(len(smp), a.B, replace=False)],
        "amortized mean": np.broadcast_to(smp.mean(0), (a.B, 14)),
        "RSiena MoM": np.broadcast_to(rsiena_point(mom_path, beh_lab), (a.B, 14)),
    }
    if ml_path.exists():
        points["RSiena ML"] = np.broadcast_to(rsiena_point(ml_path, beh_lab), (a.B, 14))

    print(
        f"posterior predictive check, {a.real} x behaviour, n = {Xs[0].shape[0]}, "
        f"{a.B} simulations per point\n"
    )
    for label, theta in points.items():
        stats = simulate(np.ascontiguousarray(theta), Xs[0], Z[:, 0], spec, rng, a.backend, a.B)
        print(f"== {label}")
        print(f"{'statistic':<14}" + "".join(f"{h:>22}" for h in ("period 1", "period 2")))
        flags = []
        for k, nm in enumerate(STAT_NAMES):
            cells = []
            for w in range(2):
                v, o = stats[w][:, k], obs[w][k]
                F = (v < o).mean()
                cells.append(f"{o:8.2f} | {v.mean():7.2f} ({F:.2f})")
                if F < 0.025 or F > 0.975:
                    flags.append(f"{nm}/p{w + 1}")
            print(f"{nm:<14}" + "".join(f"{c:>22}" for c in cells))
        print(f"  obs | pp mean (F(obs)); outside central 95%: {', '.join(flags) or 'none'}\n")


def spec_1(sp):
    return BehaviourSpec(1, 5, np.array([sp.zbar]), np.array([sp.sim_mean]))


if __name__ == "__main__":
    main()
