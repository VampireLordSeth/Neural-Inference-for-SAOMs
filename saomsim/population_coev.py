"""M3b population: network-behaviour co-evolution panels (docs/PRIORS_M3b.md).

theta layout for W waves:
    net rates (W-1) | beh rates (W-1) | density, recip, transTrip, cycle3 |
    egoZ, altZ, simZ (selection) | linear, quad, avAlt (behaviour)

Start: X0 as in the M2 population (ER or burnt-in), z0 unimodal on {1..z_max}
with z_max in {3, 4, 5}. Centring constants are computed from the start
behaviour and passed to the estimator as inputs, so at inference the analyst
can supply RSiena's all-wave constants and the estimator conditions on them.

Every draw is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import DTYPE, OUT_DTYPE, zero_diagonal
from .behaviour import (
    BehaviourModel,
    BehaviourSpec,
    behaviour_rate_statistic,
    random_behaviour,
    simulate_coevolution,
    spec_from_data,
)
from .effects import Model
from .population import (
    BURNIN_STEPS_PER_ACTOR,
    CAP_EXCESS,
    ER_DENSITY,
    MEAN_DEGREE,
    SURVEY_MEAN_DEGREE,
    N_MAX,
    N_RANGE,
    _pack,
    cap_outdegree,
)
from .prior import EXTRA_SUMMARIES, BoxPrior, summaries, transform_summaries
from .simulate import simulate_period

NET_EFFECTS = ["density", "recip", "transTrip", "cycle3"]
SEL_EFFECTS = ["egoZ", "altZ", "simZ"]
BEH_EFFECTS = ["linear", "quad", "avAlt"]
Z_MAX_CHOICES = (3, 4, 5)

RANGES = {
    "rate_net": (1.0, 12.0),
    "rate_beh": (0.3, 6.0),
    "density": (-4.0, 0.0),
    "recip": (-1.0, 4.0),
    "transTrip": (-0.5, 1.5),
    "cycle3": (-1.5, 0.5),
    "egoZ": (-1.0, 1.0),
    "altZ": (-1.0, 1.0),
    "simZ": (-1.0, 4.0),
    "linear": (-1.5, 1.5),
    "quad": (-1.5, 0.5),
    "avAlt": (-1.0, 4.0),
}


def coev_theta_names(waves: int) -> list[str]:
    R = waves - 1
    rn = ["rate_net"] if R == 1 else [f"rate_net_{w}" for w in range(1, waves)]
    rb = ["rate_beh"] if R == 1 else [f"rate_beh_{w}" for w in range(1, waves)]
    return rn + rb + NET_EFFECTS + SEL_EFFECTS + BEH_EFFECTS


def coev_prior(waves: int = 2, rate_net=None) -> BoxPrior:
    """The M3b box; ``rate_net`` overrides the network-rate range (default U(1, 12))."""
    names = coev_theta_names(waves)
    ranges = dict(RANGES)
    if rate_net is not None:
        ranges["rate_net"] = tuple(rate_net)
    lo, hi = [], []
    for nm in names:
        key = (
            "rate_net"
            if nm.startswith("rate_net")
            else "rate_beh"
            if nm.startswith("rate_beh")
            else nm
        )
        lo.append(ranges[key][0])
        hi.append(ranges[key][1])
    return BoxPrior(tuple(names), np.array(lo), np.array(hi))


def net_model() -> Model:
    return Model(NET_EFFECTS)


def beh_model() -> BehaviourModel:
    return BehaviourModel(effects=BEH_EFFECTS, selection=SEL_EFFECTS)


# ---------------------------------------------------------------- summaries
Z_DESC = ("z_mean", "z_sd", "z_at_min", "z_at_max")


def z_descriptives(z: np.ndarray, z_min: int, z_max: int) -> np.ndarray:
    z = np.asarray(z, dtype=DTYPE)
    return np.column_stack(
        [
            z.mean(axis=1),
            z.std(axis=1, ddof=1),
            (z == z_min).mean(axis=1),
            (z == z_max).mean(axis=1),
        ]
    )


def coev_summary_names(waves: int) -> list[str]:
    net = NET_EFFECTS + list(EXTRA_SUMMARIES)
    names = ["n", "z_max", "zbar", "sim_mean"]
    names += (
        [f"x0_{s}" for s in net] + [f"x0_{s}" for s in SEL_EFFECTS] + [f"z0_{s}" for s in Z_DESC]
    )
    for w in range(1, waves):
        names += [f"x{w}_{s}" for s in ["changes"] + net] + [f"x{w}_{s}" for s in SEL_EFFECTS]
        names += (
            [f"z{w}_changes"] + [f"z{w}_{s}" for s in BEH_EFFECTS] + [f"z{w}_{s}" for s in Z_DESC]
        )
    return names


def coev_summaries(
    Xs: list, zs: list, spec: BehaviourSpec, model: Model, bmodel: BehaviourModel
) -> np.ndarray:
    """Xs, zs: lists of waves ((B,n,n) and (B,n)). Selection targets use end network / start z;
    behaviour targets use start network / end z (RSiena's cross-lagged convention)."""
    B, n, _ = Xs[0].shape
    zbar = np.broadcast_to(np.asarray(spec.zbar, dtype=DTYPE), (B,))
    sm = np.broadcast_to(np.asarray(spec.sim_mean, dtype=DTYPE), (B,))
    blocks = [
        np.full(B, n, dtype=DTYPE),
        np.full(B, spec.z_max, dtype=DTYPE),
        zbar,
        sm,
        summaries(Xs[0], Xs[0], model)[:, 1:],
        bmodel.selection_statistics(Xs[0], zs[0], spec),
        z_descriptives(zs[0], spec.z_min, spec.z_max),
    ]
    for w in range(1, len(Xs)):
        blocks += [
            summaries(Xs[w - 1], Xs[w], model),
            bmodel.selection_statistics(Xs[w], zs[w - 1], spec),
            behaviour_rate_statistic(zs[w - 1], zs[w]),
            bmodel.statistics(Xs[w - 1], zs[w], spec),
            z_descriptives(zs[w], spec.z_min, spec.z_max),
        ]
    return np.column_stack(blocks)


def transform_coev(S: np.ndarray, names: list, model: Model) -> np.ndarray:
    """log1p on network counts and the change counts, asinh on signed selection/behaviour sums."""
    out = np.array(S, dtype=DTYPE, copy=True)
    net = NET_EFFECTS + list(EXTRA_SUMMARIES)
    for nm in names:
        k = names.index(nm)
        if nm.startswith("x") and "_" in nm:
            stat = nm.split("_", 1)[1]
            if stat in net or stat == "changes":
                out[:, [k]] = transform_summaries(out[:, [k]], [stat], model)
            elif stat in SEL_EFFECTS:
                out[:, k] = np.arcsinh(out[:, k])
        elif nm.endswith("_changes") or nm.endswith("_quad"):
            out[:, k] = np.log1p(np.clip(out[:, k], 0, None))
        elif nm.endswith("_linear") or nm.endswith("_avAlt"):
            out[:, k] = np.arcsinh(out[:, k])
        elif nm == "n":
            out[:, k] = np.log(out[:, k])
    return out


# ------------------------------------------------------------- training set
@dataclass
class CoevTrainingSet:
    theta: np.ndarray
    summary: np.ndarray
    n: np.ndarray
    z_max: np.ndarray
    theta_names: list
    summary_names: list
    X: np.ndarray | None = None  # (N, W, N_MAX, N_MAX/8) packed
    z: np.ndarray | None = None  # (N, W, N_MAX) int8, -1 padded
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            theta=self.theta,
            summary=self.summary,
            n=self.n,
            z_max=self.z_max,
            theta_names=np.array(self.theta_names),
            summary_names=np.array(self.summary_names),
            X=self.X if self.X is not None else np.zeros((0,), dtype=np.uint8),
            z=self.z if self.z is not None else np.zeros((0,), dtype=np.int8),
            meta=np.array(repr(self.meta)),
        )

    @classmethod
    def load(cls, path) -> CoevTrainingSet:
        import ast

        with np.load(path, allow_pickle=False) as f:
            return cls(
                theta=f["theta"],
                summary=f["summary"],
                n=f["n"],
                z_max=f["z_max"],
                theta_names=[str(s) for s in f["theta_names"]],
                summary_names=[str(s) for s in f["summary_names"]],
                X=f["X"] if f["X"].size else None,
                z=f["z"] if f["z"].size else None,
                meta=ast.literal_eval(str(f["meta"])),
            )


START_REGIMES = ("m2", "sparse", "homophilous", "survey")


def _start_networks(
    B, n, rng, prior, model, backend, start: str = "m2", *, z0=None, spec=None, bmodel=None
):
    """As ``population.sample_start_networks`` (regimes "m2" and "sparse"), but the burn-in
    runs under the structural effects only, drawn from the co-evolution prior.

    ``start="homophilous"`` (docs/PRIORS_M4.md, step 3): the sparse density mixture, and
    the burnt-in half evolves under the structural *and selection* effects with the
    behaviour ``z0`` frozen (rate_beh = 0), so start networks carry the alcohol homophily
    real friendship networks have at wave 1 — which the other regimes lack entirely.
    """
    if start not in START_REGIMES:
        raise ValueError(f"unknown start regime {start!r}")
    d = rng.uniform(*ER_DENSITY, size=B)
    cap = np.full(B, -1)
    if start in ("sparse", "homophilous", "survey"):
        # "survey" (docs/PRIORS_M5.md): every start sparse, k ~ U(0.5, 6)
        sparse = rng.random(B) < 0.5 if start != "survey" else np.ones(B, dtype=bool)
        k = rng.uniform(*(MEAN_DEGREE if start != "survey" else SURVEY_MEAN_DEGREE), size=B)
        d = np.where(sparse, k / (n - 1), d)
        capped = sparse & (rng.random(B) < 0.5)
        cap = np.where(
            capped, np.ceil(k) + rng.integers(CAP_EXCESS[0], CAP_EXCESS[1] + 1, size=B), -1
        )
    X = (rng.random((B, n, n)) < d[:, None, None]).astype(OUT_DTYPE)
    zero_diagonal(X)
    burn = rng.random(B) < 0.5
    # burn in at an independent draw of the effects; rates play no part
    theta0 = prior.sample(B, rng)
    idx = [prior.names.index(e) for e in NET_EFFECTS]
    if start == "homophilous":
        if z0 is None or spec is None or bmodel is None:
            raise ValueError("homophilous starts need z0, spec and bmodel")
        idx_sel = [prior.names.index(e) for e in SEL_EFFECTS]
        rate = np.where(burn, float(BURNIN_STEPS_PER_ACTOR), 0.0)
        X0, _ = simulate_coevolution(
            X,
            z0,
            theta0[:, idx],
            theta0[:, idx_sel],
            np.zeros((B, bmodel.K)),
            rate,
            np.zeros(B),
            model,
            bmodel,
            spec,
            rng,
            backend=backend,
        )
    else:
        steps = np.where(burn, BURNIN_STEPS_PER_ACTOR * n, 0)
        X0 = simulate_period(
            X, theta0[:, idx], model=model, rng=rng, n_steps=steps, backend=backend
        )
    if (cap > 0).any():
        X0 = np.ascontiguousarray(X0)
        cap_outdegree(X0, cap, rng)
    return X0


def generate_coev(
    prior: BoxPrior,
    N: int,
    rng: np.random.Generator,
    *,
    waves: int = 2,
    n_range=N_RANGE,
    chunk: int = 2048,
    backend="numpy",
    keep_networks: bool = True,
    progress: bool = False,
    start: str = "m2",
) -> CoevTrainingSet:
    if list(prior.names) != coev_theta_names(waves):
        raise ValueError("prior names must match coev_theta_names(waves)")
    if keep_networks and n_range[1] > N_MAX:
        raise ValueError(f"n_range exceeds N_MAX={N_MAX}; pass keep_networks=False for larger n")
    R = waves - 1
    model, bmodel = net_model(), beh_model()
    names = coev_summary_names(waves)
    n_lo, n_hi = n_range
    parts = {k: [] for k in ("theta", "S", "n", "zmax", "X", "z")}
    n_chunks = -(-N // chunk)
    for c in range(n_chunks):
        B = min(chunk, N - c * chunk)
        n = int(rng.integers(n_lo, n_hi + 1))
        z_max = int(rng.choice(Z_MAX_CHOICES))
        z0 = random_behaviour(B, n, 1, z_max, rng)
        # constants from the start behaviour, per chain
        zbar = z0.mean(axis=1)
        rng_ = z_max - 1
        S0 = 1.0 - np.abs(z0[:, :, None] - z0[:, None, :]) / rng_
        off = ~np.eye(n, dtype=bool)
        sim_mean = S0[:, off].mean(axis=1)
        spec = BehaviourSpec(1, z_max, zbar, sim_mean)
        X0 = _start_networks(
            B, n, rng, prior, model, backend, start=start, z0=z0, spec=spec, bmodel=bmodel
        )
        theta = prior.sample(B, rng)
        rn, rb = theta[:, :R], theta[:, R : 2 * R]
        eff = theta[:, 2 * R :]
        th_net = eff[:, : len(NET_EFFECTS)]
        th_sel = eff[:, len(NET_EFFECTS) : len(NET_EFFECTS) + len(SEL_EFFECTS)]
        th_beh = eff[:, len(NET_EFFECTS) + len(SEL_EFFECTS) :]
        Xs, zs = [X0], [z0]
        for w in range(R):
            Xw, zw = simulate_coevolution(
                Xs[-1],
                zs[-1],
                th_net,
                th_sel,
                th_beh,
                rn[:, w],
                rb[:, w],
                model,
                bmodel,
                spec,
                rng,
                backend=backend,
            )
            Xs.append(Xw)
            zs.append(zw)
        parts["theta"].append(theta)
        parts["S"].append(coev_summaries(Xs, zs, spec, model, bmodel))
        parts["n"].append(np.full(B, n, dtype=np.int64))
        parts["zmax"].append(np.full(B, z_max, dtype=np.int64))
        if keep_networks:
            parts["X"].append(np.stack([_pack(Xw, N_MAX) for Xw in Xs], axis=1))
            zp = np.full((B, waves, N_MAX), -1, dtype=np.int8)
            for w, zw in enumerate(zs):
                zp[:, w, :n] = zw
            parts["z"].append(zp)
        if progress and (c % max(1, n_chunks // 20) == 0 or c == n_chunks - 1):
            print(f"  chunk {c + 1}/{n_chunks} (n={n}, z_max={z_max})", flush=True)
    cat = np.concatenate
    return CoevTrainingSet(
        theta=cat(parts["theta"]),
        summary=cat(parts["S"]),
        n=cat(parts["n"]),
        z_max=cat(parts["zmax"]),
        theta_names=list(prior.names),
        summary_names=names,
        X=cat(parts["X"]) if keep_networks else None,
        z=cat(parts["z"]) if keep_networks else None,
        meta={
            "N": int(N),
            "waves": int(waves),
            "n_range": list(n_range),
            "z_max_choices": list(Z_MAX_CHOICES),
            "prior_low": prior.low.tolist(),
            "prior_high": prior.high.tolist(),
            "start_regime": start,
            "constants": "zbar, sim_mean from the start behaviour; passed as summaries",
        },
    )


def real_coev_summary(Xs, zs, z_min: int, z_max: int, spec: BehaviourSpec | None = None):
    """Summary row for one observed co-evolution panel. ``spec`` defaults to RSiena's
    all-wave constants (``spec_from_data``)."""
    Xs = [np.asarray(X)[None] for X in Xs]
    zs_arr = np.column_stack([np.asarray(z) for z in zs])  # (n, W)
    if spec is None:
        spec = spec_from_data(zs_arr, z_min, z_max)
    spec_b = BehaviourSpec(z_min, z_max, np.array([spec.zbar]), np.array([spec.sim_mean]))
    zl = [zs_arr[:, w][None] for w in range(zs_arr.shape[1])]
    return coev_summaries(Xs, zl, spec_b, net_model(), beh_model()), spec_b
