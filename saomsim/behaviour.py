"""Network-behaviour co-evolution (M3b, docs/PRIORS_M3b.md).

A dependent actor variable ``z`` on an integer scale ``[z_min, z_max]``
evolves alongside the network. Conventions follow RSiena and were pinned by
``benchmarks/rsiena_coevolution.R`` on the s50 data (all numbers reproduced
exactly, see ``benchmarks/test_rsiena_coevolution.py``):

  centring        z~ = z - zbar (RSiena: grand mean over actors and waves);
                  sim_ij = 1 - |z_i - z_j| / range;  similarity centred by simMean
  behaviour effects (actor statistic s_i(x, z); change for a move z_i -> z_i + d):
    linear   z~_i
    quad     z~_i^2
    avAlt    z~_i * mean_{j in out(i)} z~_j            (0 if outdegree 0)
    avSim    mean_{j in out(i)} (sim_ij - simMean)     (0 if outdegree 0)
  selection effects on the network (creation contribution for tie i->j):
    egoX(z)  z~_i        altX(z)  z~_j        simX(z)  sim_ij - simMean
  target statistics per period (cross-lagged, RSiena):
    behaviour effects: z at the END of the period, network at the START
    selection effects: network at the END, z at the START
    behaviour rate:    sum_i |z_i(end) - z_i(start)|

Joint process (RSiena unconditional): events at total rate n (rate_x + rate_z)
per period; each event is a network ministep with probability
rate_x / (rate_x + rate_z), else a behaviour ministep in which a uniformly
chosen actor picks z_i - 1, z_i, or z_i + 1 (options outside the scale
excluded) by multinomial logit on the behaviour objective.

Everything is written against the backend interface so it runs on numpy and
torch alike. ``zbar``, ``simMean`` and ``z_range`` are inputs (per chain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import DTYPE, get_backend
from .effects import Model

BEHAVIOUR_KINDS = ("linear", "quad", "avAlt", "avSim")
SELECTION_KINDS = ("egoZ", "altZ", "simZ")


@dataclass(frozen=True)
class BehaviourSpec:
    """Scale and centring constants of the behaviour variable (per chain arrays or scalars)."""

    z_min: int
    z_max: int
    zbar: object  # (B,) or scalar
    sim_mean: object  # (B,) or scalar

    @property
    def z_range(self) -> int:
        return self.z_max - self.z_min


class BehaviourModel:
    """Behaviour effects (``linear``, ``quad``, ``avAlt``, ``avSim``) and the network's
    selection effects on z (``egoZ``, ``altZ``, ``simZ``)."""

    def __init__(self, effects=("linear", "quad", "avAlt"), selection=("egoZ", "altZ", "simZ")):
        self.effects = list(effects)
        self.selection = list(selection)
        for e in self.effects:
            if e not in BEHAVIOUR_KINDS:
                raise ValueError(f"unknown behaviour effect {e!r}")
        for e in self.selection:
            if e not in SELECTION_KINDS:
                raise ValueError(f"unknown selection effect {e!r}")

    @property
    def K(self) -> int:
        return len(self.effects)

    @property
    def K_sel(self) -> int:
        return len(self.selection)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def constants(spec, B, bk):
        """zbar and simMean as (B, 1) device arrays."""
        zbar = bk.array(np.broadcast_to(np.asarray(spec.zbar, dtype=DTYPE), (B,)))[:, None]
        sm = bk.array(np.broadcast_to(np.asarray(spec.sim_mean, dtype=DTYPE), (B,)))[:, None]
        return zbar, sm

    # ------------------------------------------------ behaviour ministep
    def move_contributions(self, X, z, actor, spec, bk):
        """Change statistics for the three options {-1, 0, +1}: ``(B, 3, K)``.
        Uses the current network ``X`` (B, n, n) and behaviour ``z`` (B, n)."""
        B, n = z.shape
        rows = bk.arange(B)
        zbar, _ = self.constants(spec, B, bk)
        zt = z - zbar
        zi = zt[rows, actor][:, None]  # (B,1) centred focal value
        raw_i = z[rows, actor][:, None]
        out_row = X[rows, actor, :]  # (B,n)
        deg = out_row.sum(axis=-1)[:, None]  # (B,1)
        has = bk.as_float(deg > 0)
        safe_deg = bk.clamp_min(deg, 1.0)
        alt_mean = (out_row * zt).sum(axis=-1)[:, None] / safe_deg  # (B,1)
        deltas = bk.array(np.array([-1.0, 0.0, 1.0]))[None, :]  # (1,3)
        z_new = zi + deltas  # (B,3) centred
        rng_ = float(spec.z_range)
        out = bk.empty((B, 3, self.K))
        for k, e in enumerate(self.effects):
            if e == "linear":
                out[:, :, k] = z_new - zi
            elif e == "quad":
                out[:, :, k] = z_new**2 - zi**2
            elif e == "avAlt":
                out[:, :, k] = (z_new - zi) * alt_mean * has
            elif e == "avSim":
                # mean over out-neighbours of sim(z_i + d, z_j) - sim(z_i, z_j); simMean cancels
                cur = (out_row * (1.0 - abs(raw_i - z) / rng_)).sum(axis=-1)[:, None]
                for d in range(3):
                    moved = raw_i + deltas[:, d : d + 1]
                    new = (out_row * (1.0 - abs(moved - z) / rng_)).sum(axis=-1)[:, None]
                    out[:, d : d + 1, k] = (new - cur) / safe_deg * has
        return out

    def behaviour_objective(self, X, z, actor, theta_beh, spec, bk):
        """(B, 3) objective over {-1, 0, +1}; options off the scale get -inf."""
        c = self.move_contributions(X, z, actor, spec, bk)
        f = (c * theta_beh[:, None, :]).sum(axis=-1)
        rows = bk.arange(z.shape[0])
        zi = z[rows, actor]
        opt = bk.arange(3)[None, :]
        f = bk.where((zi <= spec.z_min)[:, None] & (opt == 0), -1e30, f)
        f = bk.where((zi >= spec.z_max)[:, None] & (opt == 2), -1e30, f)
        return f

    # ------------------------------------------ selection (network side)
    def selection_contributions(self, z, actor, spec, bk):
        """Creation contributions of the selection effects for tie i->j: ``(B, n, K_sel)``."""
        B, n = z.shape
        rows = bk.arange(B)
        zbar, sm = self.constants(spec, B, bk)
        zt = z - zbar
        zi = zt[rows, actor][:, None]
        raw_i = z[rows, actor][:, None]
        out = bk.empty((B, n, self.K_sel))
        for k, e in enumerate(self.selection):
            if e == "egoZ":
                out[:, :, k] = zi + 0.0 * zt
            elif e == "altZ":
                out[:, :, k] = zt
            elif e == "simZ":
                out[:, :, k] = (1.0 - abs(raw_i - z) / float(spec.z_range)) - sm
        return out

    # ------------------------------------------------- target statistics
    def statistics(self, X_start: np.ndarray, z_end: np.ndarray, spec) -> np.ndarray:
        """Behaviour targets (B, K): z at the end of the period, network at the start."""
        X = np.asarray(X_start, dtype=DTYPE)
        z = np.asarray(z_end, dtype=DTYPE)
        zt = z - np.reshape(np.asarray(spec.zbar, dtype=DTYPE), (-1, 1))
        deg = X.sum(axis=2)
        out = np.empty((z.shape[0], self.K))
        for k, e in enumerate(self.effects):
            if e == "linear":
                out[:, k] = zt.sum(axis=1)
            elif e == "quad":
                out[:, k] = (zt**2).sum(axis=1)
            elif e == "avAlt":
                alt = np.where(deg > 0, (X @ zt[:, :, None])[:, :, 0] / np.maximum(deg, 1), 0.0)
                out[:, k] = (zt * alt).sum(axis=1)
            elif e == "avSim":
                S = 1.0 - np.abs(z[:, :, None] - z[:, None, :]) / float(spec.z_range)
                sm = np.reshape(np.asarray(spec.sim_mean, dtype=DTYPE), (-1, 1))
                num = (X * (S - sm[:, :, None])).sum(axis=2)
                out[:, k] = np.where(deg > 0, num / np.maximum(deg, 1), 0.0).sum(axis=1)
        return out

    def selection_statistics(self, X_end: np.ndarray, z_start: np.ndarray, spec) -> np.ndarray:
        """Selection targets (B, K_sel): network at the end of the period, z at the start."""
        X = np.asarray(X_end, dtype=DTYPE)
        z = np.asarray(z_start, dtype=DTYPE)
        zt = z - np.reshape(np.asarray(spec.zbar, dtype=DTYPE), (-1, 1))
        out = np.empty((z.shape[0], self.K_sel))
        for k, e in enumerate(self.selection):
            if e == "egoZ":
                out[:, k] = (X * zt[:, :, None]).sum(axis=(1, 2))
            elif e == "altZ":
                out[:, k] = (X * zt[:, None, :]).sum(axis=(1, 2))
            elif e == "simZ":
                S = 1.0 - np.abs(z[:, :, None] - z[:, None, :]) / float(spec.z_range)
                sm = np.reshape(np.asarray(spec.sim_mean, dtype=DTYPE), (-1, 1, 1))
                out[:, k] = (X * (S - sm)).sum(axis=(1, 2))
        return out


def behaviour_rate_statistic(z_start, z_end) -> np.ndarray:
    return np.abs(np.asarray(z_end, dtype=DTYPE) - np.asarray(z_start, dtype=DTYPE)).sum(axis=1)


# ------------------------------------------------------------ joint simulation


def simulate_coevolution(
    X0,
    z0,
    theta_net,
    theta_sel,
    theta_beh,
    rate_net,
    rate_beh,
    model: Model,
    bmodel: BehaviourModel,
    spec: BehaviourSpec,
    rng: np.random.Generator,
    *,
    backend="numpy",
    return_info: bool = False,
):
    """One period of joint network-behaviour evolution.

    X0 (B, n, n) 0/1; z0 (B, n) integers on [z_min, z_max];
    theta_net (K,) or (B, K) for ``model``; theta_sel (K_sel,) or (B, K_sel);
    theta_beh (K_beh,) or (B, K_beh); rate_net, rate_beh scalar or (B,).
    Returns (X1 int8, z1 int) and optionally the event counts.
    """
    bk = get_backend(backend)
    X0 = np.asarray(X0)
    z0 = np.asarray(z0)
    B, n, _ = X0.shape
    rate_net = np.broadcast_to(np.asarray(rate_net, dtype=DTYPE), (B,))
    rate_beh = np.broadcast_to(np.asarray(rate_beh, dtype=DTYPE), (B,))
    total = rate_net + rate_beh
    n_events = rng.poisson(total * n)
    p_net = np.where(total > 0, rate_net / np.maximum(total, 1e-300), 0.5)

    def bcast(th, K):
        th = np.asarray(th, dtype=DTYPE)
        return np.broadcast_to(th, (B, K)) if th.ndim == 1 else th

    th_net = bk.array(bcast(theta_net, model.K))
    th_sel = bk.array(bcast(theta_sel, bmodel.K_sel))
    th_beh = bk.array(bcast(theta_beh, bmodel.K))
    X = bk.network(X0)
    z = bk.array(z0.astype(DTYPE))
    steps = bk.int_array(n_events)
    p_net_d = bk.array(p_net)
    state = bk.rng_state(rng)
    needs = model.needs
    n_net = np.zeros(B, dtype=np.int64)
    n_beh = np.zeros(B, dtype=np.int64)
    rowsB = bk.arange(B)

    for t in range(int(n_events.max()) if B else 0):
        active = t < steps
        if not bool(active.any()):
            break
        is_net = bk.random(B, state) < p_net_d
        actor = bk.integers(n, B, state)
        u = bk.random(B, state)
        # network ministep (all chains compute; only active & is_net toggle)
        rows = bk.row_products(X, actor, needs)
        c_sel = bmodel.selection_contributions(z, actor, spec, bk)
        sign = 1.0 - 2.0 * rows.out_row
        f_sel = (c_sel * th_sel[:, None, :]).sum(axis=-1) * sign
        f_sel[rowsB, actor] = 0.0
        f = model.objective(rows, actor, th_net, bk) + f_sel
        target = bk.categorical_sample(bk.softmax(f), u)
        do_net = active & is_net
        bk.toggle(X, actor, target, do_net)
        # behaviour ministep
        fb = bmodel.behaviour_objective(X, z, actor, th_beh, spec, bk)
        choice = bk.categorical_sample(bk.softmax(fb), u)  # 0,1,2 -> -1,0,+1
        do_beh = active & ~is_net
        dz = bk.as_float(choice) - 1.0
        r, a = rowsB[do_beh], actor[do_beh]
        z[r, a] = z[r, a] + dz[do_beh]
        if return_info:
            n_net += bk.to_numpy(do_net).astype(np.int64)
            n_beh += bk.to_numpy(do_beh).astype(np.int64)

    X1 = bk.finalize(X)
    z1 = np.rint(bk.to_numpy(z)).astype(np.int64)
    if return_info:
        return X1, z1, {"n_events": n_events, "n_net": n_net, "n_beh": n_beh}
    return X1, z1


def random_behaviour(
    B: int, n: int, z_min: int, z_max: int, rng: np.random.Generator
) -> np.ndarray:
    """Start behaviour: per chain a random mean on the scale, roughly unimodal."""
    centre = rng.uniform(z_min, z_max, size=(B, 1))
    z = np.rint(centre + rng.normal(scale=1.0, size=(B, n)))
    return np.clip(z, z_min, z_max).astype(np.int64)


def spec_from_data(z_waves, z_min: int, z_max: int) -> BehaviourSpec:
    """RSiena's constants from observed behaviour ``z_waves`` (n, W):
    ``zbar`` = grand mean over all actors and waves; ``simMean`` = mean similarity over
    ordered pairs i != j in the *period-start* waves 1..W-1 (verified on s50: 0.67439).
    With a single wave (simulation from a start), both use that wave."""
    z = np.asarray(z_waves, dtype=DTYPE)
    if z.ndim == 1:
        z = z[:, None]
    n, W = z.shape
    zbar = z.mean()
    rng_ = z_max - z_min
    start_waves = range(W - 1) if W > 1 else range(1)
    sims = []
    for w in start_waves:
        S = 1.0 - np.abs(z[:, w][:, None] - z[:, w][None, :]) / rng_
        sims.append(S[~np.eye(n, dtype=bool)])
    return BehaviourSpec(z_min, z_max, zbar, float(np.concatenate(sims).mean()))
