"""M2: a population of start networks, sizes and covariates (docs/PRIORS_M2.md).

The estimator trained on this learns ``p(theta | X1, X0, covariates, n)`` for
any dataset inside the population. Everything here is a *prior over inputs*;
because the estimator conditions on those inputs, this prior decides where the
estimator is trained, not what it estimates.

Effects are fixed to the M1 set with two generic covariates: ``v`` (continuous
or Likert, centred) for altX/egoX and ``g`` (categorical) for sameX.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import DTYPE, OUT_DTYPE, zero_diagonal
from .effects import Model
from .prior import EXTRA_SUMMARIES, BoxPrior, summaries, transform_summaries
from .simulate import simulate_period

M2_EFFECTS = [
    "density",
    "recip",
    "transTrip",
    "cycle3",
    ("altX", "v"),
    ("egoX", "v"),
    ("sameX", "g"),
]
N_RANGE = (20, 80)
N_MAX = N_RANGE[1]
LIKERT_K = (3, 4, 5)
GROUP_K = (2, 3, 4)
ER_DENSITY = (0.02, 0.20)
BURNIN_STEPS_PER_ACTOR = 10
# "sparse" start regime (docs/PRIORS_M4.md): half the starts draw a mean degree
# instead of a tie fraction, and half of those cap out-degree as nomination surveys do.
MEAN_DEGREE = (2.0, 10.0)
# "survey" start regime (docs/PRIORS_M5.md): every start draws a mean degree k ~ U(0.5, 6),
# the range of emotional-support and best-friend nomination surveys (Baerveldt schools
# 0.8-3.4), half of them out-degree capped as above.
SURVEY_MEAN_DEGREE = (0.5, 6.0)
CAP_EXCESS = (1, 3)  # cap = ceil(mean degree) + U{1..3}


def m2_model(covariates: dict) -> Model:
    return Model(M2_EFFECTS, covariates)


# ------------------------------------------------------------------ covariates


def sample_covariates(B: int, n: int, rng: np.random.Generator) -> dict:
    """Per-chain covariates: ``v`` (B, n) centred continuous-or-Likert, ``g`` (B, n) categorical."""
    v = rng.normal(size=(B, n))
    likert = rng.random(B) < 0.5
    K = rng.choice(LIKERT_K, size=B)
    for b in np.nonzero(likert)[0]:
        v[b] = rng.integers(1, K[b] + 1, size=n)
    v -= v.mean(axis=1, keepdims=True)
    Kg = rng.choice(GROUP_K, size=B)
    g = np.stack([rng.integers(0, k, size=n) for k in Kg]).astype(DTYPE)
    return {"v": v, "g": g}


def covariate_shape(covs: dict) -> np.ndarray:
    """(B, 3): sd of v, number of categories of g, entropy of g (nats)."""
    v, g = covs["v"], covs["g"]
    sd = v.std(axis=1, ddof=1)
    ncat = np.array([len(np.unique(row)) for row in g], dtype=DTYPE)
    ent = np.empty(len(g))
    for b, row in enumerate(g):
        p = np.bincount(row.astype(int)) / row.size
        p = p[p > 0]
        ent[b] = -(p * np.log(p)).sum()
    return np.column_stack([sd, ncat, ent])


# --------------------------------------------------------------- start networks


def cap_outdegree(X: np.ndarray, cap: np.ndarray, rng: np.random.Generator) -> None:
    """In place: actors with more than ``cap[b]`` out-ties keep a random ``cap[b]`` of them.
    ``cap[b] <= 0`` means no cap for that network."""
    B, n, _ = X.shape
    for b in range(B):
        c = int(cap[b])
        if c <= 0:
            continue
        over = X[b].sum(axis=1) > c
        if not over.any():
            continue
        keys = rng.random((int(over.sum()), n)) * X[b, over]  # 0 where there is no tie
        thr = np.partition(keys, n - c, axis=1)[:, n - c]  # the c-th largest key per row
        X[b, over] = (keys >= thr[:, None]).astype(X.dtype)


def sample_start_networks(
    B: int,
    n: int,
    rng: np.random.Generator,
    theta_prior: BoxPrior,
    model: Model,
    *,
    backend="numpy",
    start: str = "m2",
) -> tuple[np.ndarray, dict]:
    """Start networks: half Erdos-Renyi, half burnt in for 10 n ministeps at an
    independent theta0 ~ prior from such a seed. Returns X0 (B, n, n) int8.

    ``start="m2"``: tie fraction d ~ U(0.02, 0.20) for every start (docs/PRIORS_M2.md).
    ``start="sparse"``: half the starts instead draw a mean degree k ~ U(2, 10), so
    d = k / (n - 1) falls with n as real friendship networks do; half of those are
    then capped at ceil(k) + U{1..3} out-ties per actor, as nomination surveys cap
    them (docs/PRIORS_M4.md). The burn-in is applied before the cap.
    ``start="survey"``: every start draws k ~ U(0.5, 6) (docs/PRIORS_M5.md), the
    same cap rule on half of them.
    """
    if start not in ("m2", "sparse", "survey"):
        raise ValueError(f"unknown start regime {start!r}")
    d = rng.uniform(*ER_DENSITY, size=B)
    sparse = np.zeros(B, dtype=bool)
    cap = np.full(B, -1)
    if start in ("sparse", "survey"):
        sparse = rng.random(B) < 0.5 if start == "sparse" else np.ones(B, dtype=bool)
        k = rng.uniform(*(MEAN_DEGREE if start == "sparse" else SURVEY_MEAN_DEGREE), size=B)
        d = np.where(sparse, k / (n - 1), d)
        capped = sparse & (rng.random(B) < 0.5)
        cap = np.where(
            capped, np.ceil(k) + rng.integers(CAP_EXCESS[0], CAP_EXCESS[1] + 1, size=B), -1
        )
    X = (rng.random((B, n, n)) < d[:, None, None]).astype(OUT_DTYPE)
    zero_diagonal(X)
    burn = rng.random(B) < 0.5
    theta0 = theta_prior.sample(B, rng)[:, theta_prior.dim - model.K :]  # drop the rate column(s)
    steps = np.where(burn, BURNIN_STEPS_PER_ACTOR * n, 0)
    X0 = simulate_period(X, theta0, model=model, rng=rng, n_steps=steps, backend=backend)
    if (cap > 0).any():
        X0 = np.ascontiguousarray(X0)
        cap_outdegree(X0, cap, rng)
    return X0, {"er_density": d, "burnin": burn, "theta0": theta0, "sparse": sparse, "cap": cap}


# ------------------------------------------------------------------- summaries


def rate_names(waves: int) -> list[str]:
    """``["rate"]`` for two waves, ``["rate_1", ..., "rate_{W-1}"]`` otherwise (RSiena's layout)."""
    return ["rate"] if waves == 2 else [f"rate_{w}" for w in range(1, waves)]


def m2_summary_names(model: Model, waves: int = 2) -> list[str]:
    x0 = [f"x0_{s}" for s in model.labels + list(EXTRA_SUMMARIES)]
    names = ["n", "v_sd", "g_ncat", "g_entropy"] + x0
    for w in range(1, waves):
        names += [f"x{w}_{s}" for s in ["changes"] + model.labels + list(EXTRA_SUMMARIES)]
    return names


def m2_summaries(X0: np.ndarray, X1, model: Model, covs: dict) -> np.ndarray:
    """Raw summaries: n, covariate shape (3), X0 block (12), then one 13-column block per
    period (change count from the previous wave + statistics of the new wave).
    ``X1`` is a single ``(B, n, n)`` wave or a list of waves after ``X0``."""
    waves_after = X1 if isinstance(X1, (list, tuple)) else [X1]
    B, n, _ = X0.shape
    blocks = [np.full(B, n, dtype=DTYPE), covariate_shape(covs), summaries(X0, X0, model)[:, 1:]]
    prev = X0
    for Xw in waves_after:
        blocks.append(summaries(prev, Xw, model))
        prev = Xw
    return np.column_stack(blocks)


def transform_m2(S: np.ndarray, names: list, model: Model) -> np.ndarray:
    """log1p on counts, asinh on signed sums, in both network blocks; n -> log n."""
    out = np.array(S, dtype=DTYPE, copy=True)
    base = model.labels + list(EXTRA_SUMMARIES)
    prefixes = sorted({nm.split("_", 1)[0] for nm in names if nm.startswith("x")})
    for prefix in prefixes:
        cols = base if prefix == "x0" else ["changes"] + base
        idx = [names.index(f"{prefix}_{c}") for c in cols]
        out[:, idx] = transform_summaries(out[:, idx], cols, model)
    out[:, names.index("n")] = np.log(out[:, names.index("n")])
    return out


# ---------------------------------------------------------------- training set


def _pack(X: np.ndarray, n_max: int) -> np.ndarray:
    """(B, n, n) 0/1 -> (B, n_max, ceil(n_max/8)) uint8, zero padded."""
    B, n, _ = X.shape
    P = np.zeros((B, n_max, n_max), dtype=np.uint8)
    P[:, :n, :n] = X
    return np.packbits(P, axis=-1)


def unpack(packed: np.ndarray, n: int, n_max: int = N_MAX) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=n_max)[:, :n, :n].astype(OUT_DTYPE)


@dataclass
class M2TrainingSet:
    theta: np.ndarray  # (N, 8)
    summary: np.ndarray  # (N, 30) raw
    n: np.ndarray  # (N,)
    theta_names: list
    summary_names: list
    X0: np.ndarray | None = None  # (N, N_MAX, N_MAX/8) packed
    X1: np.ndarray | None = None
    Xw: np.ndarray | None = None  # (N, W-2, N_MAX, N_MAX/8) waves after the second, if any
    v: np.ndarray | None = None  # (N, N_MAX) float32, zero padded
    g: np.ndarray | None = None  # (N, N_MAX) int8, -1 padded
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        empty_u8 = np.zeros((0,), dtype=np.uint8)
        np.savez_compressed(
            path,
            theta=self.theta,
            summary=self.summary,
            n=self.n,
            theta_names=np.array(self.theta_names),
            summary_names=np.array(self.summary_names),
            X0=self.X0 if self.X0 is not None else empty_u8,
            X1=self.X1 if self.X1 is not None else empty_u8,
            Xw=self.Xw if self.Xw is not None else empty_u8,
            v=self.v if self.v is not None else np.zeros((0,), dtype=np.float32),
            g=self.g if self.g is not None else np.zeros((0,), dtype=np.int8),
            meta=np.array(repr(self.meta)),
        )

    @classmethod
    def load(cls, path) -> M2TrainingSet:
        import ast

        with np.load(path, allow_pickle=False) as z:
            keys = ("X0", "X1", "v", "g") + (("Xw",) if "Xw" in z.files else ())
            opt = {k: (z[k] if z[k].size else None) for k in keys}
            return cls(
                theta=z["theta"],
                summary=z["summary"],
                n=z["n"],
                theta_names=[str(s) for s in z["theta_names"]],
                summary_names=[str(s) for s in z["summary_names"]],
                meta=ast.literal_eval(str(z["meta"])),
                **opt,
            )


def load_m2_summaries(paths) -> M2TrainingSet:
    """Concatenate theta / summary / n from several saved sets (networks left on disk)."""
    import glob

    files = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        files.extend(sorted(glob.glob(str(p))) or [str(p)])
    parts = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            parts.append(
                (
                    z["theta"],
                    z["summary"],
                    z["n"],
                    [str(x) for x in z["theta_names"]],
                    [str(x) for x in z["summary_names"]],
                    str(z["meta"]),
                )
            )
    names, snames = parts[0][3], parts[0][4]
    for _, _, _, tn, sn, _ in parts:
        if tn != names or sn != snames:
            raise ValueError("shards have different theta/summary layouts")
    import ast

    return M2TrainingSet(
        theta=np.concatenate([p[0] for p in parts]),
        summary=np.concatenate([p[1] for p in parts]),
        n=np.concatenate([p[2] for p in parts]),
        theta_names=names,
        summary_names=snames,
        meta={
            **ast.literal_eval(parts[0][5]),
            "files": files,
            "N": int(sum(len(p[2]) for p in parts)),
        },
    )


def generate_m2(
    prior: BoxPrior,
    N: int,
    rng: np.random.Generator,
    *,
    n_range: tuple[int, int] = N_RANGE,
    chunk: int = 2048,
    backend="numpy",
    keep_networks: bool = True,
    progress: bool = False,
    waves: int = 2,
    start: str = "m2",
) -> M2TrainingSet:
    """Draw ``N`` (n, covariates, X0, theta) from the population, simulate ``waves - 1``
    periods with one rate per period and shared effects, summarise.

    One ``n`` per chunk so each chunk is a dense (B, n, n) batch. Every draw is
    kept. Networks are stored padded to ``N_MAX`` and bit-packed. ``theta`` is
    ``(rate_1..rate_{W-1}, effects)``.
    """
    n_lo, n_hi = n_range
    if keep_networks and n_hi > N_MAX:
        raise ValueError(f"n_range exceeds N_MAX={N_MAX}; pass keep_networks=False for larger n")
    if waves < 2:
        raise ValueError("waves must be >= 2")
    R = waves - 1
    probe = m2_model(sample_covariates(1, n_lo, rng))
    if list(prior.names) != rate_names(waves) + probe.labels:
        raise ValueError(f"prior names must be {rate_names(waves)} + M2 effect labels")
    names = m2_summary_names(probe, waves)

    theta_parts, S_parts, n_parts, X0_parts, X1_parts, v_parts, g_parts = ([] for _ in range(7))
    Xw_parts = []
    n_chunks = -(-N // chunk)
    for c in range(n_chunks):
        B = min(chunk, N - c * chunk)
        n = int(rng.integers(n_lo, n_hi + 1))
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend, start=start)
        theta = prior.sample(B, rng)
        effects, rates = theta[:, R:], theta[:, :R]
        later = []
        prev = X0
        for w in range(R):
            prev = simulate_period(prev, effects, rates[:, w], model, rng, backend=backend)
            later.append(prev)
        theta_parts.append(theta)
        S_parts.append(m2_summaries(X0, later, model, covs))
        n_parts.append(np.full(B, n, dtype=np.int64))
        if keep_networks:
            X0_parts.append(_pack(X0, N_MAX))
            X1_parts.append(_pack(later[0], N_MAX))
            if R > 1:
                Xw_parts.append(np.stack([_pack(Xw, N_MAX) for Xw in later[1:]], axis=1))
            v = np.zeros((B, N_MAX), dtype=np.float32)
            v[:, :n] = covs["v"]
            g = np.full((B, N_MAX), -1, dtype=np.int8)
            g[:, :n] = covs["g"]
            v_parts.append(v)
            g_parts.append(g)
        if progress and (c % max(1, n_chunks // 20) == 0 or c == n_chunks - 1):
            print(f"  chunk {c + 1}/{n_chunks} (n={n})", flush=True)

    cat = np.concatenate
    return M2TrainingSet(
        theta=cat(theta_parts),
        summary=cat(S_parts),
        n=cat(n_parts),
        theta_names=list(prior.names),
        summary_names=names,
        X0=cat(X0_parts) if keep_networks else None,
        X1=cat(X1_parts) if keep_networks else None,
        Xw=cat(Xw_parts) if (keep_networks and R > 1) else None,
        v=cat(v_parts) if keep_networks else None,
        g=cat(g_parts) if keep_networks else None,
        meta={
            "N": int(N),
            "waves": int(waves),
            "n_range": list(n_range),
            "n_max": N_MAX,
            "effects": probe.labels,
            "prior_low": prior.low.tolist(),
            "prior_high": prior.high.tolist(),
            "start": {
                "m2": "population: 50% ER d~U(0.02,0.2), 50% 10n-step SAOM burn-in at theta0~prior",
                "sparse": "sparse: 50% d~U(0.02,0.2) / 50% mean degree k~U(2,10), half of those "
                "capped at ceil(k)+U{1..3} out-ties; 50% 10n-step SAOM burn-in at theta0~prior",
                "survey": "survey: mean degree k~U(0.5,6) for every start, half capped at "
                "ceil(k)+U{1..3} out-ties; 50% 10n-step SAOM burn-in at theta0~prior",
            }[start],
            "start_regime": start,
            "covariates": "v: 50% N(0,1) / 50% Likert K in {3,4,5}, centred; g: K in {2,3,4}",
        },
    )


def real_data_summary(x0, x1, v, g) -> tuple[np.ndarray, Model]:
    """Summary vector for one observed panel with covariates ``v`` (centred) and ``g``.
    ``x1`` may be a single later wave or a list of later waves."""
    covs = {"v": np.asarray(v, dtype=DTYPE)[None], "g": np.asarray(g, dtype=DTYPE)[None]}
    model = m2_model(covs)
    later = x1 if isinstance(x1, (list, tuple)) else [x1]
    S = m2_summaries(np.asarray(x0)[None], [np.asarray(x)[None] for x in later], model, covs)
    return S, model
