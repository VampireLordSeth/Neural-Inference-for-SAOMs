"""Prior over SAOM parameters and training-set generation for amortized inference.

Design (see docs/PRIORS.md for the reasoning):

  * the start network ``X0`` is fixed and empirical (the observed first wave),
    so the estimator learns ``p(theta | X1, X0)`` for that ``X0`` and those
    covariates and is specific to them;
  * ``theta = (rate, beta_1..beta_K)`` has an independent uniform prior on a
    box; the box is the domain of validity of the estimator, nothing more;
  * every simulated panel goes into the training set. Degenerate outcomes
    (empty or complete networks) are part of the prior predictive and dropping
    them would silently change the prior the estimator is trained under.

Summaries: the default summary statistic vector is exactly the method-of-
moments target, ``(tie changes, sum_i s_ik(X1))``, optionally extended with a
few degree-distribution summaries. Raw networks can be stored too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import DTYPE
from .effects import Model
from .estimate import moments
from .simulate import simulate_period

EXTRA_SUMMARIES = ("outdeg_sd", "indeg_sd", "isolates", "mutual_dyads", "tie_fraction")


@dataclass(frozen=True)
class BoxPrior:
    """Independent uniform prior on ``[low_k, high_k]`` for each named parameter."""

    names: tuple
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        low = np.asarray(self.low, dtype=DTYPE)
        high = np.asarray(self.high, dtype=DTYPE)
        if low.shape != high.shape or low.shape != (len(self.names),):
            raise ValueError("names, low and high must have the same length")
        if np.any(high <= low):
            raise ValueError("every high must exceed its low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def dim(self) -> int:
        return len(self.names)

    def sample(self, N: int, rng: np.random.Generator) -> np.ndarray:
        return self.low + (self.high - self.low) * rng.random((N, self.dim))

    def log_prob(self, theta) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=DTYPE))
        inside = np.all((theta >= self.low) & (theta <= self.high), axis=1)
        return np.where(inside, -np.log(self.high - self.low).sum(), -np.inf)

    def contains(self, theta) -> np.ndarray:
        return np.isfinite(self.log_prob(theta))

    def table(self) -> str:
        w = max(len(n) for n in self.names)
        return "\n".join(
            f"{n:<{w}}  U({lo:g}, {hi:g})"
            for n, lo, hi in zip(self.names, self.low, self.high, strict=True)
        )

    @classmethod
    def for_model(cls, model: Model, rate=(1.0, 12.0), **ranges) -> BoxPrior:
        """Build a prior with a ``rate`` range and one range per effect label,
        e.g. ``BoxPrior.for_model(m, density=(-4, 0), recip=(-1, 4), ...)``.
        Covariate effects are keyed by their label, ``"altX(alc)"``.
        """
        names = ["rate"] + model.labels
        low, high = [rate[0]], [rate[1]]
        for lbl in model.labels:
            if lbl not in ranges:
                raise KeyError(f"no prior range given for effect {lbl!r}")
            lo, hi = ranges[lbl]
            low.append(lo)
            high.append(hi)
        extra = set(ranges) - set(model.labels)
        if extra:
            raise KeyError(f"ranges given for effects not in the model: {sorted(extra)}")
        return cls(tuple(names), np.array(low), np.array(high))


def summaries(X0: np.ndarray, X1: np.ndarray, model: Model, extra: bool = True) -> np.ndarray:
    """Summary statistics per panel: MoM targets, plus degree/dyad summaries if ``extra``."""
    S = moments(X0, X1, model)
    if not extra:
        return S
    X1f = np.asarray(X1, dtype=DTYPE)
    n = X1f.shape[1]
    out = X1f.sum(axis=2)
    ind = X1f.sum(axis=1)
    ex = np.column_stack(
        [
            out.std(axis=1, ddof=1),
            ind.std(axis=1, ddof=1),
            ((out + ind) == 0).sum(axis=1),
            (X1f * X1f.transpose(0, 2, 1)).sum(axis=(1, 2)) / 2,
            X1f.sum(axis=(1, 2)) / (n * (n - 1)),
        ]
    )
    return np.column_stack([S, ex])


def summary_names(model: Model, extra: bool = True) -> list[str]:
    names = ["changes"] + model.labels
    return names + list(EXTRA_SUMMARIES) if extra else names


COUNT_SUMMARIES = {"changes", "isolates", "mutual_dyads"}
COUNT_EFFECTS = {"density", "recip", "transTrip", "cycle3", "sameX"}
SIGNED_EFFECTS = {"altX", "egoX"}


def transform_summaries(S: np.ndarray, names: list, model: Model) -> np.ndarray:
    """Tame the heavy tails before a density estimator sees the summaries.

    Count statistics (non-negative, cubic in ties for the triadic ones) get
    ``log1p``; signed covariate sums get ``asinh`` (log-like in both tails,
    linear near zero); ratios and standard deviations are left alone.
    See docs/PRIORS.md §4. Any downstream z-scoring comes after this.
    """
    by_label = {e.label: e.kind for e in model.effects}
    out = np.array(S, dtype=DTYPE, copy=True)
    for k, name in enumerate(names):
        kind = by_label.get(name)
        if name in COUNT_SUMMARIES or kind in COUNT_EFFECTS:
            out[:, k] = np.log1p(np.clip(out[:, k], 0, None))
        elif kind in SIGNED_EFFECTS:
            out[:, k] = np.arcsinh(out[:, k])
    return out


@dataclass
class TrainingSet:
    theta: np.ndarray  # (N, dim)
    summary: np.ndarray  # (N, S)
    theta_names: list
    summary_names: list
    X1: np.ndarray | None = None  # (N, n, n) int8 if kept
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        """npz; networks are bit-packed along the last axis (n(n) bits -> n * ceil(n/8) bytes)."""
        if self.X1 is not None:
            packed = np.packbits(self.X1.astype(np.uint8), axis=-1)
            n = self.X1.shape[-1]
        else:
            packed, n = np.zeros((0,), dtype=np.uint8), 0
        np.savez_compressed(
            path,
            theta=self.theta,
            summary=self.summary,
            theta_names=np.array(self.theta_names),
            summary_names=np.array(self.summary_names),
            X1_packed=packed,
            X1_n=np.array(n),
            meta=np.array(repr(self.meta)),
        )

    @classmethod
    def load(cls, path) -> TrainingSet:
        import ast

        with np.load(path, allow_pickle=False) as z:
            packed = z["X1_packed"]
            n = int(z["X1_n"])
            X1 = None
            if packed.size:
                X1 = np.unpackbits(packed, axis=-1, count=n).astype(np.int8)
            return cls(
                theta=z["theta"],
                summary=z["summary"],
                theta_names=list(z["theta_names"]),
                summary_names=list(z["summary_names"]),
                X1=X1,
                meta=ast.literal_eval(str(z["meta"])),
            )


def generate_training_set(
    prior: BoxPrior,
    x0: np.ndarray,
    model: Model,
    N: int,
    rng: np.random.Generator,
    *,
    chunk: int = 4096,
    backend="numpy",
    keep_networks: bool = False,
    extra_summaries: bool = True,
    theta: np.ndarray | None = None,
    sort_by_rate: bool = True,
    progress: bool = False,
) -> TrainingSet:
    """Draw ``N`` parameters from ``prior`` (or use ``theta``), simulate one period
    each from the fixed start ``x0``, and return parameters with summaries.

    Every draw is kept. ``chunk`` bounds memory: ``chunk * n * n * 8`` bytes of
    float64 per working array. ``theta`` overrides the prior draw (for SBC or
    fixed-parameter checks) and must have shape ``(N, prior.dim)``.

    ``sort_by_rate`` simulates the draws in order of increasing rate so the
    chains sharing a chunk have similar Poisson step counts (a chunk runs until
    its slowest chain is done). Results are returned in the original order.
    """
    x0 = np.asarray(x0)
    if x0.ndim != 2 or x0.shape[0] != x0.shape[1]:
        raise ValueError("x0 must be a single (n, n) network")
    if list(prior.names) != ["rate"] + model.labels:
        raise ValueError("prior names must be ['rate'] + model.labels, in order")
    theta = prior.sample(N, rng) if theta is None else np.asarray(theta, dtype=DTYPE)
    if theta.shape != (N, prior.dim):
        raise ValueError(f"theta must be ({N}, {prior.dim})")

    n = x0.shape[0]
    order = np.argsort(theta[:, 0], kind="stable") if sort_by_rate else np.arange(N)
    inverse = np.empty(N, dtype=np.int64)
    inverse[order] = np.arange(N)
    theta_sorted = theta[order]

    S_parts, X_parts = [], []
    n_chunks = -(-N // chunk)
    for c, start in enumerate(range(0, N, chunk)):
        th = theta_sorted[start : start + chunk]
        B = th.shape[0]
        X0 = np.repeat(x0[None], B, axis=0)
        X1 = simulate_period(X0, th[:, 1:], th[:, 0], model, rng, backend=backend)
        S_parts.append(summaries(X0, X1, model, extra_summaries))
        if keep_networks:
            X_parts.append(X1)
        if progress and (c % max(1, n_chunks // 20) == 0 or c == n_chunks - 1):
            print(f"  chunk {c + 1}/{n_chunks}", flush=True)
    summary = np.concatenate(S_parts, axis=0)[inverse]
    X1_all = np.concatenate(X_parts, axis=0)[inverse] if keep_networks else None
    return TrainingSet(
        theta=theta,
        summary=summary,
        theta_names=list(prior.names),
        summary_names=summary_names(model, extra_summaries),
        X1=X1_all,
        meta={
            "n": int(n),
            "N": int(N),
            "x0_ties": int(x0.sum()),
            "effects": model.labels,
            "prior_low": prior.low.tolist(),
            "prior_high": prior.high.tolist(),
            "start": "fixed empirical",
        },
    )


def prior_predictive_report(ts: TrainingSet, observed: np.ndarray | None = None) -> str:
    """Quantiles of each summary under the prior predictive, and where the observed
    value falls (its empirical CDF) if given. Values near 0 or 1 mean the prior
    predictive barely covers the data; a flat CDF value ~0.5 means nothing."""
    q = np.quantile(ts.summary, [0.01, 0.05, 0.5, 0.95, 0.99], axis=0)
    w = max(len(s) for s in ts.summary_names)
    lines = [
        f"{'summary':<{w}} {'q01':>9} {'q05':>9} {'median':>9} {'q95':>9} {'q99':>9}"
        + ("   obs     F(obs)" if observed is not None else "")
    ]
    for k, name in enumerate(ts.summary_names):
        row = f"{name:<{w}} " + " ".join(f"{v:>9.2f}" for v in q[:, k])
        if observed is not None:
            F = (ts.summary[:, k] < observed[k]).mean()
            row += f"   {observed[k]:>7.2f}  {F:>6.3f}"
        lines.append(row)
    dens = (
        ts.summary[:, ts.summary_names.index("tie_fraction")]
        if "tie_fraction" in ts.summary_names
        else None
    )
    if dens is not None:
        empty, full, dense = np.mean(dens == 0), np.mean(dens == 1), np.mean(dens > 0.5)
        lines.append(
            f"\nprior predictive: {empty:.3%} empty, {full:.3%} complete, "
            f"{dense:.2%} denser than 0.5"
        )
    return "\n".join(lines)
