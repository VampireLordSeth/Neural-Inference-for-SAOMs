"""Effect definitions: vectorized change statistics and target statistics.

Conventions follow the RSiena manual (Snijders et al.), section on evaluation
function effects. For actor ``i`` and candidate ``j`` the change statistic is

    delta_ijk = s_ik(x with x_ij toggled) - s_ik(x) = (1 - 2 x_ij) * c_ijk

where ``c_ijk`` is the "creation contribution": the change in ``s_ik`` when the
tie i->j is *added* to a network in which it is absent. The diagonal ``j == i``
is the no-change option and always has ``delta = 0``.

Actor statistics ``s_ik`` (summed over ``i`` to give the target statistic):

    density    sum_j x_ij
    recip      sum_j x_ij x_ji
    transTrip  sum_{j,h} x_ij x_ih x_hj
    cycle3     sum_{j,h} x_ij x_jh x_hi
    sameX      sum_j x_ij 1{v_i == v_j}
    altX       sum_j x_ij v_j
    egoX       sum_j x_ij v_i

Note that ``recip`` and ``cycle3`` count each mutual dyad / 3-cycle once *per
actor involved* (twice and three times respectively). This is RSiena's
convention too, but it is exactly the kind of thing the benchmark in
``benchmarks/`` exists to confirm.

Covariates are used as given. RSiena centres actor covariates by default;
centre them yourself before passing them in if you want to match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import DTYPE, RowProducts

STRUCTURAL = ("density", "recip", "transTrip", "cycle3")
COVARIATE = ("sameX", "altX", "egoX")
KINDS = STRUCTURAL + COVARIATE


@dataclass(frozen=True)
class Effect:
    kind: str
    covariate: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown effect {self.kind!r}; known: {KINDS}")
        if self.kind in COVARIATE and self.covariate is None:
            raise ValueError(f"effect {self.kind!r} needs a covariate name")
        if self.kind in STRUCTURAL and self.covariate is not None:
            raise ValueError(f"effect {self.kind!r} takes no covariate")

    @property
    def label(self) -> str:
        return self.kind if self.covariate is None else f"{self.kind}({self.covariate})"


def as_effect(spec) -> Effect:
    if isinstance(spec, Effect):
        return spec
    if isinstance(spec, str):
        return Effect(spec)
    kind, cov = spec
    return Effect(kind, cov)


class Model:
    """A list of effects plus the covariates they refer to.

    ``covariates`` maps name -> array of shape ``(n,)`` (shared by every chain)
    or ``(B, n)`` (one covariate vector per chain).
    """

    def __init__(self, effects, covariates: dict | None = None):
        self.effects = [as_effect(e) for e in effects]
        self.covariates = {k: np.asarray(v, dtype=DTYPE) for k, v in (covariates or {}).items()}
        for e in self.effects:
            if e.covariate is not None and e.covariate not in self.covariates:
                raise KeyError(f"effect {e.label} refers to missing covariate {e.covariate!r}")

    @property
    def K(self) -> int:
        return len(self.effects)

    @property
    def labels(self) -> list[str]:
        return [e.label for e in self.effects]

    @property
    def needs(self) -> frozenset[str]:
        """Which derived row products the effects require (see ``backend``)."""
        out = set()
        for e in self.effects:
            if e.kind == "transTrip":
                out |= {"two_path", "shared_out"}
            elif e.kind == "cycle3":
                out.add("back_path")
        return frozenset(out)

    # ------------------------------------------------------------------ helpers
    def _cov(self, name: str, B: int) -> np.ndarray:
        """Covariate as ``(B, n)``."""
        v = self.covariates[name]
        if v.ndim == 1:
            return np.broadcast_to(v, (B, v.shape[0]))
        if v.shape[0] != B:
            raise ValueError(f"covariate {name!r} has batch {v.shape[0]}, expected {B}")
        return v

    # ------------------------------------------------------- change statistics
    def creation_contributions(self, rows: RowProducts, actor: np.ndarray) -> np.ndarray:
        """``c_ijk`` for every candidate ``j``: shape ``(B, n, K)``."""
        B, n = rows.out_row.shape
        out = np.empty((B, n, self.K), dtype=DTYPE)
        arangeB = np.arange(B)
        for k, e in enumerate(self.effects):
            if e.kind == "density":
                out[:, :, k] = 1.0
            elif e.kind == "recip":
                out[:, :, k] = rows.in_col
            elif e.kind == "transTrip":
                out[:, :, k] = rows.two_path + rows.shared_out
            elif e.kind == "cycle3":
                out[:, :, k] = rows.back_path
            else:
                v = self._cov(e.covariate, B)
                if e.kind == "sameX":
                    out[:, :, k] = v == v[arangeB, actor][:, None]
                elif e.kind == "altX":
                    out[:, :, k] = v
                elif e.kind == "egoX":
                    out[:, :, k] = v[arangeB, actor][:, None]
        return out

    def change_statistics(self, rows: RowProducts, actor: np.ndarray) -> np.ndarray:
        """``delta_ijk`` for every candidate ``j``: shape ``(B, n, K)``. Diagonal is 0."""
        c = self.creation_contributions(rows, actor)
        sign = 1.0 - 2.0 * rows.out_row
        delta = c * sign[:, :, None]
        delta[np.arange(rows.out_row.shape[0]), actor, :] = 0.0
        return delta

    def objective(self, rows: RowProducts, actor: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``f_ij = sum_k theta_k delta_ijk``; ``theta`` is ``(B, K)`` or ``(K,)``."""
        delta = self.change_statistics(rows, actor)
        theta = np.asarray(theta, dtype=DTYPE)
        if theta.ndim == 1:
            return delta @ theta
        return (delta @ theta[:, :, None])[:, :, 0]

    # ------------------------------------------------------- target statistics
    def statistics(self, X: np.ndarray) -> np.ndarray:
        """Target statistics ``sum_i s_ik(x)`` for each chain: shape ``(B, K)``."""
        Xf = X if X.dtype == DTYPE else X.astype(DTYPE)
        B = Xf.shape[0]
        out = np.empty((B, self.K), dtype=DTYPE)
        XX = None
        for k, e in enumerate(self.effects):
            if e.kind == "density":
                out[:, k] = Xf.sum(axis=(1, 2))
            elif e.kind == "recip":
                out[:, k] = (Xf * Xf.transpose(0, 2, 1)).sum(axis=(1, 2))
            elif e.kind == "transTrip":
                XX = Xf @ Xf if XX is None else XX
                out[:, k] = (Xf * XX).sum(axis=(1, 2))
            elif e.kind == "cycle3":
                XX = Xf @ Xf if XX is None else XX
                out[:, k] = np.einsum("bij,bji->b", XX, Xf)
            else:
                v = self._cov(e.covariate, B)
                if e.kind == "sameX":
                    out[:, k] = (Xf * (v[:, :, None] == v[:, None, :])).sum(axis=(1, 2))
                elif e.kind == "altX":
                    out[:, k] = (Xf * v[:, None, :]).sum(axis=(1, 2))
                elif e.kind == "egoX":
                    out[:, k] = (Xf * v[:, :, None]).sum(axis=(1, 2))
        return out
