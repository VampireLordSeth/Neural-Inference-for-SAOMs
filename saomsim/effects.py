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

Target statistics are ``sum_i s_ik`` with one exception, verified against
RSiena 1.6.6 on the s50 data (``benchmarks/``): RSiena's ``cycle3`` target
counts each 3-cycle *once*, i.e. ``sum_i s_i / 3``, whereas its ``recip`` target
is the plain actor sum (each mutual dyad counted twice). ``statistics()``
follows RSiena. Change statistics are unaffected: the factor 1/3 is a constant
rescaling of the parameter, and RSiena's change statistic for ``cycle3`` is the
plain two-path count ``sum_h x_jh x_hi`` as implemented here.

Covariates are used as given. RSiena centres actor covariates by default;
centre them yourself before passing them in if you want to match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import DTYPE, NUMPY, RowProducts

STRUCTURAL = ("density", "recip", "transTrip", "cycle3")
# Effects taking a numeric parameter rather than a covariate name. outTrunc(c) has
# statistic min(x_i+, c): ties up to c are worth having and further ones are not, which
# is how RSiena represents a questionnaire that caps nominations (docs/GOF_RESULTS.md
# shows the canonical effect set cannot reproduce such a degree distribution without it).
PARAMETRIC = ("outTrunc",)
COVARIATE = ("sameX", "altX", "egoX")
KINDS = STRUCTURAL + PARAMETRIC + COVARIATE


@dataclass(frozen=True)
class Effect:
    kind: str
    covariate: str | None = None
    parameter: float | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown effect {self.kind!r}; known: {KINDS}")
        if self.kind in COVARIATE and self.covariate is None:
            raise ValueError(f"effect {self.kind!r} needs a covariate name")
        if self.kind not in COVARIATE and self.covariate is not None:
            raise ValueError(f"effect {self.kind!r} takes no covariate")
        if self.kind in PARAMETRIC:
            if self.parameter is None:
                raise ValueError(f"effect {self.kind!r} needs a numeric parameter, e.g. "
                                 f"('{self.kind}', None, 5)")
            if not float(self.parameter).is_integer() or self.parameter < 1:
                raise ValueError(f"{self.kind} takes a positive whole number of ties, "
                                 f"got {self.parameter!r}")
        elif self.parameter is not None:
            raise ValueError(f"effect {self.kind!r} takes no parameter")

    @property
    def label(self) -> str:
        if self.parameter is not None:
            return f"{self.kind}({self.parameter:g})"
        return self.kind if self.covariate is None else f"{self.kind}({self.covariate})"


def as_effect(spec) -> Effect:
    if isinstance(spec, Effect):
        return spec
    if isinstance(spec, str):
        return Effect(spec)
    kind, *rest = spec
    cov = rest[0] if rest else None
    par = rest[1] if len(rest) > 1 else None
    return Effect(kind, cov, par)


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
        self._device_cov: dict = {}  # (backend name, device, covariate) -> device array

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
    def _cov(self, name: str, B: int, backend=NUMPY):
        """Covariate as a ``(B, n)`` array on ``backend`` (cached per device)."""
        key = (backend.name, str(backend.device), name)
        v = self._device_cov.get(key)
        if v is None:
            v = backend.covariate(self.covariates[name])
            self._device_cov[key] = v
        if v.ndim == 1:
            return backend.broadcast_to(v, (B, v.shape[0]))
        if v.shape[0] != B:
            raise ValueError(f"covariate {name!r} has batch {v.shape[0]}, expected {B}")
        return v

    # ------------------------------------------------------- change statistics
    # These methods are written against the backend interface only (no bare
    # numpy calls) so the same code runs on numpy arrays and torch tensors.

    def creation_contributions(self, rows: RowProducts, actor, backend=NUMPY):
        """``c_ijk`` for every candidate ``j``: shape ``(B, n, K)``."""
        B, n = rows.out_row.shape
        out = backend.empty((B, n, self.K))
        arangeB = backend.arange(B)
        for k, e in enumerate(self.effects):
            if e.kind == "density":
                out[:, :, k] = 1.0
            elif e.kind == "recip":
                out[:, :, k] = rows.in_col
            elif e.kind == "transTrip":
                out[:, :, k] = rows.two_path + rows.shared_out
            elif e.kind == "cycle3":
                out[:, :, k] = rows.back_path
            elif e.kind == "outTrunc":
                # min(d + 1, c) - min(d, c) for a tie that would be added, and the
                # negative of the same for one that would be removed. Both reduce to
                # "does this actor have fewer than c ties not counting j", because the
                # sign is applied by change_statistics().
                # torch accepts axis= as an alias for dim=, so this is backend-agnostic
                deg = rows.out_row.sum(axis=-1)[:, None]
                out[:, :, k] = backend.as_float((deg - rows.out_row) < e.parameter)
            else:
                v = self._cov(e.covariate, B, backend)
                if e.kind == "sameX":
                    out[:, :, k] = backend.as_float(v == v[arangeB, actor][:, None])
                elif e.kind == "altX":
                    out[:, :, k] = v
                elif e.kind == "egoX":
                    out[:, :, k] = v[arangeB, actor][:, None]
        return out

    def change_statistics(self, rows: RowProducts, actor, backend=NUMPY):
        """``delta_ijk`` for every candidate ``j``: shape ``(B, n, K)``. Diagonal is 0."""
        c = self.creation_contributions(rows, actor, backend)
        sign = 1.0 - 2.0 * rows.out_row
        delta = c * sign[:, :, None]
        delta[backend.arange(rows.out_row.shape[0]), actor, :] = 0.0
        return delta

    def objective(self, rows: RowProducts, actor, theta, backend=NUMPY):
        """``f_ij = sum_k theta_k delta_ijk`` for ``theta`` of shape ``(B, K)`` or ``(K,)``."""
        delta = self.change_statistics(rows, actor, backend)
        if not hasattr(theta, "ndim"):
            theta = backend.array(theta)
        if theta.ndim == 1:
            return delta @ theta
        # multiply-reduce rather than a (B, n, K) @ (B, K, 1) bmm: torch routes that
        # tiny inner dimension to a Triton JIT kernel, which needs a C toolchain
        return (delta * theta[:, None, :]).sum(axis=-1)

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
                # trace(X^3) counts each 3-cycle three times; RSiena's target counts it once
                out[:, k] = np.einsum("bij,bji->b", XX, Xf) / 3.0
            elif e.kind == "outTrunc":
                out[:, k] = np.minimum(Xf.sum(axis=2), e.parameter).sum(axis=1)
            else:
                v = self._cov(e.covariate, B)
                if e.kind == "sameX":
                    out[:, k] = (Xf * (v[:, :, None] == v[:, None, :])).sum(axis=(1, 2))
                elif e.kind == "altX":
                    out[:, k] = (Xf * v[:, None, :]).sum(axis=(1, 2))
                elif e.kind == "egoX":
                    out[:, k] = (Xf * v[:, :, None]).sum(axis=(1, 2))
        return out
