"""Method-of-moments estimation by simulation.

Given ``P`` observed panels ``(X0, X1)`` sharing one parameter vector, find
``theta = (rate, beta_1..beta_K)`` such that the mean simulated statistics
match the mean observed statistics:

    S(x0, x1) = ( #tie changes,  sum_i s_i1(x1), ..., sum_i s_iK(x1) )

This is a Newton iteration with a finite-difference Jacobian. Common random
numbers (the same seed for the base and perturbed simulations) keep the
Jacobian estimate from being swamped by Monte Carlo noise; the ridge in the
pseudo-inverse keeps a near-singular Jacobian from throwing the iterate across
the parameter space. Standard errors are the usual delta-method sandwich

    cov(theta_hat) = D^-1  (Sigma / P)  D^-T

with ``Sigma`` the per-panel covariance of ``S`` at ``theta_hat``.

It is not meant to be a competitor to RSiena's Robbins-Monro algorithm. It is
here so the quickstart can show that a batch of simulated panels carries
recoverable information about the parameters that generated it, and to give
the neural estimator a classical baseline to be compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backend import DTYPE
from .effects import Model
from .simulate import rate_statistic, simulate_period


def moments(X0: np.ndarray, X1: np.ndarray, model: Model) -> np.ndarray:
    """Per-panel statistic vector ``(P, K + 1)``: tie changes, then effect statistics."""
    return np.column_stack([rate_statistic(X0, X1), model.statistics(X1)])


def ridge_pinv(D: np.ndarray, ridge: float) -> np.ndarray:
    """``(D^T D + ridge I)^-1 D^T``, a pseudo-inverse that tolerates near-singular ``D``."""
    k = D.shape[1]
    return np.linalg.solve(D.T @ D + ridge * np.eye(k), D.T)


@dataclass
class EstimateResult:
    names: list[str]
    theta: np.ndarray
    se: np.ndarray
    targets: np.ndarray
    simulated: np.ndarray
    tratios: np.ndarray
    jacobian: np.ndarray
    stat_cov: np.ndarray
    iterations: int
    converged: bool
    history: list[np.ndarray] = field(default_factory=list)

    def table(self, truth=None) -> str:
        head = f"{'parameter':<18}{'estimate':>10}{'s.e.':>9}{'t-conv':>9}"
        if truth is not None:
            head += f"{'truth':>9}{'(est-truth)/se':>16}"
        lines = [head]
        for k, name in enumerate(self.names):
            line = f"{name:<18}{self.theta[k]:>10.3f}{self.se[k]:>9.3f}{self.tratios[k]:>9.2f}"
            if truth is not None:
                z = (self.theta[k] - truth[k]) / self.se[k] if self.se[k] > 0 else np.nan
                line += f"{truth[k]:>9.3f}{z:>16.2f}"
            lines.append(line)
        lines.append(
            f"converged={self.converged} after {self.iterations} iterations "
            f"(max |t-conv| = {np.abs(self.tratios).max():.3f})"
        )
        return "\n".join(lines)


def _initial_theta(X0, X1, model: Model) -> np.ndarray:
    n = X0.shape[1]
    theta = np.zeros(model.K + 1, dtype=DTYPE)
    theta[0] = max(rate_statistic(X0, X1).mean() / n, 0.1)
    for k, e in enumerate(model.effects):
        if e.kind == "density":
            d = np.clip(X1.mean(), 1e-3, 1 - 1e-3)
            theta[k + 1] = np.log(d / (1 - d))
    return theta


def estimate(
    X0: np.ndarray,
    X1: np.ndarray,
    model: Model,
    rng: np.random.Generator,
    *,
    theta0=None,
    n_sim: int = 10,
    max_iter: int = 20,
    fd_step: float = 0.1,
    ridge: float = 1e-3,
    max_step: float = 1.0,
    max_backtrack: int = 4,
    tol: float = 0.1,
    n_final: int | None = None,
    min_rate: float = 0.02,
    verbose: bool = False,
) -> EstimateResult:
    """Simulated method of moments for ``P`` panels ``X0 -> X1``.

    Each iteration simulates at ``theta`` and at ``theta + fd_step e_k`` for
    every ``k`` with common random numbers, forms the Jacobian, and takes a
    Gauss-Newton step on the *standardized* residual ``(mean_sim - mean_obs) / sd``.
    The step is clipped to ``max_step`` per parameter and halved (up to
    ``max_backtrack`` times) until the standardized residual norm decreases.

    n_sim    simulated chains per observed panel, per evaluation
    fd_step  absolute finite-difference step for every parameter
    ridge    Tikhonov term added to ``D^T D`` (in standardized units)
    n_final  chains per panel in the final variance run (default ``5 * n_sim``)
    tol      convergence: all ``|mean_sim - mean_obs| / sd_sim < tol``
    """
    X0 = np.asarray(X0)
    X1 = np.asarray(X1)
    if X0.shape != X1.shape or X0.ndim != 3:
        raise ValueError("X0 and X1 must both be (P, n, n)")
    P = X0.shape[0]
    K1 = model.K + 1
    names = ["rate"] + model.labels

    obs = moments(X0, X1, model)
    target = obs.mean(axis=0)
    X0rep = np.repeat(X0, n_sim, axis=0)

    def simulate_mean(th: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(seed)
        X1s = simulate_period(X0rep, th[1:], th[0], model, r)
        S = moments(X0rep, X1s, model)
        return S.mean(axis=0), S

    def jacobian(th: np.ndarray, seed: int, base: np.ndarray) -> np.ndarray:
        D = np.empty((K1, K1))
        for k in range(K1):
            th_k = th.copy()
            th_k[k] += fd_step
            D[:, k] = (simulate_mean(th_k, seed)[0] - base) / fd_step
        return D

    def clamp(th: np.ndarray) -> np.ndarray:
        th = th.copy()
        th[0] = max(th[0], min_rate)
        return th

    theta = _initial_theta(X0, X1, model) if theta0 is None else np.array(theta0, dtype=DTYPE)
    theta = clamp(theta)
    history = [theta.copy()]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        seed = int(rng.integers(0, 2**63 - 1))
        base, S = simulate_mean(theta, seed)
        sd = S.std(axis=0, ddof=1) + 1e-12
        resid = (base - target) / sd
        if verbose:
            print(f"iter {it:2d}  theta={np.round(theta, 3)}  max|t|={np.abs(resid).max():.3f}")
        if np.all(np.abs(resid) < tol):
            converged = True
            break
        Ds = jacobian(theta, seed, base) / sd[:, None]
        step = ridge_pinv(Ds, ridge) @ resid
        biggest = np.abs(step).max()
        if biggest > max_step:
            step *= max_step / biggest
        cand = clamp(theta - step)
        norm0 = np.linalg.norm(resid)
        for _ in range(max_backtrack):
            new_base, _ = simulate_mean(cand, seed)
            if np.linalg.norm((new_base - target) / sd) < norm0:
                break
            step *= 0.5
            cand = clamp(theta - step)
        theta = cand
        history.append(theta.copy())

    # Final pass at theta_hat: fresh randomness, more chains, Jacobian for the s.e.
    n_final = 5 * n_sim if n_final is None else n_final
    X0fin = np.repeat(X0, n_final, axis=0)
    seed = int(rng.integers(0, 2**63 - 1))
    r = np.random.default_rng(seed)
    S = moments(X0fin, simulate_period(X0fin, theta[1:], theta[0], model, r), model)
    simulated = S.mean(axis=0)
    stat_cov = np.cov(S, rowvar=False)
    sd = np.sqrt(np.diag(stat_cov)) + 1e-12
    tratios = (simulated - target) / sd
    converged = converged and bool(np.all(np.abs(tratios) < tol))

    base_fd, _ = simulate_mean(theta, seed)
    D = jacobian(theta, seed, base_fd)
    Dinv = ridge_pinv(D / sd[:, None], ridge) / sd[None, :]
    theta_cov = Dinv @ (stat_cov / P) @ Dinv.T
    se = np.sqrt(np.clip(np.diag(theta_cov), 0, None))

    return EstimateResult(
        names=names,
        theta=theta,
        se=se,
        targets=target,
        simulated=simulated,
        tratios=tratios,
        jacobian=D,
        stat_cov=stat_cov,
        iterations=it,
        converged=converged,
        history=history,
    )
