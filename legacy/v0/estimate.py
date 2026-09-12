"""
Method-of-moments estimation by Robbins-Monro stochastic approximation.

This is the *baseline*, not the contribution. It exists so that the amortized
neural estimator has something to be compared against on identical simulated
data, and so that the simulator can be exercised end to end.

Three phases, following the structure of the standard SAOM estimation
algorithm:

    Phase 1  estimate the derivative matrix D = d E[s] / d theta by finite
             differences around the starting value
    Phase 2  Robbins-Monro updates  theta <- theta - a_n D^{-1} (s_sim - s_obs)
             with the gain sequence decreasing across subphases
    Phase 3  a long simulation at the final theta to estimate Cov(s), giving
             standard errors  sqrt(diag(D^{-1} Sigma D^{-T}))  and the
             convergence t-ratios used to judge the fit

Conditioning. Estimation runs conditionally on the observed number of changing
tie entries by default, matching RSiena's default and removing the rate
parameter from the moment problem. Pass ``lam`` instead to estimate
unconditionally, in which case the rate is held fixed rather than estimated;
joint rate estimation is deliberately left for a later milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .effects import statistics
from .simulate import observed_distance, simulate_period


@dataclass
class EstimationResult:
    theta: np.ndarray
    se: np.ndarray
    t_convergence: np.ndarray
    target: np.ndarray
    simulated_mean: np.ndarray
    derivative: np.ndarray
    effect_names: list = field(default_factory=list)
    n_iterations: int = 0

    @property
    def max_t_convergence(self) -> float:
        return float(np.max(np.abs(self.t_convergence)))

    def converged(self, threshold: float = 0.10) -> bool:
        """RSiena's rule of thumb: all |t| below ~0.10."""
        return self.max_t_convergence < threshold

    def summary(self) -> str:
        lines = [
            f"{'effect':<14}{'estimate':>11}{'s.e.':>10}{'t-conv':>9}",
            "-" * 44,
        ]
        for k, name in enumerate(self.effect_names):
            lines.append(
                f"{name:<14}{self.theta[k]:>11.4f}{self.se[k]:>10.4f}"
                f"{self.t_convergence[k]:>9.4f}"
            )
        lines.append("-" * 44)
        lines.append(
            f"max |t-conv| = {self.max_t_convergence:.4f}  "
            f"({'converged' if self.converged() else 'NOT converged'})"
        )
        return "\n".join(lines)


def _simulate_stats(X1, theta, effects, cov, n_steps, lam, n_sim, rng):
    """Mean and full matrix of simulated statistics from n_sim chains."""
    B = n_sim
    X0 = np.repeat(X1[None, :, :], B, axis=0)
    steps = None if n_steps is None else np.full(B, n_steps)
    res = simulate_period(X0, theta, effects, cov,
                          n_steps=steps, lam=lam, rng=rng)
    return statistics(res.X, effects, cov)


def estimate_mom(
    X1: np.ndarray,
    X2: np.ndarray,
    effects,
    cov: dict | None = None,
    *,
    theta0=None,
    lam: float | None = None,
    n_sim_phase1: int = 200,
    n_subphases: int = 4,
    n_iter_per_subphase: int = 50,
    n_sim_phase3: int = 1000,
    initial_gain: float = 0.2,
    fd_step: float = 0.15,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> EstimationResult:
    """Estimate SAOM parameters from a two-wave observation.

    Parameters
    ----------
    X1, X2 : (n, n) arrays
        Observed networks at wave 1 and wave 2.
    effects : list of Effect
    cov : dict, optional
    theta0 : (p,) array, optional
        Starting values. Defaults to zeros with a crude density start.
    lam : float, optional
        If given, simulate unconditionally at this rate. Default is
        conditional simulation on the observed Hamming distance.
    """
    rng = rng or np.random.default_rng()
    cov = cov or {}
    p = len(effects)
    names = [e.name for e in effects]

    target = statistics(X2[None, :, :], effects, cov)[0]
    n_steps = None if lam is not None else int(observed_distance(X1[None], X2[None])[0])

    if theta0 is None:
        theta = np.zeros(p)
        n = X1.shape[0]
        dens = X2.sum() / (n * (n - 1))
        dens = min(max(dens, 1e-3), 1 - 1e-3)
        for k, name in enumerate(names):
            if name == "density":
                theta[k] = np.log(dens / (1 - dens))
    else:
        theta = np.asarray(theta0, dtype=np.float64).copy()

    # ---- Phase 1: derivative matrix by central finite differences ----
    D = np.zeros((p, p))
    for k in range(p):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += fd_step
        tm[k] -= fd_step
        sp = _simulate_stats(X1, tp, effects, cov, n_steps, lam, n_sim_phase1, rng).mean(0)
        sm = _simulate_stats(X1, tm, effects, cov, n_steps, lam, n_sim_phase1, rng).mean(0)
        D[:, k] = (sp - sm) / (2 * fd_step)

    # ridge-regularise: the derivative matrix is near-singular when effects are
    # collinear, which is exactly the identification geometry we care about
    D_inv = np.linalg.pinv(D + 1e-8 * np.eye(p))

    # ---- Phase 2: Robbins-Monro ----
    gain = initial_gain
    n_iterations = 0
    for subphase in range(n_subphases):
        theta_sum = np.zeros(p)
        for _ in range(n_iter_per_subphase):
            s = _simulate_stats(X1, theta, effects, cov, n_steps, lam, 1, rng)[0]
            theta = theta - gain * (D_inv @ (s - target))
            theta = np.clip(theta, -10.0, 10.0)
            theta_sum += theta
            n_iterations += 1
        theta = theta_sum / n_iter_per_subphase      # average over the subphase
        gain /= 2.0
        if verbose:
            print(f"  subphase {subphase}: theta = "
                  + ", ".join(f"{v:.3f}" for v in theta))

    # ---- Phase 3: standard errors and convergence check ----
    S = _simulate_stats(X1, theta, effects, cov, n_steps, lam, n_sim_phase3, rng)
    s_mean = S.mean(0)
    s_sd = S.std(0, ddof=1)
    Sigma = np.cov(S, rowvar=False)
    cov_theta = D_inv @ np.atleast_2d(Sigma) @ D_inv.T
    se = np.sqrt(np.clip(np.diag(cov_theta), 0, None))
    t_conv = (s_mean - target) / np.where(s_sd > 0, s_sd, np.inf)

    return EstimationResult(
        theta=theta, se=se, t_convergence=t_conv, target=target,
        simulated_mean=s_mean, derivative=D, effect_names=names,
        n_iterations=n_iterations,
    )
