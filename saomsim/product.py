"""Sampling the product of per-period posteriors.

A SAOM's likelihood factorises over periods, and with a flat box prior the per-period
two-wave posteriors multiply to the joint posterior over (rate_1 .. rate_P, beta); see
``saomsim/multiwave.py`` for the argument and ``docs/MULTIWAVE.md`` for the evidence.
This module does the sampling. Writing phi = (rate_1 .. rate_P, beta),

    log pi(phi) = sum_w log q_w(rate_w, beta),     -inf outside the box,

where each q_w is a normalising flow, so its density is available directly and the box
needs no separate enforcement.

Each q_w is summarised by the mean and covariance of its draws; as a Gaussian it carries
precision A_w^T S_w^-1 A_w into phi-space, where A_w selects (rate_w, beta), so the
Gaussian product has precision sum_w A_w^T S_w^-1 A_w in closed form. We propose from a
multivariate t on that and weight by the exact flow densities, with adaptive random-walk
Metropolis as a fallback and cross-check.

Contrary to the usual advice the proposal is *not* widened: the Gaussian product is
already wider than the true product, the flows having lighter tails than a Gaussian where
they concentrate, and inflating it cost a factor of two to twenty in effective sample size
on both the network and the co-evolution models.

This module needs numpy, scipy and torch but nothing from ``benchmarks/``, so it installs
with the package.
"""

import numpy as np

__all__ = [
    "require_torch",
    "load_posterior",
    "period_posteriors",
    "selectors",
    "gaussian_product",
    "t_logpdf",
    "t_sample",
    "log_target",
    "combine",
    "split_rhat",
    "metropolis",
    "period_spread",
]


def require_torch():
    """Import torch, or explain how to get it.

    torch and sbi are extras: simulating needs neither, and a user who only wants the
    simulator should not have to install a deep-learning stack. But they are needed to
    *load* a trained estimator -- sbi defines the posterior class the pickle refers to --
    and a bare ModuleNotFoundError at that point is an unhelpful way to find out.
    """
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "reading a panel with a trained estimator needs torch and sbi, which are "
            "optional extras of this package. Install them with: "
            "pip install 'saomsim[fit]'"
        ) from e
    return torch


def load_posterior(path, device="cpu"):
    """Load an ``sbi`` posterior saved on any device and pin it to ``device``.

    A posterior trained on the GPU carries its device tag into the pickle, and asking it
    for a density on a CPU-only machine then fails inside torch rather than anywhere
    informative.
    """
    torch = require_torch()

    p = torch.load(path, weights_only=False, map_location=device)
    p._device = device
    for attr in ("potential_fn", "posterior_estimator"):
        obj = getattr(p, attr, None)
        if obj is not None and hasattr(obj, "device"):
            try:
                obj.device = device
            except (AttributeError, RuntimeError):
                pass
    return p


def period_posteriors(post, views, draws=4000, seed=0):
    """(P, d_x) conditioning vectors -> (P, draws, d_theta) per-period posterior samples."""
    torch = require_torch()

    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(views), dtype=torch.float32)
    return np.stack(
        [
            post.sample((draws,), x=X[w : w + 1], show_progress_bars=False).cpu().numpy()
            for w in range(len(X))
        ]
    )


def selectors(P, n_rate, n_eff):
    """A_w: (n_rate + n_eff, dim_phi), picking (rate_w, beta) out of phi.

    phi groups the rates *by period*, so a co-evolution phi reads
    (rate_net_1, rate_beh_1, rate_net_2, ...) rather than grouping by kind.
    """
    dim = P * n_rate + n_eff
    A = np.zeros((P, n_rate + n_eff, dim))
    for w in range(P):
        for j in range(n_rate):
            A[w, j, w * n_rate + j] = 1.0
        for j in range(n_eff):
            A[w, n_rate + j, P * n_rate + j] = 1.0
    return A


def gaussian_product(samples, n_rate):
    """Closed-form Gaussian approximation to the product, as (mean, cov, selectors)."""
    P, _, d = samples.shape
    n_eff = d - n_rate
    A = selectors(P, n_rate, n_eff)
    dim = P * n_rate + n_eff
    Lam, h = np.zeros((dim, dim)), np.zeros(dim)
    for w in range(P):
        mu = samples[w].mean(0)
        S = np.cov(samples[w], rowvar=False) + 1e-9 * np.eye(d)
        Si = np.linalg.inv(S)
        Lam += A[w].T @ Si @ A[w]
        h += A[w].T @ Si @ mu
    cov = np.linalg.inv(Lam + 1e-9 * np.eye(dim))
    return cov @ h, cov, A


def t_logpdf(x, mu, cov, df):
    """Log density of a multivariate t at rows of ``x``."""
    from scipy.linalg import solve_triangular
    from scipy.special import gammaln

    d = len(mu)
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (x - mu).T, lower=True)
    m = (z**2).sum(0)
    return (
        gammaln((df + d) / 2)
        - gammaln(df / 2)
        - 0.5 * d * np.log(df * np.pi)
        - np.log(np.diag(L)).sum()
        - 0.5 * (df + d) * np.log1p(m / df)
    )


def t_sample(mu, cov, df, N, rng):
    d = len(mu)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((N, d))
    u = rng.chisquare(df, N) / df
    return mu + (z / np.sqrt(u)[:, None]) @ L.T


def log_target(post, phi, views, A, batch=50000):
    """sum_w log q_w(A_w phi). The flow returns -inf outside the box."""
    torch = require_torch()

    X = torch.as_tensor(np.asarray(views), dtype=torch.float32)
    total = np.zeros(len(phi))
    for w in range(len(A)):
        th_all = phi @ A[w].T
        out = np.empty(len(phi))
        for s in range(0, len(phi), batch):
            th = torch.as_tensor(th_all[s : s + batch], dtype=torch.float32)
            with torch.no_grad():
                out[s : s + batch] = (
                    post.log_prob(th, x=X[w : w + 1], norm_posterior=False).cpu().numpy()
                )
        total += out
    return total


def combine(
    post,
    views,
    n_rate,
    draws=10000,
    proposal=200000,
    df=8.0,
    inflate=1.0,
    min_ess=500,
    seed=0,
    verbose=True,
):
    """Sample the product of the per-period posteriors. Returns (phi, info).

    ``min_ess`` is an absolute number of effective draws, not a fraction, since that is
    what the posterior summaries actually rest on: 3,000 effective draws out of a poorly
    matched 200,000 is a usable posterior, while 2 % of 5,000 is not.
    """
    rng = np.random.default_rng(seed)
    per = period_posteriors(post, views, draws=max(4000, draws // 2), seed=seed)
    mu, cov, A = gaussian_product(per, n_rate)

    prop = t_sample(mu, cov * inflate, df, proposal, rng)
    lw = log_target(post, prop, views, A) - t_logpdf(prop, mu, cov * inflate, df)
    finite = np.isfinite(lw)
    ess, w = 0.0, None
    if finite.any():
        lw = np.where(finite, lw, -np.inf) - lw[finite].max()
        w = np.exp(lw)
        if w.sum() > 0 and np.isfinite(w.sum()):
            w /= w.sum()
            ess = 1.0 / (w**2).sum()
    info = {
        "ess": ess,
        "ess_frac": ess / proposal,  # how well the proposal matched, in both branches
        "per_period": per,
        "proposal_mean": mu,
        "proposal_cov": cov,
    }
    if ess >= min_ess:
        idx = rng.choice(len(prop), size=draws, replace=True, p=w)
        info["method"] = "importance"
        info["ess_per_draw"] = ess / draws
        return prop[idx], info
    if verbose:
        print(f"  ESS {ess:.0f} effective draws below {min_ess}; falling back to Metropolis")
    phi, acc, rhat, chain_e = metropolis(post, views, A, mu, cov, draws, rng)
    # ``ess_frac`` stays the importance diagnostic, since that is what every caller reads
    # it as; ``ess`` becomes the chain's own, which is what the returned draws are worth.
    info["ess_importance"] = ess
    info["ess"] = float(np.nanmin(chain_e))
    info["ess_per_parameter"] = chain_e
    info["ess_per_draw"] = info["ess"] / len(phi)
    info["method"], info["accept"], info["rhat"] = "metropolis", acc, rhat
    return phi, info


def split_rhat(chain):
    """Split-Rhat per parameter for a (n_draws, n_chains, d) array."""
    n, m, d = chain.shape
    h = n // 2
    if h < 2:
        return np.full(d, np.nan)
    s = np.concatenate([chain[:h], chain[h : 2 * h]], axis=1)  # (h, 2m, d)
    W = s.var(0, ddof=1).mean(0)
    B = h * s.mean(0).var(0, ddof=1)
    var = (h - 1) / h * W + B / h
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(W > 0, var / W, np.nan))


def chain_ess(chain):
    """Effective sample size per parameter for a (n_draws, n_chains, d) array.

    Geyer's initial-positive-sequence estimator, as Stan computes it: autocovariances are
    averaged over chains, turned into autocorrelations against the pooled variance, summed
    in adjacent pairs, and the sum truncated at the first pair that goes negative.

    This is what the Metropolis fallback's draws are worth. The importance ESS that
    triggered the fallback is a statement about the proposal that failed, not about the
    chain that replaced it, and reporting it as the ESS of the returned draws -- which is
    what this code used to do -- overstates them.
    """
    n, m, d = chain.shape
    out = np.full(d, np.nan)
    if n < 8:
        return out
    means = chain.mean(axis=0)  # (m, d)
    W = chain.var(axis=0, ddof=1).mean(axis=0)  # (d,)
    B = means.var(axis=0, ddof=1) * n if m > 1 else np.zeros(d)
    var_plus = ((n - 1) * W + B) / n
    pad = 1
    while pad < 2 * n:
        pad *= 2
    f = np.fft.rfft(chain - means, n=pad, axis=0)
    acov = np.fft.irfft(f * np.conjugate(f), n=pad, axis=0)[:n].real / n
    acov = acov.mean(axis=1)  # (n, d), averaged over chains
    for j in range(d):
        if not np.isfinite(var_plus[j]) or var_plus[j] <= 0:
            continue
        rho = 1.0 - (W[j] - acov[:, j]) / var_plus[j]
        rho[0] = 1.0
        tau, k = -1.0, 0
        while k + 1 < n:
            pair = rho[k] + rho[k + 1]
            if k > 0 and pair < 0:
                break
            tau += 2.0 * pair
            k += 2
        out[j] = m * n / max(tau, 1.0)
    return np.minimum(out, m * n)


def metropolis(post, views, A, mu, cov, draws, rng, chains=8, thin=5, burn=400):
    """Random-walk Metropolis on the same product target, seeded from the proposal.

    The step size adapts during burn-in towards the 0.234 acceptance rate that is optimal
    for a random walk in this many dimensions; adaptation stops before the first kept draw,
    so what is kept is a proper Markov chain. Returns split-Rhat and the per-parameter
    effective sample size alongside the draws, because an unmixed fourteen-dimensional
    chain otherwise looks just like a good one.
    """
    d = len(mu)
    base = np.linalg.cholesky(cov)
    scale = 2.38 / np.sqrt(d)
    cur = mu + rng.standard_normal((chains, d)) @ base.T * 0.5
    lp = log_target(post, cur, views, A)
    keep, acc, n_acc = [], 0.0, 0
    n_iter = draws * thin // chains + burn
    for it in range(n_iter):
        prop = cur + rng.standard_normal((chains, d)) @ base.T * scale
        lpp = log_target(post, prop, views, A)
        take = np.log(rng.random(chains)) < (lpp - lp)
        cur[take], lp[take] = prop[take], lpp[take]
        if it < burn:
            scale *= np.exp(0.5 * (take.mean() - 0.234) / np.sqrt(max(20.0, it + 1)))
        else:
            acc += take.mean()
            n_acc += 1
            if it % thin == 0:
                keep.append(cur.copy())
    chain = np.asarray(keep)  # (n_kept, chains, d)
    phi = chain.reshape(-1, d)[:draws]
    ess = chain_ess(chain) * len(phi) / max(chain.shape[0] * chain.shape[1], 1)
    return phi, acc / max(n_acc, 1), split_rhat(chain), ess


def period_spread(per, n_rate):
    """Per shared effect, max over period pairs of |mu_a - mu_b| / sqrt(s_a^2 + s_b^2).

    The model *asserts* one beta for the whole panel. A large spread says the periods
    disagree about it, which is a claim about time-varying mechanism strength rather than a
    numerical complaint, and a joint estimator cannot report it because it never forms the
    per-period posteriors.
    """
    P = len(per)
    eff = per[:, :, n_rate:]
    mu, sd = eff.mean(1), eff.std(1, ddof=1)
    out = np.zeros(eff.shape[2])
    for a in range(P):
        for b in range(a + 1, P):
            out = np.maximum(out, np.abs(mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2))
    return out
