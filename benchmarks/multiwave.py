"""Read a panel of any length with the two-wave estimator, by sampling the product
of its per-period posteriors.

    python benchmarks/multiwave.py --posterior data/npe_m5c.pt --real s50
    python benchmarks/multiwave.py --posterior data/npe_m5c.pt --real glasgow --draws 20000

The SAOM likelihood factorises over periods and our prior is flat on a box, so the
per-period posteriors multiply to the joint one (``saomsim/multiwave.py``). Writing
phi = (rate_1 .. rate_P, beta) for the P = waves - 1 periods and the shared effects,

    log pi(phi) = sum_w log q_w(rate_w, beta)      inside the box, -inf outside.

We sample it by importance sampling from a heavy-tailed approximation of that product.
Each q_w is summarised by the mean and covariance of its draws; as a Gaussian it carries
precision A_w^T S_w^-1 A_w into phi-space, where A_w selects (rate_w, beta), so the
Gaussian product has precision sum_w A_w^T S_w^-1 A_w in closed form. We propose from a
multivariate t on that mean and covariance and weight by the exact flow densities. The
effective sample size says whether that worked; below ``--min-ess`` effective draws we
fall back to adaptive random-walk Metropolis on the same target, seeded from the proposal
and reporting split-Rhat.

Contrary to the usual advice, the proposal is *not* widened: the Gaussian product is
already wider than the true product, because the flows have lighter tails than a Gaussian
where they concentrate. Inflating it by 1.5-4.0 cut the effective sample size by a factor
of two to twenty in both models, so ``inflate`` defaults to 1.0.

Two diagnostics come free and are worth more than the point estimates:

* **ESS** -- a low one means the proposal misses the product, usually because the
  per-period posteriors overlap poorly.
* **period spread** -- per shared effect, (mu_a - mu_b) / sqrt(s_a^2 + s_b^2) over pairs
  of periods. The model assumes one beta for the whole panel; a large value says the
  periods disagree about it, which is a finding about the data (a time-varying effect),
  not a numerical problem. The product posterior means little when this is large, and it
  is exactly the check a single joint estimator cannot show you, because it never forms
  the per-period posteriors in the first place.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saomsim.multiwave import m2_period_views, n_periods  # noqa: E402
from saomsim.population import m2_summary_names  # noqa: E402


def load_posterior(path, device="cpu"):
    """Load an sbi posterior saved on any device and pin it to ``device``.

    A posterior trained on the GPU carries its device tag into the pickle, and asking it
    for a density on a CPU-only machine then fails inside torch rather than anywhere
    informative.
    """
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
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(views), dtype=torch.float32)
    return np.stack(
        [
            post.sample((draws,), x=X[w : w + 1], show_progress_bars=False).cpu().numpy()
            for w in range(len(X))
        ]
    )


def selectors(P, n_rate, n_eff):
    """A_w: (n_rate + n_eff, dim_phi), picking (rate_w, beta) out of phi."""
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
    """sum_w log q_w(A_w phi). The flow returns -inf outside the box, so the box needs
    no separate enforcement."""
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

    ``inflate`` defaults to 1.0 rather than the usual widening: the Gaussian product is
    already *wider* than the true product, because the flows have lighter-than-Gaussian
    tails where they concentrate, so inflating it only wastes proposals. Measured on the
    s50 three-wave panel, inflating by 1.5-4.0 cut the effective sample size by a factor
    of two to twenty in both the network and the co-evolution models.

    ``min_ess`` is an absolute number of effective draws, not a fraction, since that is
    what the posterior summaries actually rest on -- 3,000 effective draws out of a poorly
    matched 200,000 is a perfectly usable posterior, while 2 % of 5,000 is not.
    """
    rng = np.random.default_rng(seed)
    per = period_posteriors(post, views, draws=max(4000, draws // 2), seed=seed)
    mu, cov, A = gaussian_product(per, n_rate)

    prop = t_sample(mu, cov * inflate, df, proposal, rng)
    lw = log_target(post, prop, views, A) - t_logpdf(prop, mu, cov * inflate, df)
    finite = np.isfinite(lw)
    ess = 0.0
    if finite.any():
        lw = np.where(finite, lw, -np.inf) - lw[finite].max()
        w = np.exp(lw)
        if w.sum() > 0 and np.isfinite(w.sum()):
            w /= w.sum()
            ess = 1.0 / (w**2).sum()
    info = {
        "ess": ess,
        "ess_frac": ess / proposal,
        "per_period": per,
        "proposal_mean": mu,
        "proposal_cov": cov,
    }
    if ess >= min_ess:
        idx = rng.choice(len(prop), size=draws, replace=True, p=w)
        info["method"] = "importance"
        return prop[idx], info
    if verbose:
        print(f"  ESS {ess:.0f} effective draws below {min_ess}; falling back to Metropolis")
    phi, acc, rhat = metropolis(post, views, A, mu, cov, draws, rng)
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


def metropolis(post, views, A, mu, cov, draws, rng, chains=8, thin=5, burn=400):
    """Random-walk Metropolis on the same product target, seeded from the proposal.

    The step size adapts during burn-in towards the 0.234 acceptance rate that is optimal
    for a random walk in this many dimensions; adaptation stops before the first kept
    draw, so what is kept is a proper Markov chain. Returns split-Rhat alongside the
    draws, because an unmixed 14-dimensional chain otherwise looks just like a good one.
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
    return chain.reshape(-1, d)[:draws], acc / max(n_acc, 1), split_rhat(chain)


def period_spread(per, n_rate):
    """Per shared effect, max over period pairs of |mu_a - mu_b| / sqrt(s_a^2 + s_b^2)."""
    P = len(per)
    eff = per[:, :, n_rate:]
    mu, sd = eff.mean(1), eff.std(1, ddof=1)
    out = np.zeros(eff.shape[2])
    for a in range(P):
        for b in range(a + 1, P):
            out = np.maximum(out, np.abs(mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2))
    return out


def net_views(real, waves):
    """Real network-only panel -> (views, effect names, n_rate, periods)."""
    from benchmarks.m2 import glasgow_as_m2, s50_as_m2

    loader = s50_as_m2 if real == "s50" else glasgow_as_m2
    x0, _, _, _, S, model = loader(waves=waves)
    names = m2_summary_names(model, waves)
    views = m2_period_views(S, names, model, transform=True)[0]
    return views, list(model.labels), 1, n_periods(names), x0.shape[0]


def coev_views(real, waves):
    """Real co-evolution panel -> (views, effect names, n_rate, periods).

    Re-slicing will not do here (see ``saomsim.multiwave.coev_period_views``), so the
    per-period vectors are recomputed from the waves themselves.
    """
    from saomsim.behaviour import BehaviourSpec, spec_from_data
    from saomsim.multiwave import coev_period_views
    from saomsim.population_coev import (
        BEH_EFFECTS,
        NET_EFFECTS,
        SEL_EFFECTS,
        beh_model,
        coev_summary_names,
        net_model,
        transform_coev,
    )

    net_paths, z_path = REAL_COEV[real]
    Xs = [np.loadtxt(p, delimiter=",", dtype=np.int8)[None] for p in net_paths[:waves]]
    Z = np.loadtxt(z_path, delimiter=",")
    sp = spec_from_data(Z, 1, 5)
    spec = BehaviourSpec(1, 5, np.array([sp.zbar]), np.array([sp.sim_mean]))
    zs = [Z[:, w][None] for w in range(waves)]
    model, bmodel = net_model(), beh_model()
    views = coev_period_views(Xs, zs, spec, model, bmodel)[0]
    two = coev_summary_names(2)
    views = transform_coev(views, two, model)
    return views, NET_EFFECTS + SEL_EFFECTS + BEH_EFFECTS, 2, waves - 1, Xs[0].shape[1]


TRAINED_N = {  # two-wave estimator -> the n range its population actually covered
    "npe_m5.pt": (30, 100),
    "npe_m5b.pt": (30, 100),
    "npe_m5c.pt": (30, 100),
    "npe_m5c_mom.pt": (30, 100),
    "npe_coev_m5.pt": (30, 100),
    "npe_coev_m5b.pt": (30, 100),
    "npe_coev_m5c.pt": (30, 100),
}


def check_n(posterior_path, n):
    """Warn if the panel's n falls outside the estimator's training range.

    The screening references band n in steps of 20, so they cannot tell n = 25 from
    n = 35 and will not raise this themselves; the range is recorded here instead.
    Amortized estimators extrapolate silently and confidently, so an out-of-range n has
    to be stated rather than left to the reader.
    """
    rng = TRAINED_N.get(Path(posterior_path).name)
    if rng is None:
        print(f"  n = {n}; training range of {Path(posterior_path).name} not recorded")
        return None
    lo, hi = rng
    if lo <= n <= hi:
        return True
    print(
        f"  ** n = {n} is outside this estimator's training range [{lo}, {hi}]. The "
        f"posterior below is an extrapolation and should not be read as calibrated. **"
    )
    return False


def csv_views(paths, v_path=None, g_path=None, beh_path=None):
    """Any panel from CSVs -> (views, effect names, n_rate, periods). Co-evolution when
    ``beh_path`` is given, network-only otherwise."""
    from benchmarks.fit import load_waves

    Xs = load_waves(paths)
    W = len(Xs)
    if beh_path is None:
        from saomsim.population import real_data_summary

        if not (v_path and g_path):
            raise SystemExit("the network model needs --v and --g")
        v = np.loadtxt(v_path, delimiter=",", ndmin=1)
        g = np.loadtxt(g_path, delimiter=",", ndmin=1)
        g = g - g.min()
        S, model = real_data_summary(Xs[0], Xs[1:], v, g)
        names = m2_summary_names(model, W)
        views = m2_period_views(S, names, model, transform=True)[0]
        return views, list(model.labels), 1, W - 1, Xs[0].shape[0]

    from saomsim.behaviour import BehaviourSpec, spec_from_data
    from saomsim.multiwave import coev_period_views
    from saomsim.population_coev import (
        BEH_EFFECTS,
        NET_EFFECTS,
        SEL_EFFECTS,
        beh_model,
        coev_summary_names,
        net_model,
        transform_coev,
    )

    Z = np.loadtxt(beh_path, delimiter=",", ndmin=2)
    if Z.shape[1] != W:
        raise SystemExit(f"behaviour has {Z.shape[1]} waves but {W} networks were given")
    sp = spec_from_data(Z, 1, 5)
    spec = BehaviourSpec(1, 5, np.array([sp.zbar]), np.array([sp.sim_mean]))
    model, bmodel = net_model(), beh_model()
    views = coev_period_views([X[None] for X in Xs], [Z[:, w][None] for w in range(W)],
                              spec, model, bmodel)[0]
    views = transform_coev(views, coev_summary_names(2), model)
    return views, NET_EFFECTS + SEL_EFFECTS + BEH_EFFECTS, 2, W - 1, Xs[0].shape[0]


REAL_COEV = {
    "s50": ([Path(__file__).parent / f"s50{w}.csv" for w in (1, 2, 3)],
            Path(__file__).parent / "s50a.csv"),
    "glasgow": ([Path(__file__).parent / "glasgow" / f"glasgow_net{w}.csv" for w in (1, 2, 3)],
                Path(__file__).parent / "glasgow" / "glasgow_alcohol.csv"),
}


def rsiena_compare(path, phi_names, net="net", beh="alc"):
    """Map our phi names onto the RSiena fit JSONs the R scripts here write.

    Returns {our name: (estimate, se)} for the names present, and the list of names that
    had no counterpart, rather than quietly dropping them.
    """
    import json

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    est, se = d["estimate"], d.get("se", {})
    sel = {"egoZ": "egoX", "altZ": "altX", "simZ": "simX"}
    out, missing = {}, []
    for nm in phi_names:
        if nm.startswith("rate_net_"):
            key = f"{net}:rate_{nm.rsplit('_', 1)[1]}"
        elif nm.startswith("rate_beh_"):
            key = f"{beh}:rate_{nm.rsplit('_', 1)[1]}"
        elif nm.startswith("rate_"):
            key = f"{net}:rate_{nm.rsplit('_', 1)[1]}"
        elif nm in ("linear", "quad", "avAlt"):
            key = f"{beh}:{nm}"
        else:
            base = nm.split("(", 1)[0]  # altX(v) -> altX, as RSiena labels it
            key = f"{net}:{sel.get(base, base)}"
        if key in est:
            out[nm] = (est[key], se.get(key, float("nan")))
        else:
            missing.append((nm, key))
    return out, missing


def phi_names_for(kind, waves, n_rate, eff_names):
    """Names for phi = (rates of period 1, ..., rates of period P, shared effects).

    Note the rate order: phi groups rates *by period*, so co-evolution reads
    (rate_net_1, rate_beh_1, rate_net_2, rate_beh_2, ...), whereas
    ``coev_theta_names`` groups them by kind. They are not interchangeable.
    """
    P = waves - 1
    if n_rate == 1:
        rates = [f"rate_{w + 1}" for w in range(P)]
    else:
        rates = [f"{k}_{w + 1}" for w in range(P) for k in ("rate_net", "rate_beh")]
    return rates + list(eff_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_m5c.pt")
    ap.add_argument("--model", default="net", choices=["net", "coev"])
    ap.add_argument("--real", default="s50", choices=["s50", "glasgow"])
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--proposal", type=int, default=200000)
    ap.add_argument("--min-ess", type=float, default=500,
                    help="effective draws below which to fall back to Metropolis")
    ap.add_argument("--waves-csv", nargs="+",
                    help="adjacency CSVs in time order; any number of waves")
    ap.add_argument("--v", help="numeric actor covariate CSV (network model)")
    ap.add_argument("--g", help="categorical actor covariate CSV (network model)")
    ap.add_argument("--behaviour", help="n x waves behaviour CSV (co-evolution model)")
    ap.add_argument("--rsiena", help="RSiena fit JSON to compare against")
    ap.add_argument("--rsiena-labels", default="net,alc",
                    help="network,behaviour prefixes in that JSON")
    ap.add_argument("--cross-check", action="store_true",
                    help="also run Metropolis and report how far the two agree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the joint samples to this .npz")
    a = ap.parse_args()

    if a.waves_csv:
        views, eff_names, n_rate, P, n = csv_views(a.waves_csv, a.v, a.g, a.behaviour)
        label, n_waves = Path(a.waves_csv[0]).stem, len(a.waves_csv)
    else:
        views, eff_names, n_rate, P, n = (net_views if a.model == "net" else coev_views)(
            a.real, a.waves
        )
        label, n_waves = a.real, a.waves

    post = load_posterior(a.posterior)
    n_eff = len(post.prior.base_dist.low) - n_rate
    if len(eff_names) != n_eff:
        raise SystemExit(
            f"{Path(a.posterior).name} has {n_rate} rates + {n_eff} effects per period, but the "
            f"{a.model} panel supplies {len(eff_names)} effect names -- wrong --model?"
        )

    print(f"{label}, {n_waves} waves -> {P} periods, two-wave estimator {Path(a.posterior).name}")
    check_n(a.posterior, n)
    phi, info = combine(
        post, views, n_rate, draws=a.draws, proposal=a.proposal, min_ess=a.min_ess, seed=a.seed
    )
    print(
        f"  {info['method']}, ESS {info['ess']:.0f} effective draws "
        f"({info['ess_frac'] * 100:.1f}% of {a.proposal})"
        + (f", accept {info['accept']:.2f}" if "accept" in info else "")
        + (f", max Rhat {np.nanmax(info['rhat']):.3f}" if "rhat" in info else "")
    )
    if a.cross_check:
        mu, cov = info["proposal_mean"], info["proposal_cov"]
        A = selectors(P, n_rate, len(eff_names))
        alt, acc, rhat = metropolis(
            post, views, A, mu, cov, a.draws, np.random.default_rng(a.seed + 1)
        )
        gap = (phi.mean(0) - alt.mean(0)) / np.sqrt(phi.var(0, ddof=1) + alt.var(0, ddof=1))
        print(
            f"  cross-check by Metropolis: accept {acc:.2f}, max Rhat {np.nanmax(rhat):.3f}, "
            f"largest standardised gap in the means {np.abs(gap).max():.3f}"
        )

    phi_names = phi_names_for(a.model, n_waves, n_rate, eff_names)
    per = info["per_period"]
    if len(phi_names) != phi.shape[1]:
        raise SystemExit(
            f"{len(phi_names)} names for {phi.shape[1]} sampled parameters -- the wave count "
            "used for naming does not match the panel"
        )
    print(f"\n{'parameter':<14}{'mean':>9}{'sd':>8}{'2.5%':>9}{'97.5%':>9}   per-period means")
    for j, nm in enumerate(phi_names):
        col = phi[:, j]
        q = np.percentile(col, [2.5, 97.5])
        if j < P * n_rate:
            note = f"period {j // n_rate + 1} only: {per[j // n_rate][:, j % n_rate].mean():+.3f}"
        else:
            k = j - P * n_rate + n_rate
            note = "  ".join(f"{per[w][:, k].mean():+.3f}" for w in range(P))
        print(f"{nm:<14}{col.mean():9.3f}{col.std(ddof=1):8.3f}{q[0]:9.3f}{q[1]:9.3f}   {note}")

    if a.rsiena:
        net_lab, beh_lab = (a.rsiena_labels.split(",") + ["alc"])[:2]
        ref, missing = rsiena_compare(a.rsiena, phi_names, net_lab, beh_lab)
        print(f"\nagainst RSiena ({Path(a.rsiena).name})")
        print(f"{'parameter':<14}{'ours':>10}{'RSiena':>10}{'se':>8}{'z':>8}")
        zs = []
        for j, nm in enumerate(phi_names):
            if nm not in ref:
                continue
            e, s = ref[nm]
            z = (phi[:, j].mean() - e) / np.sqrt(phi[:, j].var(ddof=1) + (s if s == s else 0) ** 2)
            zs.append(abs(z))
            print(f"{nm:<14}{phi[:, j].mean():10.3f}{e:10.3f}{s:8.3f}{z:+8.2f}")
        if zs:
            print(f"  largest |z| {max(zs):.2f}")
        if missing:
            print(f"  no counterpart in that JSON: {', '.join(n for n, _ in missing)}")

    spread = period_spread(per, n_rate)
    print("\nperiod spread, max |mu_a - mu_b| / sqrt(s_a^2 + s_b^2) over period pairs:")
    for nm, s in zip(eff_names, spread, strict=True):
        print(f"  {nm:<14}{s:6.2f}" + ("   <-- periods disagree" if s > 2 else ""))

    if a.out:
        np.savez(
            a.out,
            samples=phi,
            names=np.array(phi_names),
            per_period=per,
            ess_frac=info["ess_frac"],
            spread=spread,
        )
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
