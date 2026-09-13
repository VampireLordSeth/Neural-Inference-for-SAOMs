"""Out-of-distribution characterisation of an M2 estimator (paper plan §4).

    python benchmarks/ood_m2.py --posterior data/npe_m2_10m.pt --ref data/m2_shards/train_m2_s10.npz

Five ways a real panel can leave the training population, 1,000 panels each,
plus an in-distribution control:

  id         the population itself (control)
  n_small    n = 15            (population: 20..80)
  n_large    n = 110           (population: 20..80)
  dense_x0   ER start at d = 0.35 (population ER: 0.02..0.20)
  covariates v with sd 3, g with 6 categories (population: sd ~1, 2..4 cats)
  theta_out  theta on/over a face of the prior box (rate 14, density -4.5,
             transTrip 1.8; other parameters inside)
  hidden_hom homophily (beta = 1.5) on a covariate the analyst does not observe
             (misspecification: the fitted model has no such effect)

For each set: SBC mean rank and coverage where a true in-box theta exists,
posterior-sd inflation relative to the control, mass against the box faces,
and the detection score. The detection heuristic is the applied user's tool:
z-score the transformed summary vector against the training population and
take the maximum |z| (and a robust percentile variant). Threshold at the 95th
percentile of the control; report the flag rate per set and on s50.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.coverage import coverage_table  # noqa: E402
from benchmarks.m2 import m2_prior, s50_as_m2  # noqa: E402
from saomsim import Model, simulate_period  # noqa: E402
from saomsim.backend import DTYPE, OUT_DTYPE, zero_diagonal  # noqa: E402
from saomsim.population import (  # noqa: E402
    M2_EFFECTS,
    load_m2_summaries,
    m2_model,
    m2_summaries,
    sample_covariates,
    sample_start_networks,
    transform_m2,
)


# ------------------------------------------------------------- generators
def covariates_custom(B, n, rng, v_sd=1.0, g_cats=(2, 3, 4)):
    covs = sample_covariates(B, n, rng)
    covs["v"] = covs["v"] * v_sd
    Kg = rng.choice(g_cats, size=B)
    covs["g"] = np.stack([rng.integers(0, k, size=n) for k in Kg]).astype(DTYPE)
    return covs


def start_er(B, n, rng, d):
    X = (rng.random((B, n, n)) < d).astype(OUT_DTYPE)
    return zero_diagonal(X)


def make_set(kind, B, rng, prior, backend):
    """Return theta (B,8) (or None when no in-model truth), n, X0, X1, covs, model."""
    n = int(rng.integers(20, 81))
    theta = prior.sample(B, rng)
    hidden = None
    if kind == "id":
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
    elif kind == "n_small":
        n = 15
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
    elif kind == "n_large":
        n = 110
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
    elif kind == "dense_x0":
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0 = start_er(B, n, rng, 0.35)
    elif kind == "covariates":
        covs = covariates_custom(B, n, rng, v_sd=3.0, g_cats=(6,))
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
    elif kind == "theta_out":
        covs = sample_covariates(B, n, rng)
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
        theta[:, 0] = 14.0  # rate above 12
        theta[:, 1] = -4.5  # density below -4
        theta[:, 3] = 1.8  # transTrip above 1.5
    elif kind == "hidden_hom":
        covs = sample_covariates(B, n, rng)
        w = rng.integers(0, 3, size=(B, n)).astype(DTYPE)  # unobserved attribute
        gen_model = Model(M2_EFFECTS + [("sameX", "w")], {**covs, "w": w})
        model = m2_model(covs)
        X0, _ = sample_start_networks(B, n, rng, prior, model, backend=backend)
        hidden = np.full(B, 1.5)
    else:
        raise ValueError(kind)
    if kind == "hidden_hom":
        th_gen = np.column_stack([theta[:, 1:], hidden])
        X1 = simulate_period(X0, th_gen, theta[:, 0], gen_model, rng, backend=backend)
    else:
        X1 = simulate_period(X0, theta[:, 1:], theta[:, 0], model, rng, backend=backend)
    return theta, n, X0, X1, covs, model


# --------------------------------------------------------------- detector
class Detector:
    """max |z| of the transformed summaries against the training population,
    plus a robust variant using empirical percentiles (probit-transformed)."""

    def __init__(self, ref_T: np.ndarray):
        self.mean = ref_T.mean(axis=0)
        self.sd = ref_T.std(axis=0) + 1e-9
        self.sorted = np.sort(ref_T, axis=0)
        self.N = ref_T.shape[0]

    def maxz(self, T):
        return np.abs((T - self.mean) / self.sd).max(axis=1)

    def robust(self, T):
        from scipy.stats import norm

        out = np.empty(T.shape)
        for k in range(T.shape[1]):
            p = np.searchsorted(self.sorted[:, k], T[:, k]) / self.N
            p = np.clip(p, 0.5 / self.N, 1 - 0.5 / self.N)
            out[:, k] = norm.ppf(p)
        return np.abs(out).max(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_m2_10m.pt")
    ap.add_argument(
        "--ref",
        default="data/m2_shards/train_m2_s10.npz",
        help="training summaries for the detector",
    )
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--out", default="data/ood_m2")
    args = ap.parse_args()

    from sbi.diagnostics import run_sbc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior = m2_prior()
    posterior = torch.load(args.posterior, weights_only=False)
    probe = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    ref = load_m2_summaries(args.ref)
    names = ref.summary_names
    det = Detector(transform_m2(ref.summary, names, probe))
    report = []

    def log(msg=""):
        print(msg, flush=True)
        report.append(msg)

    log(f"estimator {args.posterior}; detector reference {ref.theta.shape[0]} training panels\n")
    kinds = ["id", "n_small", "n_large", "dense_x0", "covariates", "theta_out", "hidden_hom"]
    results = {}
    L = args.posterior_samples + 1
    for kind in kinds:
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        theta, n, X0, X1, covs, model = make_set(kind, args.B, rng, prior, args.backend)
        S = m2_summaries(X0, X1, model, covs)
        T = transform_m2(S, names, probe)
        xs = torch.as_tensor(T, dtype=torch.float32, device=device)
        # posterior samples for every panel (also gives SBC ranks)
        thetas = torch.as_tensor(theta, dtype=torch.float32, device=device)
        ranks, _ = run_sbc(
            thetas,
            xs,
            posterior,
            num_posterior_samples=args.posterior_samples,
            show_progress_bar=False,
        )
        ranks = ranks.cpu().numpy()
        post_sd = np.empty((args.B, theta.shape[1]))
        post_mean = np.empty_like(post_sd)
        edge = np.zeros(theta.shape[1])
        width = prior.high - prior.low
        for i in range(args.B):
            smp = posterior.sample((300,), x=xs[i : i + 1], show_progress_bars=False).cpu().numpy()
            post_sd[i] = smp.std(axis=0)
            post_mean[i] = smp.mean(axis=0)
            edge += ((smp < prior.low + 0.05 * width) | (smp > prior.high - 0.05 * width)).mean(
                axis=0
            )
        edge /= args.B
        results[kind] = dict(
            n=n,
            theta=theta,
            ranks=ranks,
            post_sd=post_sd,
            post_mean=post_mean,
            edge=edge,
            maxz=det.maxz(T),
            robust=det.robust(T),
            tie1=S[:, names.index("x1_tie_fraction")],
        )
        log(f"[{kind}] n={n}, {args.B} panels, {time.perf_counter() - t0:.0f}s")

    ctrl = results["id"]
    thr_z = np.quantile(ctrl["maxz"], 0.95)
    thr_r = np.quantile(ctrl["robust"], 0.95)
    sd_ref = ctrl["post_sd"].mean(axis=0)

    log("\n== detection: fraction flagged at the control's 95th percentile ==")
    log(f"{'set':<12}{'max|z| flagged':>15}{'robust flagged':>15}{'median max|z|':>15}")
    for kind in kinds:
        r = results[kind]
        fz, fr, med = np.mean(r["maxz"] > thr_z), np.mean(r["robust"] > thr_r), np.median(r["maxz"])
        log(f"{kind:<12}{fz:>15.3f}{fr:>15.3f}{med:>15.2f}")
    x0, x1, v, g, S50, _ = s50_as_m2()
    T50 = transform_m2(S50, names, probe)
    f50z = "yes" if det.maxz(T50)[0] > thr_z else "no"
    f50r = "yes" if det.robust(T50)[0] > thr_r else "no"
    log(f"{'s50 (real)':<12}{f50z:>15}{f50r:>15}{det.maxz(T50)[0]:>15.2f}")
    log(f"(thresholds: max|z| {thr_z:.2f}, robust {thr_r:.2f})")

    log("\n== posterior behaviour ==")
    pn = list(prior.names)
    for kind in kinds:
        r = results[kind]
        log(f"\n[{kind}]  n={r['n']}  mean X1 tie fraction {r['tie1'].mean():.3f}")
        if kind == "theta_out":
            log("  true theta outside the box (no calibration statement);")
            log("  mass within 5% of a box face:")
            log("  " + " ".join(f"{p}={e:.2f}" for p, e in zip(pn, r["edge"], strict=True)))
            log(
                "  posterior mean: "
                + " ".join(f"{p}={m:.2f}" for p, m in zip(pn, r["post_mean"].mean(0), strict=True))
            )
            continue
        if kind == "hidden_hom":
            log("  misspecified: bias of posterior mean vs generating theta, in posterior sd:")
            z = (r["post_mean"] - r["theta"]) / r["post_sd"]
            log("  " + " ".join(f"{p}={b:+.2f}" for p, b in zip(pn, z.mean(0), strict=True)))
            log(
                "  mass within 5% of a box face: "
                + " ".join(f"{p}={e:.2f}" for p, e in zip(pn, r["edge"], strict=True))
            )
            continue
        u = r["ranks"] / L
        log(
            "  mean rank (0.5): "
            + " ".join(f"{p}={m:.3f}" for p, m in zip(pn, u.mean(0), strict=True))
        )
        log(
            "  posterior sd / control sd: "
            + " ".join(
                f"{p}={s:.2f}" for p, s in zip(pn, r["post_sd"].mean(0) / sd_ref, strict=True)
            )
        )
        log(
            "  mass within 5% of a box face: "
            + " ".join(f"{p}={e:.2f}" for p, e in zip(pn, r["edge"], strict=True))
        )
        log(coverage_table(r["ranks"].astype(float), L, pn))

    np.savez_compressed(
        f"{args.out}.npz",
        **{
            f"{k}_{f}": v
            for k, r in results.items()
            for f, v in r.items()
            if isinstance(v, np.ndarray)
        },
        names=np.array(pn),
        thr_z=thr_z,
        thr_r=thr_r,
    )
    Path(f"{args.out}_report.txt").write_text("\n".join(report), encoding="utf-8")
    log(f"\nwrote {args.out}.npz, {args.out}_report.txt")


if __name__ == "__main__":
    main()
