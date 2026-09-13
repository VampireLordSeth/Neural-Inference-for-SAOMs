"""SBC and coverage of a saved M2 posterior on a fresh, independent population sample.

    python benchmarks/sbc_m2.py --posterior data/npe_m2.pt --N 4000 [--n 50] [--seed 3]

A fresh test set is simulated here (never seen in training), with a small
chunk so many distinct n are covered — or a single ``--n`` to check
calibration at one size. Reports KS p-values, C2ST, coverage of central
intervals, mean rank by size band, and saves ranks + a rank histogram.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.coverage import coverage_table  # noqa: E402
from benchmarks.m2 import m2_prior  # noqa: E402
from saomsim.population import generate_m2, m2_model, sample_covariates, transform_m2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", default="data/npe_m2.pt")
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--n", type=int, default=None, help="fix the network size")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--out", default=None)
    ap.add_argument("--waves", type=int, default=2)
    args = ap.parse_args()

    from sbi.analysis import sbc_rank_plot
    from sbi.diagnostics import check_sbc, run_sbc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"_n{args.n}" if args.n else "_pop"
    out = Path(args.out or (str(Path(args.posterior).with_suffix("")) + f"_sbc{tag}"))

    prior = m2_prior(args.waves)
    rng = np.random.default_rng(args.seed)
    n_range = (args.n, args.n) if args.n else (20, 80)
    t0 = time.perf_counter()
    ts = generate_m2(
        prior,
        args.N,
        rng,
        n_range=n_range,
        chunk=args.chunk,
        backend=args.backend,
        keep_networks=False,
        waves=args.waves,
    )
    gen_s = time.perf_counter() - t0
    print(f"fresh test set: {args.N} panels, n in [{ts.n.min()}, {ts.n.max()}], {gen_s:.0f}s")
    model = m2_model(sample_covariates(1, 20, rng))
    X = transform_m2(ts.summary, ts.summary_names, model)

    posterior = torch.load(args.posterior, weights_only=False)
    thetas = torch.as_tensor(ts.theta, dtype=torch.float32, device=device)
    xs = torch.as_tensor(X, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    ranks, dap = run_sbc(
        thetas, xs, posterior, num_posterior_samples=args.posterior_samples, show_progress_bar=False
    )
    checks = check_sbc(ranks, thetas, dap, num_posterior_samples=args.posterior_samples)
    ranks_np = ranks.cpu().numpy()
    L = args.posterior_samples + 1
    sbc_s = time.perf_counter() - t0
    print(f"SBC: {args.N} draws x {args.posterior_samples} samples in {sbc_s:.0f}s\n")
    print(f"{'parameter':<12}{'KS p-value':>11}{'C2ST ranks':>12}{'mean rank':>11}")
    for k, name in enumerate(ts.theta_names):
        print(
            f"{name:<12}{checks['ks_pvals'][k].item():>11.3f}{checks['c2st_ranks'][k].item():>12.3f}"
            f"{(ranks_np[:, k] / L).mean():>11.3f}"
        )
    print(f"C2ST data-averaged posterior vs prior: {checks['c2st_dap'].mean().item():.3f}")
    print("\n" + coverage_table(ranks_np.astype(float), L, ts.theta_names))
    if not args.n:
        print("\nmean rank/L by n band (0.5 ideal; se ~ 0.29/sqrt(N_band)):")
        for lo in range(20, 81, 15):
            m = (ts.n >= lo) & (ts.n < lo + 15)
            if m.any():
                u = ranks_np[m] / L
                print(
                    f"  n {lo:>2}..{min(lo + 14, 80):<2} (N={m.sum():>4}): "
                    + " ".join(f"{nm}={u[:, k].mean():.3f}" for k, nm in enumerate(ts.theta_names))
                )
    np.savez_compressed(
        f"{out}.npz",
        ranks=ranks_np,
        names=np.array(ts.theta_names),
        n=ts.n,
        num_samples=np.array(L),
    )
    fig, _ = sbc_rank_plot(
        ranks,
        args.posterior_samples,
        parameter_labels=list(ts.theta_names),
        plot_type="hist",
        num_bins=20,
    )
    fig.savefig(f"{out}.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}.npz, {out}.png")


if __name__ == "__main__":
    main()
