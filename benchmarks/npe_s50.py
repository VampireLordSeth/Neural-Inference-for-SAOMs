"""Neural posterior estimation for the s50 case, with SBC.

    python benchmarks/npe_s50.py --data data/train_s50.npz [--limit N] [--sbc 1000]

Steps (GETTING_STARTED §7 items 5 and 6):
  1. load the training set, transform summaries (log1p counts / asinh signed)
  2. hold out the last ``--sbc`` draws for calibration, train a neural spline
     flow q(theta | x) with sbi's NPE on the rest (GPU)
  3. posterior for the observed s501 -> s502 summaries; compare with the
     Robbins-Monro point estimate on the same data (conditional, so no rate)
  4. simulation-based calibration on the held-out draws: rank statistics,
     KS p-values per parameter, C2ST of ranks vs uniform, rank histograms

Everything written to ``--out`` + suffix: ``.pt`` (posterior), ``_posterior.npz``
(samples), ``_sbc.npz`` (ranks), ``_sbc.png``, ``_report.txt``.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.s50 import load_s50, s50_prior  # noqa: E402
from saomsim import estimate_rm  # noqa: E402
from saomsim.prior import TrainingSet, summaries, transform_summaries  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_s50.npz")
    ap.add_argument("--out", default="data/npe_s50")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N draws")
    ap.add_argument("--sbc", type=int, default=1000, help="held-out draws for SBC")
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--stop-after", type=int, default=20)
    ap.add_argument("--estimator", default="nsf")
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--transforms", type=int, default=6)
    ap.add_argument("--skip-rm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sbi.analysis import sbc_rank_plot
    from sbi.diagnostics import check_sbc, run_sbc
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = []

    def log(msg=""):
        print(msg, flush=True)
        report.append(msg)

    # ---------------------------------------------------------------- data
    x0, x1, model = load_s50()
    prior_box = s50_prior(model)
    ts = TrainingSet.load(args.data)
    if args.limit:
        ts.theta, ts.summary = ts.theta[: args.limit], ts.summary[: args.limit]
    N = ts.theta.shape[0]
    X = transform_summaries(ts.summary, ts.summary_names, model)
    n_train = N - args.sbc
    theta_tr, x_tr = ts.theta[:n_train], X[:n_train]
    theta_te, x_te = ts.theta[n_train:], X[n_train:]
    log(f"training set {args.data}: N={N}, train={n_train}, sbc={args.sbc}, device={device}")
    ts.theta_names = [str(v) for v in ts.theta_names]
    ts.summary_names = [str(v) for v in ts.summary_names]
    log(f"theta: {ts.theta_names}")
    log(f"summaries ({len(ts.summary_names)}): {ts.summary_names}")

    prior = BoxUniform(
        low=torch.as_tensor(prior_box.low, dtype=torch.float32),
        high=torch.as_tensor(prior_box.high, dtype=torch.float32),
        device=device,
    )

    # --------------------------------------------------------------- train
    net = posterior_nn(
        model=args.estimator, hidden_features=args.hidden, num_transforms=args.transforms
    )
    inference = NPE(prior=prior, density_estimator=net, device=device)
    inference.append_simulations(
        torch.as_tensor(theta_tr, dtype=torch.float32), torch.as_tensor(x_tr, dtype=torch.float32)
    )
    t0 = time.perf_counter()
    estimator = inference.train(
        training_batch_size=args.batch,
        learning_rate=args.lr,
        max_num_epochs=args.max_epochs,
        stop_after_epochs=args.stop_after,
        show_train_summary=False,
    )
    dt = time.perf_counter() - t0
    summ = inference.summary
    arch = f"{args.estimator} (hidden={args.hidden}, transforms={args.transforms})"
    best = summ["best_validation_loss"][-1]
    log(
        f"trained {arch} in {dt / 60:.1f} min, {summ['epochs_trained'][-1]} epochs; "
        f"best validation loss (neg. log-prob) {best:.3f}"
    )
    np.savez_compressed(
        f"{out}_training.npz",
        training_loss=np.array(summ["training_loss"]),
        validation_loss=np.array(summ["validation_loss"]),
    )
    posterior = inference.build_posterior(estimator)
    torch.save(posterior, f"{out}.pt")

    # ------------------------------------------------- observed s501 -> s502
    s_obs = summaries(x0[None], x1[None], model)
    x_obs = torch.as_tensor(
        transform_summaries(s_obs, ts.summary_names, model), dtype=torch.float32
    )
    t0 = time.perf_counter()
    samples = (
        posterior.sample((20_000,), x=x_obs.to(device), show_progress_bars=False).cpu().numpy()
    )
    dt = time.perf_counter() - t0
    np.savez_compressed(f"{out}_posterior.npz", samples=samples, names=np.array(ts.theta_names))
    q = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)
    log(f"\nposterior for s501 -> s502 (20,000 samples in {dt:.2f}s):")
    log(f"{'parameter':<12}{'mean':>9}{'sd':>8}{'q05':>9}{'median':>9}{'q95':>9}   prior")
    for k, name in enumerate(ts.theta_names):
        log(
            f"{name:<12}{samples[:, k].mean():>9.3f}{samples[:, k].std():>8.3f}"
            f"{q[0, k]:>9.3f}{q[1, k]:>9.3f}{q[2, k]:>9.3f}"
            f"   U({prior_box.low[k]:g}, {prior_box.high[k]:g})"
        )
    at_edge = [
        name
        for k, name in enumerate(ts.theta_names)
        if q[0, k] < prior_box.low[k] + 0.02 * (prior_box.high[k] - prior_box.low[k])
        or q[2, k] > prior_box.high[k] - 0.02 * (prior_box.high[k] - prior_box.low[k])
    ]
    log(f"posterior 90% interval touches the prior box: {at_edge or 'none'}")

    # ------------------------------------------------------- RSiena baseline
    rs_path = Path(__file__).parent / "rsiena_estimate.json"
    if rs_path.exists():
        import json

        rs = json.loads(rs_path.read_text(encoding="utf-8"))
        log(f"\nRSiena {rs['rsiena_version']} MoM (cond=FALSE, {rs['seconds']:.0f}s) vs NPE:")
        hdr = f"{'parameter':<12}{'RSiena':>9}{'se':>8}{'NPE mean':>10}{'NPE sd':>8}"
        log(hdr + f"{'(RS-NPE)/sd':>13}{'in NPE 90%':>12}")
        for k, name in enumerate(ts.theta_names):
            est, se = rs["estimate"][name], rs["se"][name]
            m, sd = samples[:, k].mean(), samples[:, k].std()
            covers = "yes" if q[0, k] <= est <= q[2, k] else "no"
            z = (est - m) / sd
            log(f"{name:<12}{est:>9.3f}{se:>8.3f}{m:>10.3f}{sd:>8.3f}{z:>13.2f}{covers:>12}")

    # ------------------------------------------------- Robbins-Monro baseline
    if not args.skip_rm:
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        rm = estimate_rm(x0, x1, model, rng)
        if not rm.converged:
            rm = estimate_rm(x0, x1, model, rng, theta0=rm.theta)
        dt = time.perf_counter() - t0
        log(f"\nRobbins-Monro on the same data ({dt:.0f}s, converged={rm.converged}), vs NPE:")
        hdr = f"{'parameter':<12}{'RM est':>9}{'RM se':>8}{'NPE mean':>10}{'NPE sd':>8}"
        log(hdr + f"{'(RM-NPE)/sd':>13}")
        for k, name in enumerate(rm.names):
            j = ts.theta_names.index(name)
            m, s = samples[:, j].mean(), samples[:, j].std()
            z = (rm.theta[k] - m) / s
            log(f"{name:<12}{rm.theta[k]:>9.3f}{rm.se[k]:>8.3f}{m:>10.3f}{s:>8.3f}{z:>13.2f}")
        log("(RM is conditional on the observed distance, so it has no rate estimate.)")

    # ------------------------------------------------------------------ SBC
    if args.sbc > 0:
        t0 = time.perf_counter()
        thetas = torch.as_tensor(theta_te, dtype=torch.float32, device=device)
        xs = torch.as_tensor(x_te, dtype=torch.float32, device=device)
        ranks, dap = run_sbc(
            thetas,
            xs,
            posterior,
            num_posterior_samples=args.posterior_samples,
            show_progress_bar=False,
        )
        checks = check_sbc(ranks, thetas, dap, num_posterior_samples=args.posterior_samples)
        dt = time.perf_counter() - t0
        np.savez_compressed(
            f"{out}_sbc.npz", ranks=ranks.cpu().numpy(), names=np.array(ts.theta_names)
        )
        log(
            f"\nSBC on {args.sbc} held-out draws x {args.posterior_samples} posterior samples "
            f"({dt / 60:.1f} min):"
        )
        log(f"{'parameter':<12}{'KS p-value':>11}{'C2ST ranks':>12}")
        for k, name in enumerate(ts.theta_names):
            log(
                f"{name:<12}{checks['ks_pvals'][k].item():>11.3f}{checks['c2st_ranks'][k].item():>12.3f}"
            )
        log(f"C2ST data-averaged posterior vs prior: {checks['c2st_dap'].mean().item():.3f}")
        log(
            "(calibrated: KS p not small, C2ST near 0.5; "
            "C2ST dap near 0.5 means the posterior averages back to the prior)"
        )
        fig, _ = sbc_rank_plot(
            ranks,
            args.posterior_samples,
            parameter_labels=list(ts.theta_names),
            plot_type="hist",
            num_bins=20,
        )
        fig.savefig(f"{out}_sbc.png", dpi=120, bbox_inches="tight")

    Path(f"{out}_report.txt").write_text("\n".join(report), encoding="utf-8")
    log(f"\nwrote {out}.pt, {out}_posterior.npz, {out}_sbc.npz, {out}_sbc.png, {out}_report.txt")


if __name__ == "__main__":
    main()
