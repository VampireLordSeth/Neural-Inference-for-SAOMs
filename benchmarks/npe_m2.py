"""M2: neural posterior estimation over the population of starts, with SBC and the s50 test.

  python benchmarks/npe_m2.py --data "data/m2_shards/*.npz" [--limit N] [--sbc 2000]

1. load the population training set, transform the 29 summaries
2. hold out the last ``--sbc`` draws; train an NSF q(theta | summaries) with sbi
3. SBC + coverage over the population (all n, all starts)
4. posterior for the real s50 panel (n = 50, real start s501, alc -> v, smk -> g);
   compare with RSiena (benchmarks/rsiena_estimate.json) and with the M1
   estimator's posterior (data/npe_s50_posterior.npz) if present
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.coverage import coverage_table  # noqa: E402
from benchmarks.m2 import glasgow_as_m2, s50_as_m2  # noqa: E402
from saomsim.population import (  # noqa: E402
    load_m2_summaries,
    summary_subset,
    transform_m2,
)
from saomsim.prior import BoxPrior  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_m2.npz", help="path or glob (shards)")
    ap.add_argument("--out", default="data/npe_m2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sbc", type=int, default=2000)
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--stop-after", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--transforms", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-max", type=float, default=None, help="upper rate prior (default 12)")
    ap.add_argument("--real", default="s50", choices=["s50", "glasgow"])
    ap.add_argument(
        "--summaries", default="full", choices=["full", "mom"],
        help="conditioning set: all summaries, or only the MoM targets plus x0 and n"
    )
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
    ts = load_m2_summaries(args.data)
    waves = int(ts.meta.get("waves", 2))
    # the training set carries its own box (rate range, and "default" or "sparse" effects);
    # rebuilding it here from defaults would silently train against the wrong prior
    prior_box = BoxPrior(
        tuple(ts.theta_names),
        np.asarray(ts.meta["prior_low"], dtype=float),
        np.asarray(ts.meta["prior_high"], dtype=float),
    )
    if args.rate_max is not None and not np.isclose(prior_box.high[0], args.rate_max):
        raise SystemExit(
            f"--rate-max {args.rate_max} but the training set has {prior_box.high[0]}"
        )
    if args.limit:
        ts.theta, ts.summary, ts.n = (
            ts.theta[: args.limit],
            ts.summary[: args.limit],
            ts.n[: args.limit],
        )
    N = ts.theta.shape[0]
    x0, x1, v, g, s_obs, model = (s50_as_m2 if args.real == "s50" else glasgow_as_m2)(waves)
    keep = summary_subset(ts.summary_names, args.summaries)
    X = transform_m2(ts.summary, ts.summary_names, model)[:, keep]
    log(f"summaries: {args.summaries} ({len(keep)} of {len(ts.summary_names)})")
    # random hold-out: consecutive rows share one n (one n per chunk), so a tail
    # split would test a single network size
    perm = np.random.default_rng(args.seed).permutation(N)
    te, tr = perm[: args.sbc], perm[args.sbc :]
    n_train = tr.size
    theta_tr, x_tr = ts.theta[tr], X[tr]
    theta_te, x_te, n_te = ts.theta[te], X[te], ts.n[te]
    log(f"population set {args.data}: N={N}, train={n_train}, sbc={args.sbc}, device={device}")
    log(f"waves={waves}; n in [{ts.n.min()}, {ts.n.max()}]; {len(ts.summary_names)} summaries")
    log(f"theta {ts.theta_names}")

    prior = BoxUniform(
        low=torch.as_tensor(prior_box.low, dtype=torch.float32),
        high=torch.as_tensor(prior_box.high, dtype=torch.float32),
        device=device,
    )

    # --------------------------------------------------------------- train
    net = posterior_nn(model="nsf", hidden_features=args.hidden, num_transforms=args.transforms)
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
    best = summ["best_validation_loss"][-1]
    log(
        f"trained nsf (hidden={args.hidden}, transforms={args.transforms}) in {dt / 60:.1f} min, "
        f"{summ['epochs_trained'][-1]} epochs; best validation loss {best:.3f}"
    )
    np.savez_compressed(
        f"{out}_training.npz",
        training_loss=np.array(summ["training_loss"]),
        validation_loss=np.array(summ["validation_loss"]),
    )
    posterior = inference.build_posterior(estimator)
    torch.save(posterior, f"{out}.pt")

    # ------------------------------------------------------------------ SBC
    thetas = torch.as_tensor(theta_te, dtype=torch.float32, device=device)
    xs = torch.as_tensor(x_te, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    ranks, dap = run_sbc(
        thetas, xs, posterior, num_posterior_samples=args.posterior_samples, show_progress_bar=False
    )
    checks = check_sbc(ranks, thetas, dap, num_posterior_samples=args.posterior_samples)
    ranks_np = ranks.cpu().numpy()
    np.savez_compressed(
        f"{out}_sbc.npz",
        ranks=ranks_np,
        names=np.array(ts.theta_names),
        n=n_te,
        num_samples=np.array(args.posterior_samples + 1),
    )
    sbc_min = (time.perf_counter() - t0) / 60
    log(f"\nSBC over the population, {args.sbc} held-out draws ({sbc_min:.1f} min):")
    log(f"{'parameter':<12}{'KS p-value':>11}{'C2ST ranks':>12}")
    for k, name in enumerate(ts.theta_names):
        log(
            f"{name:<12}{checks['ks_pvals'][k].item():>11.3f}{checks['c2st_ranks'][k].item():>12.3f}"
        )
    log(f"C2ST data-averaged posterior vs prior: {checks['c2st_dap'].mean().item():.3f}")
    log("\ncoverage of central intervals:")
    log(coverage_table(ranks_np.astype(float), args.posterior_samples + 1, ts.theta_names))
    # calibration by size band
    log("\nmean rank/L by n band (0.5 ideal):")
    u = ranks_np / (args.posterior_samples + 1)
    n_hi = int(n_te.max())
    step = 20 if n_hi <= 100 else 30
    for lo in range(20, n_hi + 1, step):
        m = (n_te >= lo) & (n_te < lo + step)
        if m.any():
            log(
                f"  n {lo:>3}..{min(lo + step - 1, n_hi):<3} (N={m.sum():>4}): "
                + " ".join(f"{name}={u[m, k].mean():.3f}" for k, name in enumerate(ts.theta_names))
            )
    fig, _ = sbc_rank_plot(
        ranks,
        args.posterior_samples,
        parameter_labels=list(ts.theta_names),
        plot_type="hist",
        num_bins=20,
    )
    fig.savefig(f"{out}_sbc.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------- s50: real start
    x_obs = torch.as_tensor(
        transform_m2(s_obs, ts.summary_names, model)[:, keep], dtype=torch.float32, device=device
    )
    t0 = time.perf_counter()
    samples = posterior.sample((20_000,), x=x_obs, show_progress_bars=False).cpu().numpy()
    dt = time.perf_counter() - t0
    np.savez_compressed(f"{out}_posterior_s50.npz", samples=samples, names=np.array(ts.theta_names))
    q = np.quantile(samples, [0.05, 0.5, 0.95], axis=0)
    tag = f"{args.real}, {waves} waves, n={x0.shape[0]}"
    log(f"\nM2 posterior for {tag} (real start, never seen in training; {dt:.2f}s):")
    if args.real == "glasgow":
        rs_path = Path(__file__).parent / "glasgow" / "rsiena_network3w.json"
    else:
        rs_file = "rsiena_estimate.json" if waves == 2 else f"rsiena_estimate_{waves}w.json"
        rs_path = Path(__file__).parent / rs_file
    rs = json.loads(rs_path.read_text(encoding="utf-8")) if rs_path.exists() else None
    m1_path = Path("data/npe_s50_posterior.npz")
    m1 = np.load(m1_path) if (m1_path.exists() and waves == 2) else None
    hdr = f"{'parameter':<12}{'M2 mean':>9}{'M2 sd':>8}{'M2 90%':>18}"
    if rs:
        hdr += f"{'RSiena':>9}{'se':>7}{'z':>6}"
    if m1 is not None:
        hdr += f"{'M1 mean':>9}{'M1 sd':>7}"
    log(hdr)
    rs_names = {"altX(v)": "altX(alc)", "egoX(v)": "egoX(alc)", "sameX(g)": "sameX(smk)"}
    if args.real == "glasgow":  # labels from benchmarks/glasgow/prepare_and_fit.R
        rs_names = {nm: f"net:{nm.split('(')[0]}" for nm in ts.theta_names}
        rs_names.update({"rate_1": "net:rate_1", "rate_2": "net:rate_2", "rate": "net:rate_1"})
    for k, name in enumerate(ts.theta_names):
        m, sd = samples[:, k].mean(), samples[:, k].std()
        line = f"{name:<12}{m:>9.3f}{sd:>8.3f}{f'[{q[0, k]:.2f}, {q[2, k]:.2f}]':>18}"
        if rs:
            e, se = rs["estimate"][rs_names.get(name, name)], rs["se"][rs_names.get(name, name)]
            line += f"{e:>9.3f}{se:>7.3f}{(e - m) / sd:>6.2f}"
        if m1 is not None:
            s1 = m1["samples"][:, k]
            line += f"{s1.mean():>9.3f}{s1.std():>7.3f}"
        log(line)

    Path(f"{out}_report.txt").write_text("\n".join(report), encoding="utf-8")
    log(f"\nwrote {out}.pt, {out}_sbc.{{npz,png}}, {out}_posterior_s50.npz, {out}_report.txt")


if __name__ == "__main__":
    main()
