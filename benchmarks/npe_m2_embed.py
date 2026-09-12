"""M2 with a learned panel embedding (saomsim.embedding.PanelEmbedding) in front of the flow.

    python benchmarks/npe_m2_embed.py --data data/train_m2.npz [--limit N] [--out data/npe_m2_embed]

Same population, prior and evaluation as npe_m2.py; the flow is conditioned
on a learned embedding of (X0, X1, v, g, n) with the 29 hand summaries
appended as a residual, instead of the summaries alone. Requires a training
file that kept networks. Evaluation: fresh-sample SBC across n (with
networks), coverage, and the s50 real-start posterior next to RSiena,
the M1 estimator and the summary-only M2 estimator.
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
from benchmarks.m2 import m2_prior, s50_as_m2  # noqa: E402
from saomsim.embedding import PanelEmbedding, pack_panel  # noqa: E402
from saomsim.population import (  # noqa: E402
    N_MAX,
    M2TrainingSet,
    generate_m2,
    m2_model,
    sample_covariates,
    transform_m2,
)


def packed_input(ts: M2TrainingSet, model) -> np.ndarray:
    S = transform_m2(ts.summary, ts.summary_names, model)
    return pack_panel(ts.X0, ts.X1, ts.v, ts.g, ts.n, S)


def s50_packed(model, names):
    x0, x1, v, g, S, _ = s50_as_m2()
    n = x0.shape[0]
    P0 = np.zeros((1, N_MAX, N_MAX), dtype=np.uint8)
    P1 = np.zeros_like(P0)
    P0[0, :n, :n], P1[0, :n, :n] = x0, x1
    vv = np.zeros((1, N_MAX), dtype=np.float32)
    gg = np.full((1, N_MAX), -1, dtype=np.int8)
    vv[0, :n], gg[0, :n] = v, g
    St = transform_m2(S, names, model)
    return pack_panel(np.packbits(P0, axis=-1), np.packbits(P1, axis=-1), vv, gg, np.array([n]), St)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_m2.npz")
    ap.add_argument("--out", default="data/npe_m2_embed")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sbc", type=int, default=4000, help="fresh SBC draws across n")
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--stop-after", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--transforms", type=int, default=8)
    ap.add_argument("--embed-hidden", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=64)
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
    prior_box = m2_prior()
    ts = M2TrainingSet.load(args.data)
    if ts.X0 is None:
        raise SystemExit("training file has no networks; regenerate without --no-networks")
    if args.limit:
        for k in ("theta", "summary", "n", "X0", "X1", "v", "g"):
            setattr(ts, k, getattr(ts, k)[: args.limit])
    N = ts.theta.shape[0]
    model = m2_model(sample_covariates(1, 20, np.random.default_rng(0)))
    t0 = time.perf_counter()
    x = packed_input(ts, model)
    pack_s = time.perf_counter() - t0
    log(f"population set {args.data}: N={N}, {x.shape[1]} floats/panel, packed in {pack_s:.0f}s")
    n_summ = ts.summary.shape[1]

    prior = BoxUniform(
        low=torch.as_tensor(prior_box.low, dtype=torch.float32),
        high=torch.as_tensor(prior_box.high, dtype=torch.float32),
        device=device,
    )

    # --------------------------------------------------------------- train
    emb = PanelEmbedding(n_summ, hidden=args.embed_hidden, out_dim=args.embed_dim)
    net = posterior_nn(
        model="nsf",
        hidden_features=args.hidden,
        num_transforms=args.transforms,
        embedding_net=emb,
        z_score_x="none",  # packed bytes must reach the embedding untouched
    )
    inference = NPE(prior=prior, density_estimator=net, device=device)
    inference.append_simulations(
        torch.as_tensor(ts.theta, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
    )
    del x
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
    n_par = sum(p.numel() for p in estimator.parameters())
    best = summ["best_validation_loss"][-1]
    log(
        f"trained nsf {args.transforms}x{args.hidden} + embedding ({args.embed_hidden} hidden, "
        f"{args.embed_dim} out; {n_par:,} params total) in {dt / 60:.1f} min, "
        f"{summ['epochs_trained'][-1]} epochs; best validation loss {best:.3f}"
    )
    np.savez_compressed(
        f"{out}_training.npz",
        training_loss=np.array(summ["training_loss"]),
        validation_loss=np.array(summ["validation_loss"]),
    )
    posterior = inference.build_posterior(estimator)
    torch.save(posterior, f"{out}.pt")

    # ------------------------------------------------ fresh SBC across n
    rng = np.random.default_rng(args.seed + 1000)
    t0 = time.perf_counter()
    fresh = generate_m2(prior_box, args.sbc, rng, chunk=100, backend="torch", keep_networks=True)
    xf = torch.as_tensor(packed_input(fresh, model), dtype=torch.float32, device=device)
    thetas = torch.as_tensor(fresh.theta, dtype=torch.float32, device=device)
    ranks, dap = run_sbc(
        thetas, xf, posterior, num_posterior_samples=args.posterior_samples, show_progress_bar=False
    )
    checks = check_sbc(ranks, thetas, dap, num_posterior_samples=args.posterior_samples)
    ranks_np = ranks.cpu().numpy()
    L = args.posterior_samples + 1
    np.savez_compressed(
        f"{out}_sbc_pop.npz",
        ranks=ranks_np,
        names=np.array(ts.theta_names),
        n=fresh.n,
        num_samples=np.array(L),
    )
    sbc_min = (time.perf_counter() - t0) / 60
    log(f"\nfresh SBC across n: {args.sbc} x {args.posterior_samples} samples ({sbc_min:.1f} min)")
    log(f"{'parameter':<12}{'KS p-value':>11}{'mean rank':>11}")
    for k, name in enumerate(ts.theta_names):
        log(f"{name:<12}{checks['ks_pvals'][k].item():>11.3f}{(ranks_np[:, k] / L).mean():>11.3f}")
    log("\n" + coverage_table(ranks_np.astype(float), L, ts.theta_names))
    log("\nmean rank/L by n band (0.5 ideal):")
    for lo in range(20, 81, 15):
        m = (fresh.n >= lo) & (fresh.n < lo + 15)
        if m.any():
            u = ranks_np[m] / L
            log(
                f"  n {lo:>2}..{min(lo + 14, 80):<2} (N={m.sum():>4}): "
                + " ".join(f"{nm}={u[:, k].mean():.3f}" for k, nm in enumerate(ts.theta_names))
            )
    fig, _ = sbc_rank_plot(
        ranks,
        args.posterior_samples,
        parameter_labels=list(ts.theta_names),
        plot_type="hist",
        num_bins=20,
    )
    fig.savefig(f"{out}_sbc_pop.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------- s50: real start
    x_obs = torch.as_tensor(s50_packed(model, ts.summary_names), dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    samples = posterior.sample((20_000,), x=x_obs, show_progress_bars=False).cpu().numpy()
    dt = time.perf_counter() - t0
    np.savez_compressed(f"{out}_posterior_s50.npz", samples=samples, names=np.array(ts.theta_names))
    q = np.quantile(samples, [0.05, 0.95], axis=0)
    rs_path = Path(__file__).parent / "rsiena_estimate.json"
    rs = json.loads(rs_path.read_text(encoding="utf-8")) if rs_path.exists() else None
    rs_names = {"altX(v)": "altX(alc)", "egoX(v)": "egoX(alc)", "sameX(g)": "sameX(smk)"}
    others = {}
    for tag, path in (
        ("M1", "data/npe_s50_posterior.npz"),
        ("M2sum", "data/npe_m2_posterior_s50.npz"),
    ):
        if Path(path).exists():
            others[tag] = np.load(path)["samples"]
    log(f"\nM2-embed posterior for s501 -> s502 ({dt:.2f}s):")
    hdr = f"{'parameter':<12}{'mean':>8}{'sd':>7}{'90%':>17}"
    if rs:
        hdr += f"{'RSiena':>8}{'z':>6}"
    for tag in others:
        hdr += f"{tag + ' mean':>11}{tag + ' sd':>9}"
    log(hdr)
    for k, name in enumerate(ts.theta_names):
        m, sd = samples[:, k].mean(), samples[:, k].std()
        line = f"{name:<12}{m:>8.3f}{sd:>7.3f}{f'[{q[0, k]:.2f}, {q[1, k]:.2f}]':>17}"
        if rs:
            e = rs["estimate"][rs_names.get(name, name)]
            line += f"{e:>8.3f}{(e - m) / sd:>6.2f}"
        for s_ in others.values():
            line += f"{s_[:, k].mean():>11.3f}{s_[:, k].std():>9.3f}"
        log(line)

    Path(f"{out}_report.txt").write_text("\n".join(report), encoding="utf-8")
    log(f"\nwrote {out}.pt, {out}_sbc_pop.{{npz,png}}, {out}_posterior_s50.npz, {out}_report.txt")


if __name__ == "__main__":
    main()
