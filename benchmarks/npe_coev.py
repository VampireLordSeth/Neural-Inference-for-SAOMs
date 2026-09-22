"""M3b: generate, train, calibrate and test the co-evolution estimator.

    python benchmarks/npe_coev.py generate --N 1000000 --waves 3 --out data/train_coev3.npz
    python benchmarks/npe_coev.py train --data data/train_coev3.npz --out data/npe_coev3
    python benchmarks/npe_coev.py sbc --posterior data/npe_coev3.pt --waves 3 --N 4000
    python benchmarks/npe_coev.py s50 --posterior data/npe_coev3.pt [--real glasgow]

`s50` applies the estimator to s501 -> s502 -> s503 with alcohol, using
RSiena's all-wave constants, and compares with benchmarks/rsiena_coevolution_estimate.json;
`--real glasgow` does the same for the 129-pupil Glasgow panel (benchmarks/glasgow/).
`generate` and `sbc` take --n-max, --rate-net-max and --start sparse (M4, docs/PRIORS_M4.md).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.coverage import coverage_table, sbc_ranks_chunked  # noqa: E402
from saomsim.population_coev import (  # noqa: E402
    CoevTrainingSet,
    coev_prior,
    generate_coev,
    net_model,
    real_coev_summary,
    transform_coev,
)


def rs_names(beh: str = "alc") -> dict:
    """Our parameter names -> RSiena labels as the R scripts write them (behaviour ``beh``)."""
    return {
        "rate_net_1": "net:rate_1",
        "rate_net_2": "net:rate_2",
        "rate_beh_1": f"{beh}:rate_1",
        "rate_beh_2": f"{beh}:rate_2",
        "density": "net:density",
        "recip": "net:recip",
        "transTrip": "net:transTrip",
        "cycle3": "net:cycle3",
        "egoZ": "net:egoX",
        "altZ": "net:altX",
        "simZ": "net:simX",
        "linear": f"{beh}:linear",
        "quad": f"{beh}:quad",
        "avAlt": f"{beh}:avAlt",
    }


REAL = {  # name -> (network csv pattern, behaviour csv, RSiena json, RSiena behaviour label)
    "s50": ("s50{w}.csv", "s50a.csv", "rsiena_coevolution_estimate.json", "alc"),
    "glasgow": (
        "glasgow/glasgow_net{w}.csv",
        "glasgow/glasgow_alcohol.csv",
        "glasgow/rsiena_coevolution.json",
        "alcB",
    ),
}


def torch_backend(dtype="float32"):
    import torch

    from saomsim.backend_torch import TorchBackend

    return TorchBackend(dtype=getattr(torch, dtype))


def cmd_generate(a):
    prior = coev_prior(a.waves, rate_net=(1.0, a.rate_net_max))
    print(
        f"N={a.N} waves={a.waves} n<={a.n_max} start={a.start}\nprior:\n{prior.table()}", flush=True
    )
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    ts = generate_coev(
        prior,
        a.N,
        rng,
        waves=a.waves,
        n_range=(a.n_min, a.n_max),
        chunk=a.chunk,
        backend=torch_backend(),
        keep_networks=not a.no_networks,
        progress=True,
        start=a.start,
    )
    dt = time.perf_counter() - t0
    print(f"simulated {a.N} panels in {dt:.1f}s ({a.N / dt:,.0f} panels/s)")
    ts.meta["seed"] = a.seed
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ts.save(a.out)
    print(f"saved {a.out} ({Path(a.out).stat().st_size / 1e6:,.0f} MB)")


def load_summaries(paths):
    import glob

    files = sorted(sum((glob.glob(p) or [p] for p in paths), []))
    parts = [CoevTrainingSet.load(f) for f in files]
    ts = parts[0]
    if len(parts) > 1:
        ts.theta = np.concatenate([p.theta for p in parts])
        ts.summary = np.concatenate([p.summary for p in parts])
        ts.n = np.concatenate([p.n for p in parts])
        ts.z_max = np.concatenate([p.z_max for p in parts])
        ts.X = ts.z = None
    return ts


def cmd_train(a):
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = load_summaries(a.data)
    waves = int(ts.meta["waves"])
    prior_box = coev_prior(waves, rate_net=(ts.meta["prior_low"][0], ts.meta["prior_high"][0]))
    assert list(prior_box.names) == ts.theta_names
    assert np.allclose(prior_box.low, ts.meta["prior_low"]) and np.allclose(
        prior_box.high, ts.meta["prior_high"]
    ), "training set was generated with a different prior box"
    X = transform_coev(ts.summary, ts.summary_names, net_model())
    N = X.shape[0]
    print(f"N={N} waves={waves} summaries={X.shape[1]} device={device}", flush=True)
    prior = BoxUniform(
        low=torch.as_tensor(prior_box.low, dtype=torch.float32),
        high=torch.as_tensor(prior_box.high, dtype=torch.float32),
        device=device,
    )
    net = posterior_nn(model="nsf", hidden_features=a.hidden, num_transforms=a.transforms)
    inf = NPE(prior=prior, density_estimator=net, device=device)
    inf.append_simulations(
        torch.as_tensor(ts.theta, dtype=torch.float32), torch.as_tensor(X, dtype=torch.float32)
    )
    t0 = time.perf_counter()
    est = inf.train(
        training_batch_size=a.batch,
        learning_rate=a.lr,
        max_num_epochs=a.max_epochs,
        stop_after_epochs=a.stop_after,
        show_train_summary=False,
    )
    summ = inf.summary
    best = summ["best_validation_loss"][-1]
    print(
        f"trained nsf {a.transforms}x{a.hidden} in {(time.perf_counter() - t0) / 60:.1f} min, "
        f"{summ['epochs_trained'][-1]} epochs; best validation loss {best:.3f}"
    )
    posterior = inf.build_posterior(est)
    torch.save(posterior, f"{a.out}.pt")
    np.savez_compressed(
        f"{a.out}_training.npz",
        training_loss=np.array(summ["training_loss"]),
        validation_loss=np.array(summ["validation_loss"]),
    )
    print(f"wrote {a.out}.pt")


def cmd_sbc(a):
    import torch
    from sbi.analysis import sbc_rank_plot
    from sbi.diagnostics import check_sbc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior = coev_prior(a.waves, rate_net=(1.0, a.rate_net_max))
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    ts = generate_coev(
        prior,
        a.N,
        rng,
        waves=a.waves,
        n_range=(a.n_min, a.n_max),
        chunk=a.chunk,
        backend=torch_backend(),
        keep_networks=False,
        start=a.start,
    )
    gen_s = time.perf_counter() - t0
    print(
        f"fresh test set: {a.N} panels, n in [{ts.n.min()}, {ts.n.max()}], {gen_s:.0f}s", flush=True
    )
    X = transform_coev(ts.summary, ts.summary_names, net_model())
    posterior = torch.load(a.posterior, weights_only=False)
    thetas = torch.as_tensor(ts.theta, dtype=torch.float32, device=device)
    xs = torch.as_tensor(X, dtype=torch.float32, device=device)
    ranks, dap = sbc_ranks_chunked(
        posterior,
        thetas,
        xs,
        a.posterior_samples,
        chunk=a.chunk,
        log=lambda m: print(m, flush=True),
    )
    checks = check_sbc(ranks, thetas, dap, num_posterior_samples=a.posterior_samples)
    r = ranks.cpu().numpy()
    L = a.posterior_samples + 1
    print(f"\n{'parameter':<12}{'KS p-value':>11}{'mean rank':>11}")
    for k, name in enumerate(ts.theta_names):
        print(f"{name:<12}{checks['ks_pvals'][k].item():>11.3f}{(r[:, k] / L).mean():>11.3f}")
    print("\n" + coverage_table(r.astype(float), L, ts.theta_names))
    print("\nmean rank/L by n band:")
    step = 15 if a.n_max <= 80 else 30
    for lo in range(20, a.n_max + 1, step):
        m = (ts.n >= lo) & (ts.n < lo + step)
        if m.any():
            u = r[m] / L
            print(
                f"  n {lo:>3}..{min(lo + step - 1, a.n_max):<3} (N={m.sum():>4}): "
                + " ".join(f"{nm}={u[:, k].mean():.3f}" for k, nm in enumerate(ts.theta_names))
            )
    out = str(Path(a.posterior).with_suffix("")) + "_sbc_pop"
    np.savez_compressed(
        f"{out}.npz", ranks=r, names=np.array(ts.theta_names), n=ts.n, num_samples=np.array(L)
    )
    fig, _ = sbc_rank_plot(
        ranks,
        a.posterior_samples,
        parameter_labels=list(ts.theta_names),
        plot_type="hist",
        num_bins=20,
    )
    fig.savefig(f"{out}.png", dpi=120, bbox_inches="tight")
    print(f"wrote {out}.npz, {out}.png")


def cmd_s50(a):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    here = Path(__file__).parent
    net_pat, beh_csv, rs_json, beh_label = REAL[a.real]
    Xs = [np.loadtxt(here / net_pat.format(w=w), delimiter=",", dtype=np.int8) for w in (1, 2, 3)]
    Z = np.loadtxt(here / beh_csv, delimiter=",")
    S, spec = real_coev_summary(Xs, [Z[:, w] for w in range(3)], 1, 5)
    names = coev_prior(3).names
    labels = rs_names(beh_label)
    from saomsim.population_coev import coev_summary_names

    x = torch.as_tensor(
        transform_coev(S, coev_summary_names(3), net_model()), dtype=torch.float32, device=device
    )
    posterior = torch.load(a.posterior, weights_only=False)
    t0 = time.perf_counter()
    smp = posterior.sample((20_000,), x=x, show_progress_bars=False).cpu().numpy()
    dt = time.perf_counter() - t0
    np.savez_compressed(
        str(Path(a.posterior).with_suffix("")) + f"_posterior_{a.real}.npz",
        samples=smp,
        names=np.array(names),
    )
    rs_path = here / rs_json
    rs = json.loads(rs_path.read_text(encoding="utf-8")) if rs_path.exists() else None
    q = np.quantile(smp, [0.05, 0.95], axis=0)
    consts = f"zbar={spec.zbar[0]:.3f}, simMean={spec.sim_mean[0]:.3f}"
    print(
        f"co-evolution posterior for {a.real}, n={Xs[0].shape[0]} "
        f"(three waves, alcohol; {consts}; {dt:.2f}s)"
    )
    hdr = f"{'parameter':<12}{'mean':>8}{'sd':>7}{'90%':>17}"
    if rs:
        hdr += f"{'RSiena':>8}{'se':>7}{'z':>6}{'in 90%':>8}"
    print(hdr)
    for k, nm in enumerate(names):
        m, s = smp[:, k].mean(), smp[:, k].std()
        line = f"{nm:<12}{m:>8.3f}{s:>7.3f}{f'[{q[0, k]:.2f}, {q[1, k]:.2f}]':>17}"
        if rs:
            e, se = rs["estimate"][labels[nm]], rs["se"][labels[nm]]
            inside = "yes" if q[0, k] <= e <= q[1, k] else "no"
            line += f"{e:>8.3f}{se:>7.3f}{(e - m) / s:>6.2f}{inside:>8}"
        print(line)
    C = np.corrcoef(smp.T)
    i, j = names.index("simZ"), names.index("avAlt")
    print(f"\nposterior correlation selection (simZ) x influence (avAlt): {C[i, j]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--N", type=int, default=1_000_000)
    g.add_argument("--waves", type=int, default=3)
    g.add_argument("--out", default="data/train_coev3.npz")
    g.add_argument("--seed", type=int, default=50)
    g.add_argument("--chunk", type=int, default=2048)
    g.add_argument("--no-networks", action="store_true")
    g.add_argument("--n-min", type=int, default=20)
    g.add_argument("--n-max", type=int, default=80)
    g.add_argument("--rate-net-max", type=float, default=12.0)
    g.add_argument("--start", default="m2", choices=["m2", "sparse", "homophilous", "survey"])
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", default=["data/train_coev3.npz"])
    t.add_argument("--out", default="data/npe_coev3")
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--max-epochs", type=int, default=300)
    t.add_argument("--stop-after", type=int, default=20)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--transforms", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sbc")
    s.add_argument("--posterior", default="data/npe_coev3.pt")
    s.add_argument("--waves", type=int, default=3)
    s.add_argument("--N", type=int, default=4000)
    s.add_argument("--chunk", type=int, default=100)
    s.add_argument("--posterior-samples", type=int, default=1000)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--n-min", type=int, default=20)
    s.add_argument("--n-max", type=int, default=80)
    s.add_argument("--rate-net-max", type=float, default=12.0)
    s.add_argument("--start", default="m2", choices=["m2", "sparse", "homophilous", "survey"])
    r = sub.add_parser("s50")
    r.add_argument("--posterior", default="data/npe_coev3.pt")
    r.add_argument("--real", default="s50", choices=list(REAL))
    a = ap.parse_args()
    {"generate": cmd_generate, "train": cmd_train, "sbc": cmd_sbc, "s50": cmd_s50}[a.cmd](a)


if __name__ == "__main__":
    main()
