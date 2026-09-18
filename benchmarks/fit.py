"""Fit a saved amortized estimator to a new longitudinal network panel.

    python benchmarks/fit.py --waves w1.csv w2.csv w3.csv --v v.csv --g g.csv
    python benchmarks/fit.py --waves w1.csv w2.csv w3.csv --behaviour z.csv
    python benchmarks/fit.py --waves ... --v ... --g ... --rsiena fit.json --out results/mydata

Inputs are plain CSVs without headers: each wave an n x n 0/1 adjacency matrix (rows
send ties), ``--v`` one numeric value per actor (centred here), ``--g`` one category
label per actor (integers or strings), ``--behaviour`` an n x waves matrix of integer
scores on a 1..k scale (k <= 5). Actors must be the same, in the same order, at every
wave; there is no missing-data handling.

Two estimators, chosen by the inputs (`docs/M4_RESULTS.md`):
  network only   three waves, effects density, recip, transTrip, cycle3, altX(v), egoX(v),
                 sameX(g); rates U(1, 20); trained for 20 <= n <= 200
  co-evolution   three waves, the same structural effects plus egoZ, altZ, simZ selection
                 and linear, quad, avAlt behaviour effects; behaviour on 3-5 categories

Before sampling, the dataset's 42 (or 62) summaries are placed in the training
population: a summary below the 2nd or above the 98th percentile of same-size training
panels is flagged. A flagged dataset still gets a posterior, but one that shrinks toward
the population rather than toward the data (the step-1 Glasgow rates in
docs/M4_RESULTS.md are what that looks like), so read it with the flags in mind.
Posterior mass piling up against a prior face is flagged the same way.

Runs on CPU in under a second per dataset; needs the torch environment (.venv-torch).
The posteriors (data/npe_*.pt, ~4 MB each) are trained on the Spark and copied into data/;
the screening references (models/screen_*.npz, built by benchmarks/screen_reference.py)
are committed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {
    "network": {
        "posterior": ["data/npe_m4b.pt"],
        "screen": ["models/screen_m4b.npz"],
        "doc": "docs/M4_RESULTS.md (step 2)",
    },
    "coev": {
        "posterior": ["data/npe_coev_m4b.pt", "data/npe_coev_m4.pt"],
        "screen": ["models/screen_coev_m4b.npz", "models/screen_coev_m4.npz"],
        "doc": "docs/M4_RESULTS.md (co-evolution)",
    },
}
RS_LABELS = {  # our names -> RSiena effect names as the R scripts here label them
    "rate_1": "{net}:rate_1",
    "rate_2": "{net}:rate_2",
    "rate_net_1": "{net}:rate_1",
    "rate_net_2": "{net}:rate_2",
    "rate_beh_1": "{beh}:rate_1",
    "rate_beh_2": "{beh}:rate_2",
    "density": "{net}:density",
    "recip": "{net}:recip",
    "transTrip": "{net}:transTrip",
    "cycle3": "{net}:cycle3",
    "altX(v)": "{net}:altX",
    "egoX(v)": "{net}:egoX",
    "sameX(g)": "{net}:sameX",
    "egoZ": "{net}:egoX",
    "altZ": "{net}:altX",
    "simZ": "{net}:simX",
    "linear": "{beh}:linear",
    "quad": "{beh}:quad",
    "avAlt": "{beh}:avAlt",
}


def first_existing(paths):
    for p in paths:
        if (ROOT / p).exists():
            return ROOT / p
    raise SystemExit(f"none of {paths} found under {ROOT}; pull the posterior from the Spark")


def load_matrix(path, dtype=float):
    a = np.loadtxt(path, delimiter=",", dtype=dtype, ndmin=2)
    return a


def load_waves(paths):
    Xs = [load_matrix(p, np.int8) for p in paths]
    n = Xs[0].shape[0]
    for p, X in zip(paths, Xs, strict=True):
        if X.shape != (n, n):
            raise SystemExit(f"{p}: expected an {n} x {n} adjacency matrix, got {X.shape}")
        if not set(np.unique(X)).issubset({0, 1}):
            raise SystemExit(f"{p}: entries must be 0/1")
        np.fill_diagonal(X, 0)
    return Xs


def screen(S, names, n, ref_path):
    """Percentile of each summary among same-size training panels; flags outside 2-98."""
    ref = np.load(ref_path, allow_pickle=False)
    rnames = list(ref["names"])
    assert rnames == list(names), "screen reference does not match this summary set"
    lo, w = ref["band_lo"], int(ref["band_width"])
    b = int(np.clip((n - lo[0]) // w, 0, len(lo) - 1))
    grid = ref["quantiles"][b]  # (99, summaries)
    out = []
    for k, nm in enumerate(names):
        col = grid[:, k]
        if np.isnan(col).all():
            out.append((nm, S[k], np.nan))
            continue
        # percentile among the 99 stored quantiles, ties counted half (so a discrete
        # summary sitting on its modal value reads as central, not extreme)
        pct = ((col < S[k]).sum() + 0.5 * np.isclose(col, S[k]).sum() + 0.5) / (len(col) + 1)
        out.append((nm, S[k], pct))
    return out, (int(lo[b]), int(lo[b]) + w - 1, int(ref["counts"][b]), str(ref["source"]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--waves", nargs="+", required=True, help="adjacency CSVs in time order")
    ap.add_argument("--v", help="numeric actor covariate CSV (network-only model)")
    ap.add_argument("--g", help="categorical actor covariate CSV (network-only model)")
    ap.add_argument("--behaviour", help="n x waves integer behaviour CSV (co-evolution model)")
    ap.add_argument("--posterior", help="override the saved posterior (.pt)")
    ap.add_argument("--screen", help="override the screening reference (.npz)")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument(
        "--rsiena", help="RSiena fit JSON (as written by the R scripts here) to compare"
    )
    ap.add_argument(
        "--rsiena-labels", default="net,alc", help="network,behaviour names in that JSON"
    )
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--out", help="prefix for <out>_posterior.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    torch.manual_seed(a.seed)
    Xs = load_waves(a.waves)
    n, waves = Xs[0].shape[0], len(Xs)
    if waves != 3:
        raise SystemExit(
            "the saved M4 estimators are three-wave models; pass --posterior for others"
        )
    if a.behaviour:
        kind = "coev"
        from saomsim.population_coev import (
            coev_summary_names,
            net_model,
            real_coev_summary,
            transform_coev,
        )

        Z = load_matrix(a.behaviour)
        if Z.shape != (n, waves):
            raise SystemExit(f"--behaviour: expected {n} x {waves}, got {Z.shape}")
        z_min, z_max = int(Z.min()), int(Z.max())
        if not (1 <= z_min and z_max <= 5 and z_max - z_min >= 2):
            raise SystemExit("behaviour must be integers on a 1..k scale with 3 <= k <= 5")
        S, spec = real_coev_summary(Xs, [Z[:, w] for w in range(waves)], 1, z_max)
        S = np.asarray(S).reshape(-1)
        names = coev_summary_names(waves)
        x = transform_coev(S[None], names, net_model())
        consts = (
            f"zbar {float(spec.zbar[0]):.3f}, simMean {float(spec.sim_mean[0]):.3f}, "
            f"scale 1..{z_max}"
        )
    else:
        kind = "network"
        if not (a.v and a.g):
            raise SystemExit("network-only model needs --v and --g (or pass --behaviour)")
        from saomsim.population import m2_summary_names, real_data_summary, transform_m2

        v = load_matrix(a.v).reshape(-1)
        graw = np.loadtxt(a.g, delimiter=",", dtype=str, ndmin=1).reshape(-1)
        if len(v) != n or len(graw) != n:
            raise SystemExit("--v and --g need one value per actor")
        v = v - v.mean()
        _, g = np.unique(graw, return_inverse=True)
        S, model = real_data_summary(Xs[0], Xs[1:], v, g)
        S = np.asarray(S).reshape(-1)
        names = m2_summary_names(model, waves)
        x = transform_m2(S[None], names, model)
        consts = f"v centred (sd {v.std(ddof=1):.3f}), g {len(set(g))} categories"

    post_path = Path(a.posterior) if a.posterior else first_existing(MODELS[kind]["posterior"])
    posterior = torch.load(post_path, weights_only=False, map_location="cpu")
    low, high = (
        posterior.prior.base_dist.low.cpu().numpy(),
        posterior.prior.base_dist.high.cpu().numpy(),
    )
    if kind == "coev":
        from saomsim.population_coev import coev_theta_names

        pnames = coev_theta_names(3)
    else:
        pnames = ["rate_1", "rate_2"] + list(model.labels)
    if len(pnames) != len(low):
        raise SystemExit(
            f"{post_path} has {len(low)} parameters, expected {len(pnames)} for the {kind} model"
        )

    print(f"{kind} model: {post_path.relative_to(ROOT)}  ({MODELS[kind]['doc']})")
    print(f"data: n={n}, {waves} waves, ties per wave {[int(X.sum()) for X in Xs]}; {consts}")

    flags = []
    if not a.no_screen:
        ref = Path(a.screen) if a.screen else first_existing(MODELS[kind]["screen"])
        rows, (blo, bhi, cnt, src) = screen(S, names, n, ref)
        flags = [
            (nm, val, pct)
            for nm, val, pct in rows
            if not np.isnan(pct) and (pct < 0.02 or pct > 0.98)
        ]
        print(
            f"screen: against {cnt:,} training panels with n in [{blo}, {bhi}] ({Path(src).name})"
        )
        if flags:
            print("  outside the 2nd-98th percentile of the population:")
            for nm, val, pct in flags:
                print(f"    {nm:<20} {val:>10.3f}   percentile {pct:.3f}")
            print("  -> the posterior below leans on the population where these summaries differ")
        else:
            print("  all summaries inside the 2nd-98th percentile: in distribution")

    t0 = time.perf_counter()
    smp = posterior.sample(
        (a.samples,), x=torch.as_tensor(x, dtype=torch.float32), show_progress_bars=False
    )
    smp = smp.cpu().numpy()
    dt = time.perf_counter() - t0
    q = np.quantile(smp, [0.05, 0.5, 0.95], axis=0)
    face = [
        (
            nm,
            (
                (smp[:, k] < low[k] + 0.02 * (high[k] - low[k]))
                | (smp[:, k] > high[k] - 0.02 * (high[k] - low[k]))
            ).mean(),
        )
        for k, nm in enumerate(pnames)
    ]
    face = [(nm, f) for nm, f in face if f > 0.05]

    rs = None
    if a.rsiena:
        rs = json.loads(Path(a.rsiena).read_text(encoding="utf-8"))
        net_lab, beh_lab = a.rsiena_labels.split(",")

    print(f"\nposterior from {a.samples:,} samples ({dt:.2f}s):")
    hdr = f"{'parameter':<12}{'mean':>9}{'sd':>8}{'90%':>18}{'prior':>16}"
    if rs:
        hdr += f"{'RSiena':>9}{'se':>7}{'z':>6}{'in 90%':>8}"
    print(hdr)
    for k, nm in enumerate(pnames):
        m, s = smp[:, k].mean(), smp[:, k].std()
        line = (
            f"{nm:<12}{m:>9.3f}{s:>8.3f}{f'[{q[0, k]:.2f}, {q[2, k]:.2f}]':>18}"
            f"{f'U({low[k]:g}, {high[k]:g})':>16}"
        )
        if rs:
            lab = RS_LABELS[nm].format(net=net_lab, beh=beh_lab)
            e, se = rs["estimate"].get(lab), rs["se"].get(lab)
            if e is not None:
                inside = "yes" if q[0, k] <= e <= q[2, k] else "no"
                line += f"{e:>9.3f}{se:>7.3f}{(e - m) / s:>6.2f}{inside:>8}"
        print(line)
    if face:
        print(
            "\nposterior mass within 2% of a prior face (>5%): "
            + ", ".join(f"{nm} {f:.0%}" for nm, f in face)
            + "\n  -> the prior box is binding there; the interval is not trustworthy at that edge"
        )
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            f"{out}_posterior.npz",
            samples=smp,
            names=np.array(pnames),
            summaries=S,
            summary_names=np.array(names),
            n=np.array(n),
        )
        print(f"\nwrote {out}_posterior.npz")


if __name__ == "__main__":
    main()
