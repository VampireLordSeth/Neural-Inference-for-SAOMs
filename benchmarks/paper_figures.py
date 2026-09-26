"""The three figures that carry the most argument, for the paper and the docs.

    python benchmarks/paper_figures.py [--out docs/figures] [--only ridge]

1. ``per_period_calibration`` -- rank histograms for period 1 against period 2 of a
   three-wave panel read by the product of two-wave posteriors. This is the figure that
   makes the one measured limitation of that construction legible: period 2 starts from an
   *evolved* network, which the training population does not contain, and its rate and
   density ranks tilt. A table of KS p-values says the same thing but does not show that
   the tilt is a smooth slope rather than a spike.

2. ``tau_three_routes`` -- the population mean and the between-school spread per mechanism
   from three unrelated methods. The means agree; the spreads do not, and `sienaBayes`
   returns nearly the same spread for every effect regardless of its scale, which is the
   visual point and is what an inverse-Wishart prior on ten degrees of freedom against
   nineteen groups would produce.

3. ``influence_ridge`` -- the joint posterior of influence against behaviour curvature,
   with the method-of-moments and maximum-likelihood points marked. Three estimators lie
   along one ridge; the disagreement between them is a direction in this plane, not a
   scatter, which is the whole of the argument in the accompanying text.

Figures are written as PNG at 200 dpi. Nothing here reads the network data, only the saved
posteriors and reference fits, so it runs in seconds.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# sienaBayes posterior means of mu and tau, from PartialBayesResult.RData via
# benchmarks/baerveldt/sienabayes_extract.R (400 post-warm-up draws of an intended 500;
# see docs/M5_RESULTS.md for why that caveat matters here specifically). Transcribed
# rather than re-read because the .RData needs R and RSienaTest to open.
SIENABAYES = {
    "rate": (6.08, 2.04),
    "density": (-2.72, 1.02),
    "recip": (2.08, 0.96),
    "transTrip": (0.87, 0.78),
    "cycle3": (-0.59, 0.89),
    "altX(v)": (-0.04, 0.72),
    "egoX(v)": (0.00, 0.72),
    "sameX(g)": (0.50, 0.79),
}
# The co-evolution prior box on the two parameters plotted in the ridge figure
# (docs/PRIORS_M3b.md, PRIORS_M4.md). Both populations used the same range for these.
PRIOR_BOX = {"avAlt": (-1.0, 4.0), "quad": (-1.5, 0.5)}

PRETTY = {
    "rate": "rate",
    "density": "density",
    "recip": "reciprocity",
    "transTrip": "transitive triplets",
    "cycle3": "three-cycles",
    "altX(v)": "alter delinquency",
    "egoX(v)": "ego delinquency",
    "sameX(g)": "same sex",
}


def _siena08(path):
    """parameter -> (mu, sigma) from the siena08 meta-analysis export."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["parameter"].split(":")[-1]
            key = "rate" if key.startswith("rate") else key
            out[key] = (float(row["mu_ml"]), float(row["sigma_ml"]))
    return out


def per_period_calibration(out_dir, src="data/sbc_multiwave_3w_per_period.npz"):
    import matplotlib.pyplot as plt

    p = ROOT / src
    if not p.exists():
        print(f"  skip per-period calibration: {src} not found "
              "(run benchmarks/sbc_multiwave.py --per-period)")
        return None
    z = np.load(p, allow_pickle=False)
    names = [str(s) for s in z["names"]]
    periods = sorted(int(k.split("_")[1]) for k in z if k.startswith("ranks_"))
    from scipy.stats import kstest

    fig, axes = plt.subplots(
        len(periods), len(names), figsize=(2.0 * len(names), 2.4 * len(periods)), sharey="row"
    )
    axes = np.atleast_2d(axes)
    for r, w in enumerate(periods):
        ranks = z[f"ranks_{w}"]
        for c, nm in enumerate(names):
            ax = axes[r, c]
            ks = kstest(ranks[:, c], "uniform").pvalue
            bad = ks < 0.05
            ax.hist(
                ranks[:, c], bins=20, range=(0, 1), density=True,
                color="#c0392b" if bad else "#4a6fa5", alpha=0.85, edgecolor="white", lw=0.4,
            )
            ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(PRETTY.get(nm, nm), fontsize=9)
            if c == 0:
                lab = "population start" if w == 0 else "evolved start"
                ax.set_ylabel(f"period {w + 1}\n({lab})", fontsize=9)
            ax.text(
                0.03, 0.93, f"p {ks:.3f}\nmean {ranks[:, c].mean():.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                color="#c0392b" if bad else "0.25",
            )
    fig.suptitle(
        "Rank calibration by period, three-wave panels read as a product of two-wave "
        "posteriors\nuniform = calibrated; red = rejected at 5 %",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    dest = out_dir / "multiwave_per_period_ranks.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def tau_three_routes(out_dir, src="data/baerveldt_c_net_population.npz"):
    import matplotlib.pyplot as plt

    p = ROOT / src
    if not p.exists():
        print(f"  skip tau figure: {src} not found")
        return None
    z = np.load(p, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    mu, tau = z["mu"], z["tau"]  # (K, draws)
    s08 = _siena08(ROOT / "benchmarks/baerveldt/siena08_network2w.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    y = np.arange(len(names))[::-1]

    # left: population means, three routes
    a1.errorbar(
        mu.mean(1), y,
        xerr=np.abs(np.percentile(mu, [5, 95], axis=1) - mu.mean(1)),
        fmt="o", color="#4a6fa5", ms=5, lw=1.4, capsize=2, label="amortized (90 %)",
    )
    a1.scatter([s08.get(n, (np.nan,))[0] for n in names], y, marker="s", s=34,
               facecolor="none", edgecolor="#2e7d32", lw=1.3, label="siena08")
    a1.scatter([SIENABAYES.get(n, (np.nan,))[0] for n in names], y, marker="^", s=38,
               facecolor="none", edgecolor="#c0392b", lw=1.3, label="sienaBayes")
    a1.axvline(0, color="0.8", lw=0.8)
    a1.set_yticks(y, [PRETTY.get(n, n) for n in names], fontsize=9)
    a1.set_xlabel("population mean $\\mu$")
    a1.set_title("The means agree three ways", fontsize=10)
    a1.legend(fontsize=8, loc="lower right", frameon=False)

    # right: between-school spreads, three routes
    a2.errorbar(
        tau.mean(1), y,
        xerr=np.abs(np.percentile(tau, [5, 95], axis=1) - tau.mean(1)),
        fmt="o", color="#4a6fa5", ms=5, lw=1.4, capsize=2, label="amortized (90 %)",
    )
    a2.scatter([s08.get(n, (np.nan, np.nan))[1] for n in names], y, marker="s", s=34,
               facecolor="none", edgecolor="#2e7d32", lw=1.3, label="siena08")
    a2.scatter([SIENABAYES.get(n, (np.nan, np.nan))[1] for n in names], y, marker="^", s=38,
               facecolor="none", edgecolor="#c0392b", lw=1.3, label="sienaBayes")
    a2.set_yticks(y, [])
    a2.set_xlim(left=0)
    a2.set_xlabel("between-school spread $\\tau$")
    a2.set_title("The spreads do not", fontsize=10)
    a2.legend(fontsize=8, loc="lower right", frameon=False)
    a2.annotate(
        "sienaBayes returns nearly the same $\\tau$\nfor every effect whatever its scale",
        xy=(0.82, 0.55), xycoords="axes fraction", fontsize=8, color="#c0392b", ha="center",
    )

    fig.suptitle(
        "Nineteen school classes: three routes to a population mean and a between-class spread",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    dest = out_dir / "tau_three_routes.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


def influence_ridge(out_dir):
    """Joint posterior of influence against curvature, with the MoM and ML points."""
    import json

    import matplotlib.pyplot as plt

    panels = [
        ("s50, 50 pupils", "data/npe_coev3_10m_c_posterior_s50.npz",
         "benchmarks/rsiena_coevolution_estimate.json",
         "benchmarks/rsiena_coevolution_maxlike.json", "alc"),
        ("Glasgow, 129 pupils", "data/npe_coev_m4_10m_posterior_glasgow.npz",
         "benchmarks/glasgow/rsiena_coevolution.json",
         "benchmarks/glasgow/rsiena_coevolution_maxlike.json", "alcB"),
    ]
    have = [p for p in panels if (ROOT / p[1]).exists()]
    if not have:
        print("  skip ridge figure: no co-evolution posteriors found")
        return None

    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (title, post_path, mom_path, ml_path, beh) in zip(axes, have, strict=True):
        z = np.load(ROOT / post_path, allow_pickle=True)
        names = [str(s) for s in z["names"]]
        S = z["samples"]
        ia, iq = names.index("avAlt"), names.index("quad")
        x, y = S[:, ia], S[:, iq]
        r = np.corrcoef(x, y)[0, 1]
        ax.hexbin(x, y, gridsize=42, cmap="Blues", mincnt=1, linewidths=0)
        for path, marker, colour, lab in (
            (mom_path, "s", "#2e7d32", "RSiena MoM"),
            (ml_path, "D", "#c0392b", "RSiena ML"),
        ):
            f = ROOT / path
            if not f.exists():
                continue
            est = json.loads(f.read_text(encoding="utf-8"))["estimate"]
            key_a, key_q = f"{beh}:avAlt", f"{beh}:quad"
            if key_a not in est:
                continue
            ax.scatter([est[key_a]], [est[key_q]], marker=marker, s=90, zorder=5,
                       facecolor="none", edgecolor=colour, lw=2.0, label=lab)
        ax.scatter([x.mean()], [y.mean()], marker="o", s=70, zorder=5,
                   facecolor="none", edgecolor="#1a3d6d", lw=2.0, label="amortized mean")

        # Draw the prior box. On these panels the posterior piles against two of its
        # faces, so the amortized end of the ridge is where the prior stops it rather than
        # where the data do; hiding that would overstate what the posterior says.
        (alo, ahi), (qlo, qhi) = PRIOR_BOX["avAlt"], PRIOR_BOX["quad"]
        for val, fn in ((ahi, ax.axvline), (qlo, ax.axhline)):
            fn(val, color="0.45", lw=1.1, ls=":", zorder=1)
        fa = (x > ahi - 0.02 * (ahi - alo)).mean()
        fq = (y < qlo + 0.02 * (qhi - qlo)).mean()
        ax.text(
            0.02, 0.02,
            f"prior face (dotted): {100 * fa:.0f} % of mass at the avAlt edge,\n"
            f"{100 * fq:.0f} % at the quad edge",
            transform=ax.transAxes, fontsize=7.5, color="0.35", va="bottom",
        )
        ax.set_xlabel("avAlt  (social influence)")
        ax.set_ylabel("quad  (behaviour curvature)")
        ax.set_title(f"{title}   $r = {r:.2f}$", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(
        "The three estimators disagree along one ridge, not at random",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    dest = out_dir / "influence_curvature_ridge.png"
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    return dest


FIGURES = {
    "per-period": per_period_calibration,
    "tau": tau_three_routes,
    "ridge": influence_ridge,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", choices=list(FIGURES), help="build just one")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {a.only: FIGURES[a.only]} if a.only else FIGURES
    for name, fn in wanted.items():
        print(f"[{name}]")
        dest = fn(out_dir)
        if dest:
            print(f"  wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
