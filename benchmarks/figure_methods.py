"""Where the simulation loop sits: RSiena's estimator against an amortized one.

    python benchmarks/figure_methods.py [--out docs/figures]

Both methods fit the same model by simulating from it, and neither can write down a
likelihood. The difference is **where the loop goes**. RSiena puts it around the dataset:
simulate, compare target statistics, update theta, repeat, and every new dataset pays the
whole cost again. An amortized estimator puts the loop in training, over draws from the
prior, and what is left at inference time is one forward pass.

That makes a figure worth drawing, because the obvious reading of it is wrong. Panel B is
the cumulative-compute version of the same story, and the break-even is at **roughly 270
panels** -- more datasets than most studies have. If you train the flow yourself for one
classroom you have spent seventeen hours to save four minutes. The amortized method wins on
compute only because the training cost is paid *once for everyone*: for somebody who
downloads the estimator, the marginal cost is the 0.06 s and the break-even is the first
panel. What it buys the first user is not compute, it is a full posterior instead of an
asymptotic standard error, and a sampler that does not have to converge.

Numbers are the measured ones recorded in ``MODELS.md`` and ``docs/PRIORS_M7.md``:
RSiena 219 s and ours 0.06 s on the three-wave Glasgow panel (n = 129), and 5 h 34 m to
generate plus 10 h 56 m to train the M6 network estimator on 10^7 panels.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Measured, not illustrative. MODELS.md (npe_m4b card) for the two fit times; PRIORS_M7.md
# for the M6 network generation and training times.
RSIENA_SECONDS = 219.0
AMORTIZED_SECONDS = 0.06
TRAIN_HOURS = 5 + 34 / 60 + 10 + 56 / 60  # generate + train, one GPU

C_RSIENA = "#2c6fa8"
C_AMORT = "#1a7f64"
C_CAVEAT = "#c0392b"
C_DATA = "#e8e8e8"
C_TRAIN_BG = "#eef4f1"


def box(ax, x, y, w, h, text, ec, fc="white", fontsize=8.6, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=4, linespacing=1.45)


def down(ax, x, y_from, y_to, color):
    ax.annotate(
        "", xy=(x, y_to), xytext=(x, y_from),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color,
                        shrinkA=1, shrinkB=1, mutation_scale=13),
        zorder=2,
    )


def self_loop(ax, x_right, y, h, color, label):
    """The arrow that makes the point: a loop whose cost is multiplied by its label.

    Drawn as an explicit three-segment return path rather than a curved ``arc3``, whose
    radius has to be large enough to clear the box and then lands in the next column.
    """
    xo = x_right + 0.40
    yb, yt = y - h * 0.34, y + h * 0.34
    ax.plot([x_right, xo, xo], [yb, yb, yt], lw=1.3, color=color, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.annotate(
        "", xy=(x_right, yt), xytext=(xo, yt),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color,
                        shrinkA=0, shrinkB=0, mutation_scale=13),
        zorder=2,
    )
    ax.text(xo + 0.14, y, label, ha="left", va="center", fontsize=8.2,
            color=color, fontweight="bold", linespacing=1.4)


def panel_schematic(ax):
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0, 10)
    ax.axis("off")

    xr, xa = 2.05, 8.15
    wr, wa = 3.1, 3.1

    # ---------------------------------------------------------------- RSiena column
    ax.text(xr, 9.68, "RSiena  (siena07)", ha="center", fontsize=11.5,
            fontweight="bold", color=C_RSIENA)
    ax.text(xr, 9.26, "the loop is around your dataset", ha="center", fontsize=8.8,
            color=C_RSIENA, style="italic")

    box(ax, xr, 8.45, wr, 0.78, "observed panel  $x_0 \\ldots x_W$", C_RSIENA, C_DATA)
    down(ax, xr, 8.06, 7.35, C_RSIENA)
    box(ax, xr, 6.55, wr, 1.6,
        "simulate the SAOM forward\ncompare target statistics\nupdate $\\theta$",
        C_RSIENA)
    self_loop(ax, xr + wr / 2 + 0.1, 6.55, 1.6, C_RSIENA, "thousands of\nsimulations")
    down(ax, xr, 5.75, 5.05, C_RSIENA)
    box(ax, xr, 4.62, wr, 0.86, "$\\hat{\\theta}$ and asymptotic\nstandard errors",
        C_RSIENA, C_DATA)

    ax.text(xr, 3.62, "every dataset pays this again, in full", ha="center",
            fontsize=8.8, color=C_RSIENA, fontweight="bold")
    ax.text(xr, 3.22, "219 s for one three-wave panel, $n = 129$", ha="center",
            fontsize=8.2, color="0.35")

    # ------------------------------------------------------------- amortized column
    ax.text(xa, 9.68, "Amortized neural estimator", ha="center", fontsize=11.5,
            fontweight="bold", color=C_AMORT)
    ax.text(xa, 9.26, "the loop is in training, over the prior", ha="center",
            fontsize=8.8, color=C_AMORT, style="italic")

    tr_l, tr_r = xa - wa / 2 - 0.40, 11.55
    ax.add_patch(
        FancyBboxPatch(
            (tr_l, 5.42), tr_r - tr_l, 3.42,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=C_TRAIN_BG, edgecolor=C_AMORT, linewidth=1.0,
            linestyle=(0, (5, 3)), zorder=1,
        )
    )
    ax.text(tr_l + 0.12, 8.66, "TRAINING  ·  once, for everyone", ha="left",
            fontsize=8.3, color=C_AMORT, fontweight="bold")

    box(ax, xa, 7.72, wa, 1.1,
        "draw $\\theta \\sim$ prior box\nsimulate a panel, reduce it\nto a summary $s$",
        C_AMORT)
    self_loop(ax, xa + wa / 2 + 0.1, 7.72, 1.1, C_AMORT, "$10^7$\ndraws")
    down(ax, xa, 7.14, 6.58, C_AMORT)
    box(ax, xa, 6.12, wa, 0.9, "train conditional flow\n$q(\\theta \\mid s)$", C_AMORT)
    ax.text(xa, 5.62, "$\\approx$ 17 h on one GPU", ha="center", fontsize=8.2,
            color="0.35")

    down(ax, xa, 5.36, 4.98, C_AMORT)
    ax.text(tr_l + 0.12, 4.72, "INFERENCE  ·  per dataset", ha="left",
            fontsize=8.3, color=C_AMORT, fontweight="bold")
    box(ax, xa, 4.1, wa, 0.72, "observed panel $\\rightarrow$ summary $s$",
        C_AMORT, C_DATA)
    down(ax, xa, 3.72, 3.3, C_AMORT)
    box(ax, xa, 2.88, wa, 0.84,
        "one forward pass $\\rightarrow$\nfull posterior sample", C_AMORT, C_DATA)

    ax.text(xa, 2.12, "0.06 s, and nothing is simulated", ha="center", fontsize=8.8,
            color=C_AMORT, fontweight="bold")

    # ------------------------------------------------------------------- the trade
    ax.plot([0.25, 11.5], [1.55, 1.55], lw=0.8, color="0.8")
    ax.text(0.25, 1.18, "What you give up", fontsize=9, fontweight="bold", color="0.25")
    ax.text(
        0.25, 0.74,
        "RSiena:  the specification is chosen when you run it — any effect set, "
        "any $n$, any number of waves.",
        fontsize=8.3, color=C_RSIENA,
    )
    ax.text(
        0.25, 0.30,
        "Amortized:  effect set, $n$ range and priors are fixed at training time. "
        "Outside them the posterior does not get noisy — it gets confident and wrong.",
        fontsize=8.3, color=C_CAVEAT,
    )


def panel_crossover(ax):
    k = np.arange(0, 601)
    rsiena = RSIENA_SECONDS * k / 3600.0
    amort = TRAIN_HOURS + AMORTIZED_SECONDS * k / 3600.0
    cross = TRAIN_HOURS * 3600.0 / (RSIENA_SECONDS - AMORTIZED_SECONDS)

    ax.plot(k, rsiena, color=C_RSIENA, lw=2.1, label="RSiena, one fit per dataset")
    ax.plot(k, amort, color=C_AMORT, lw=2.1,
            label="amortized, trained once then reused")
    ax.axvline(cross, color="0.6", lw=0.9, ls=(0, (4, 3)))
    ax.plot([cross], [TRAIN_HOURS], "o", ms=6, color="0.3", zorder=5)

    ax.text(cross + 12, TRAIN_HOURS - 3.2, f"break-even at\n{cross:.0f} panels",
            fontsize=8.6, color="0.25", ha="left", va="top", linespacing=1.5)
    ax.annotate(
        "train it yourself for one classroom\nand you spend 17 h to save 4 minutes",
        xy=(6, TRAIN_HOURS + 0.25), xytext=(44, 25.4),
        fontsize=8.3, color=C_CAVEAT, linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_CAVEAT,
                        connectionstyle="arc3,rad=0.26", mutation_scale=11),
    )
    ax.annotate(
        "reuse a published estimator and the\n17 h is already paid — this line\nstarts at zero",
        xy=(487, TRAIN_HOURS - 0.25), xytext=(300, 5.2),
        fontsize=8.3, color=C_AMORT, linespacing=1.5,
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_AMORT,
                        connectionstyle="arc3,rad=-0.25", mutation_scale=11),
    )

    # "Panels", not "waves": the x axis counts datasets. A four-wave study is one panel,
    # not four, and reading it the other way makes the break-even look seven times nearer
    # than it is.
    ax.set_xlabel("panels fitted   (one panel = one group, all of its waves)",
                  fontsize=9.5)
    ax.set_ylabel("cumulative compute (hours)", fontsize=9.5)
    ax.set_title("Amortization is a claim about the second fit,\nnot the first",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 37)
    ax.legend(fontsize=8.3, frameon=False, loc="upper left",
              bbox_to_anchor=(0.015, 1.0))
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)

    dest_dir = ROOT / a.out
    dest_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.2, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.19,
                          left=0.012, right=0.985, top=0.935, bottom=0.135)
    panel_schematic(fig.add_subplot(gs[0, 0]))
    panel_crossover(fig.add_subplot(gs[0, 1]))

    fig.text(0.012, 0.978, "Two ways to fit the same model by simulating from it",
             fontsize=13, fontweight="bold", va="top")
    fig.text(
        0.012, 0.050,
        "A $\\bf{wave}$ is one observation of the network; a $\\bf{period}$ is the "
        "interval between two consecutive waves, and a $W$-wave panel has $W-1$ of them; "
        "a $\\bf{panel}$ is one group observed at all of its waves — one dataset.",
        fontsize=7.6, color="0.4", va="bottom",
    )
    fig.text(
        0.012, 0.018,
        "Fit times are the measured ones for the three-wave Glasgow panel ($n = 129$); "
        "the 17 h is 5 h 34 m to generate $10^7$ panels plus 10 h 56 m to train, for the "
        "M6 network estimator. Both are single runs on one machine, not benchmarks.",
        fontsize=7.6, color="0.4", va="bottom",
    )

    for ext in ("png", "pdf"):
        dest = dest_dir / f"methods_comparison.{ext}"
        fig.savefig(dest, dpi=200)
        print(f"wrote {dest.relative_to(ROOT)}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
