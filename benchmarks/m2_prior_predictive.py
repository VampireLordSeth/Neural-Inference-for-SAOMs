"""M2 prior predictive: population summaries, degeneracy by n, and is s50 in-distribution?

python benchmarks/m2_prior_predictive.py --data data/train_m2.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.m2 import s50_as_m2  # noqa: E402
from saomsim.population import M2TrainingSet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_m2.npz")
    args = ap.parse_args()
    ts = M2TrainingSet.load(args.data)
    S, names = ts.summary, ts.summary_names
    x0, x1, v, g, obs, _ = s50_as_m2()
    obs = obs[0]

    print(f"{ts.theta.shape[0]} panels, n in [{ts.n.min()}, {ts.n.max()}]\n")
    q = np.quantile(S, [0.01, 0.5, 0.99], axis=0)
    w = max(len(s) for s in names)
    print(f"{'summary':<{w}} {'q01':>9} {'median':>9} {'q99':>9} {'s50':>9} {'F(s50)':>8}")
    flags = []
    for k, name in enumerate(names):
        F = (S[:, k] < obs[k]).mean() + 0.5 * (S[:, k] == obs[k]).mean()
        print(
            f"{name:<{w}} {q[0, k]:>9.3f} {q[1, k]:>9.3f} {q[2, k]:>9.3f} {obs[k]:>9.3f} {F:>8.3f}"
        )
        if F < 0.01 or F > 0.99:
            flags.append(name)
    print(f"\ns50 outside (0.01, 0.99) on: {flags or 'none'}")

    tf0 = S[:, names.index("x0_tie_fraction")]
    tf1 = S[:, names.index("x1_tie_fraction")]
    print("\ndegeneracy by n (fractions): n_lo..n_hi  X0 empty  X0 >0.5  X1 empty  X1 >0.5")
    for lo in range(ts.n.min(), ts.n.max() + 1, 10):
        m = (ts.n >= lo) & (ts.n < lo + 10)
        if m.any():
            print(
                f"  {lo:>3}..{lo + 9:<3}  {np.mean(tf0[m] == 0):8.4f}  {np.mean(tf0[m] > 0.5):8.4f}"
                f"  {np.mean(tf1[m] == 0):8.4f}  {np.mean(tf1[m] > 0.5):8.4f}   (N={m.sum()})"
            )


if __name__ == "__main__":
    main()
