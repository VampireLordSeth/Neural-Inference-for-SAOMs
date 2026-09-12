"""Empirical coverage of central credible intervals from saved SBC ranks.

    python benchmarks/coverage.py data/npe_s50_sbc.npz [more.npz ...]

With L posterior samples per held-out draw, the true parameter lies in the
central (1 - a) interval iff its rank r satisfies a/2 <= r/L <= 1 - a/2. The
table reports the fraction of held-out draws for which that holds at each
nominal level, with a binomial standard error. Well-calibrated: empirical ~
nominal. Below nominal: over-confident (intervals too narrow). Above:
under-confident.
"""

import sys

import numpy as np

LEVELS = (0.50, 0.80, 0.90, 0.95)


def coverage_table(ranks: np.ndarray, num_samples: int, names) -> str:
    u = ranks / num_samples
    N = ranks.shape[0]
    se = np.sqrt(np.array(LEVELS) * (1 - np.array(LEVELS)) / N)
    w = max(len(str(n)) for n in names)
    lines = [f"{'parameter':<{w}}  " + "  ".join(f"{int(a * 100):>5d}%" for a in LEVELS)]
    for k, name in enumerate(names):
        cov = [np.mean((u[:, k] >= (1 - a) / 2) & (u[:, k] <= 1 - (1 - a) / 2)) for a in LEVELS]
        lines.append(f"{str(name):<{w}}  " + "  ".join(f"{c:>6.3f}" for c in cov))
    lines.append(f"{'binomial se':<{w}}  " + "  ".join(f"{s:>6.3f}" for s in se))
    lines.append(f"({N} held-out draws x {num_samples} posterior samples)")
    return "\n".join(lines)


def main():
    for path in sys.argv[1:]:
        z = np.load(path, allow_pickle=False)
        ranks = z["ranks"].astype(float)
        names = [str(n) for n in z["names"]]
        L = int(z["num_samples"]) if "num_samples" in z else int(ranks.max() + 1)
        print(f"\n{path}")
        print(coverage_table(ranks, L, names))


if __name__ == "__main__":
    main()
