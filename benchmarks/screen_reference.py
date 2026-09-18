"""Compact in-distribution reference for `fit.py`: per-size-band quantiles of every raw
summary in a training set, so a new dataset can be placed in the population without the
training set (hundreds of MB, on the Spark) being present.

    python benchmarks/screen_reference.py --data data/train_m4b.npz --out models/screen_m4b.npz
    python benchmarks/screen_reference.py --data data/train_coev_m4b.npz --coev \
        --out models/screen_coev_m4b.npz

Bands are 20 wide in n (20-39, 40-59, ...). For each band and summary the 1st..99th
percentiles are stored; `fit.py` reads a dataset's percentile off the band containing
its n by interpolation and flags summaries below the 2nd or above the 98th.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BAND = 20
Q = np.arange(1, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="training set (path or glob)")
    ap.add_argument("--coev", action="store_true", help="co-evolution training set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.coev:
        from benchmarks.npe_coev import load_summaries

        ts = load_summaries([a.data])
    else:
        from saomsim.population import load_m2_summaries

        ts = load_m2_summaries(a.data)
    names = list(ts.summary_names)
    n = ts.n
    lo = np.arange(n.min() // BAND * BAND, n.max() + 1, BAND)
    grids, counts = [], []
    for b in lo:
        m = (n >= b) & (n < b + BAND)
        counts.append(int(m.sum()))
        grids.append(
            np.percentile(ts.summary[m], Q, axis=0)
            if m.any()
            else np.full((len(Q), len(names)), np.nan)
        )
    np.savez_compressed(
        a.out,
        names=np.array(names),
        band_lo=lo,
        band_width=np.array(BAND),
        counts=np.array(counts),
        quantiles=np.array(grids, dtype=np.float32),  # (bands, 99, summaries)
        source=np.array(str(a.data)),
        start_regime=np.array(str(ts.meta.get("start_regime", "m2"))),
    )
    print(
        f"wrote {a.out}: {len(lo)} bands x {len(Q)} quantiles x {len(names)} summaries "
        f"({Path(a.out).stat().st_size / 1e3:.0f} kB); panels per band {counts}"
    )


if __name__ == "__main__":
    main()
