"""Simulator throughput at large n on an idle GPU (one period, rate 3, four structural effects).

    python benchmarks/throughput_large_n.py

Batches are sized to keep the working set near 1-2 GB so the GPU is saturated;
best of two runs after a warm-up. Cost per panel scales as n^3 (ministeps ~ n,
each an n^2 batched matvec).
"""

import time

import numpy as np
import torch

from saomsim import Model, random_network, simulate_period
from saomsim.backend_torch import TorchBackend


def main():
    bk = TorchBackend(dtype=torch.float32)
    m = Model(["density", "recip", "transTrip", "cycle3"])
    th = [-3.5, 1.5, 0.3, -0.2]
    print(
        f"{'n':>6} {'B':>6} {'panels/s':>10} {'s/panel':>9} {'10^5 panels':>12} {'10^6 panels':>12}"
    )
    for n, B in [
        (50, 4096),
        (100, 2048),
        (200, 1024),
        (400, 256),
        (800, 64),
        (1600, 16),
        (3200, 4),
    ]:
        rng = np.random.default_rng(0)
        X0 = random_network(B, n, min(0.05, 4.0 / n), rng)
        simulate_period(X0, th, 3.0, m, rng, backend=bk)
        best = float("inf")
        for _ in range(2):
            t = time.perf_counter()
            simulate_period(X0, th, 3.0, m, rng, backend=bk)
            best = min(best, time.perf_counter() - t)
        rate = B / best
        h5, h6 = 1e5 / rate / 3600, 1e6 / rate / 3600
        print(
            f"{n:>6} {B:>6} {rate:>10.1f} {1 / rate:>9.4f} {h5:>10.2f} h {h6:>10.1f} h", flush=True
        )


if __name__ == "__main__":
    main()
