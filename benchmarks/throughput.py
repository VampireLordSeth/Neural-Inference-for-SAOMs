"""Throughput of ``simulate_period`` across backends, dtypes, n and batch width.

Usage:  python benchmarks/throughput.py [--quick]

Prints panels/s for the model density + recip + transTrip (+ cycle3) at
rate 3 (about 3n ministeps per period). One warm-up run per configuration is
discarded so CUDA kernel launch/compile costs do not count.

Read panels/s as the training-set budget: 10^5-10^6 simulated panels divided
by this number is the wall clock for generating the neural estimator's data.
"""

import argparse
import sys
import time

import numpy as np

from saomsim import Model, get_backend, random_network, simulate_period


def backends():
    out = [("numpy", get_backend("numpy"))]
    try:
        import torch

        from saomsim.backend_torch import TorchBackend

        out.append(("torch cpu f64", TorchBackend("cpu")))
        if torch.cuda.is_available():
            out.append(("torch cuda f64", TorchBackend("cuda")))
            out.append(("torch cuda f32", TorchBackend("cuda", dtype=torch.float32)))
    except ImportError:
        pass
    return out


def bench(bk, n, B, model, theta, reps=2):
    rng = np.random.default_rng(0)
    X0 = random_network(B, n, 0.1, rng)
    simulate_period(X0, theta, 3.0, model, rng, backend=bk)  # warm-up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        simulate_period(X0, theta, 3.0, model, rng, backend=bk)
        best = min(best, time.perf_counter() - t0)
    return B / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    model = Model(["density", "recip", "transTrip"])
    theta = [-2.0, 1.0, 0.2]
    grid = [(30, 1000), (30, 4000), (30, 16000), (100, 400), (100, 2000), (200, 500)]
    if args.quick:
        grid = [(30, 4000), (100, 400)]

    bks = backends()
    names = [name for name, _ in bks]
    print(f"model {model.labels}, rate 3, panels/s (best of 2 after warm-up)\n")
    print(f"{'n':>5} {'B':>7} " + " ".join(f"{nm:>15}" for nm in names))
    for n, B in grid:
        row = []
        for name, bk in bks:
            if name == "numpy" and n * n * B > 2e8:  # ~1.6 GB float64: skip on CPU
                row.append(float("nan"))
                continue
            if name.startswith("torch cpu") and n * n * B > 2e8:
                row.append(float("nan"))
                continue
            try:
                row.append(bench(bk, n, B, model, theta))
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {name} n={n} B={B}: {type(exc).__name__}: {exc}", file=sys.stderr)
                row.append(float("nan"))
        print(
            f"{n:>5} {B:>7} "
            + " ".join(f"{v:>15,.0f}" if np.isfinite(v) else f"{'-':>15}" for v in row)
        )


if __name__ == "__main__":
    main()
