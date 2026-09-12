"""Generate the s50 training set (docs/PRIORS.md).

    python benchmarks/generate_s50.py --N 1000000 --out data/train_s50.npz [--seed 1]

Runs on the torch backend in float32 (validated by the test suite; see README
throughput). Every draw is kept. The seed fully determines the set.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.s50 import load_s50, s50_prior  # noqa: E402
from saomsim.prior import generate_training_set  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/train_s50.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no-networks", action="store_true")
    args = ap.parse_args()

    backend = args.backend
    if backend.startswith("torch"):
        import torch

        from saomsim.backend_torch import TorchBackend

        device = backend.partition(":")[2] or None
        backend = TorchBackend(device=device, dtype=getattr(torch, args.dtype))

    x0, x1, model = load_s50()
    prior = s50_prior(model)
    rng = np.random.default_rng(args.seed)
    print(f"N={args.N}  backend={backend}  chunk={args.chunk}  seed={args.seed}")
    print("prior:\n" + prior.table(), flush=True)

    t0 = time.perf_counter()
    ts = generate_training_set(
        prior,
        x0,
        model,
        args.N,
        rng,
        chunk=args.chunk,
        backend=backend,
        keep_networks=not args.no_networks,
        progress=True,
    )
    dt = time.perf_counter() - t0
    print(f"simulated {args.N} panels in {dt:.1f}s  ({args.N / dt:,.0f} panels/s)")

    ts.meta.update({"seed": args.seed, "dtype": args.dtype, "backend": str(backend)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ts.save(out)
    print(f"saved {out} ({out.stat().st_size / 1e6:,.0f} MB) in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
