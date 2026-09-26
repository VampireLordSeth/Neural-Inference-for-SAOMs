"""Generate the M2 population training set (docs/PRIORS_M2.md).

python benchmarks/generate_m2.py --N 1000000 --out data/train_m2.npz [--seed 2] [--no-networks]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.m2 import m2_prior  # noqa: E402
from saomsim.population import generate_m2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--out", default="data/train_m2.npz")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no-networks", action="store_true")
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--n-min", type=int, default=20)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--rate-max", type=float, default=12.0)
    ap.add_argument(
        "--start", default="m2", choices=["m2", "sparse", "survey"],
        help="start-network regime"
    )
    ap.add_argument(
        "--box", default="default", choices=["default", "sparse"], help="effect prior box"
    )
    args = ap.parse_args()

    backend = args.backend
    if backend.startswith("torch"):
        import torch

        from saomsim.backend_torch import TorchBackend

        backend = TorchBackend(
            device=backend.partition(":")[2] or None, dtype=getattr(torch, args.dtype)
        )

    prior = m2_prior(args.waves, rate=(1.0, args.rate_max), box=args.box)
    print(f"N={args.N} backend={backend} chunk={args.chunk} seed={args.seed}")
    print(f"prior:\n{prior.table()}", flush=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    ts = generate_m2(
        prior,
        args.N,
        rng,
        chunk=args.chunk,
        backend=backend,
        keep_networks=not args.no_networks,
        progress=True,
        waves=args.waves,
        n_range=(args.n_min, args.n_max),
        start=args.start,
    )
    dt = time.perf_counter() - t0
    print(f"simulated {args.N} panels in {dt:.1f}s ({args.N / dt:,.0f} panels/s)")
    ts.meta.update({"seed": args.seed, "dtype": args.dtype, "backend": str(backend)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ts.save(out)
    print(f"saved {out} ({out.stat().st_size / 1e6:,.0f} MB) in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
