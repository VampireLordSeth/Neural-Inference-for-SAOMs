"""Where a training step's time goes: fetching the batch, or the flow.

    python benchmarks/profile_train.py --data data/m6_shards/train_m6_s30.npz

The trainers hand sbi a CPU ``TensorDataset`` and sbi iterates it with a ``DataLoader``
over a ``SubsetRandomSampler``, which indexes the dataset one row at a time and collates
1,024 rows into each batch in Python. This script times that loader against the same
optimiser step fed from tensors that already live on the GPU, sliced by a permutation, and
splits each into time spent getting the batch and time spent on the step.

It measures throughput only. It says nothing about whether a larger batch reaches the same
validation loss or the same calibration; that needs a training run and an SBC.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.m2 import s50_as_m2  # noqa: E402
from saomsim.population import load_m2_summaries, transform_m2  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def step(net, opt, theta, x):
    opt.zero_grad()
    loss = net.loss(theta, condition=x).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    opt.step()


def timed(batches, net, opt, device, steps, warmup=20):
    """(seconds fetching, seconds stepping) per step, over ``steps`` steps after warm-up."""
    fetch = work = 0.0
    it = iter(batches)
    for k in range(warmup + steps):
        t0 = time.perf_counter()
        theta, x = next(it)
        theta, x = theta.to(device), x.to(device)
        sync(device)
        t1 = time.perf_counter()
        step(net, opt, theta, x)
        sync(device)
        t2 = time.perf_counter()
        if k >= warmup:
            fetch += t1 - t0
            work += t2 - t1
    return fetch / steps, work / steps


def cycle(loader):
    """The loader, epoch after epoch: a shard has fewer large batches than we time."""
    while True:
        yield from loader


def on_device(theta, x, batch):
    while True:
        perm = torch.randperm(theta.shape[0], device=theta.device)
        for i in range(0, perm.numel() - batch + 1, batch):
            idx = perm[i : i + batch]
            yield theta[idx], x[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--transforms", type=int, default=8)
    ap.add_argument("--batches", type=int, nargs="+", default=[1024, 4096, 16384])
    a = ap.parse_args()

    from sbi.neural_nets import posterior_nn
    from torch.utils import data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts = load_m2_summaries(a.data)
    model = s50_as_m2(int(ts.meta.get("waves", 2)))[-1]
    theta = torch.as_tensor(ts.theta, dtype=torch.float32)
    x = torch.as_tensor(transform_m2(ts.summary, ts.summary_names, model), dtype=torch.float32)
    N = theta.shape[0]
    print(f"{a.data}: N={N:,}, theta {theta.shape[1]}, x {x.shape[1]}, device={device}")

    def fresh():
        torch.manual_seed(0)
        build = posterior_nn(model="nsf", hidden_features=a.hidden, num_transforms=a.transforms)
        net = build(theta[:10_000], x[:10_000]).to(device)
        return net, torch.optim.Adam(net.parameters(), lr=5e-4)

    print(f"\n{'feed':<22}{'batch':>7}{'fetch ms':>10}{'step ms':>9}{'steps/s':>9}{'rows/s':>11}")

    def report(name, batch, fetch, work):
        per = fetch + work
        print(
            f"{name:<22}{batch:>7}{1e3 * fetch:>10.2f}{1e3 * work:>9.2f}"
            f"{1 / per:>9.1f}{batch / per:>11,.0f}",
            flush=True,
        )
        return batch / per

    rows = {}
    for batch in a.batches:
        net, opt = fresh()
        loader = data.DataLoader(
            data.TensorDataset(theta, x),
            batch_size=batch,
            drop_last=True,
            sampler=data.SubsetRandomSampler(np.random.default_rng(0).permutation(N).tolist()),
        )
        rows["sbi", batch] = report("sbi DataLoader (CPU)", batch, *timed(
            cycle(loader), net, opt, device, a.steps))
        net, opt = fresh()
        tg, xg = theta.to(device), x.to(device)
        rows["gpu", batch] = report("tensors on device", batch, *timed(
            on_device(tg, xg, batch), net, opt, device, a.steps))

    b0 = a.batches[0]
    print(
        f"\nat batch {b0}: {rows['gpu', b0] / rows['sbi', b0]:.1f}x the rows/s with the data "
        f"on the device.\nOne epoch of 10^7 rows: {1e7 / rows['sbi', b0] / 60:.1f} min as "
        f"trained now, {1e7 / rows['gpu', b0] / 60:.1f} min fed from the device."
    )


if __name__ == "__main__":
    main()
