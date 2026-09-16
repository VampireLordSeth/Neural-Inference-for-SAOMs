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
import time

import numpy as np
import torch

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


def _sample_in_box(posterior, x, num_samples, low, high, rounds=(2, 4, 16)):
    """``num_samples`` posterior draws per observation, rejecting draws outside the prior box.

    Each round oversamples by the given factor for the observations still short
    of ``num_samples`` in-box draws. Whatever is still short after the last
    round is topped up with out-of-box draws clamped to the box face; the
    number of such observations is returned so a run where that matters shows.
    """
    B, D = x.shape[0], low.shape[0]
    out = torch.empty(num_samples, B, D, device=x.device)
    filled = torch.zeros(B, dtype=torch.long, device=x.device)
    active = torch.arange(B, device=x.device)
    for f in rounds:
        with torch.no_grad():
            s = posterior.sample_batched(
                (f * num_samples,),
                x=x[active],
                reject_outside_prior=False,
                show_progress_bars=False,
            )  # (f*S, A, D)
        inbox = ((s >= low) & (s <= high)).all(-1)  # (f*S, A)
        order = torch.argsort(~inbox, dim=0, stable=True)  # in-box draws first
        s = torch.gather(s, 0, order.unsqueeze(-1).expand(-1, -1, D))
        n_in = inbox.sum(0)
        for j, b in enumerate(active.tolist()):
            k = int(filled[b])
            take = min(int(n_in[j]), num_samples - k)
            out[k : k + take, b] = s[:take, j]
            filled[b] = k + take
        active = active[filled[active] < num_samples]
        if len(active) == 0:
            return out, 0
    # top up the stubborn ones with clamped out-of-box draws
    for b in active.tolist():
        k = int(filled[b])
        with torch.no_grad():
            s = posterior.sample_batched(
                (num_samples - k,),
                x=x[b : b + 1],
                reject_outside_prior=False,
                show_progress_bars=False,
            )[:, 0]
        out[k:, b] = torch.maximum(torch.minimum(s, high), low)
    return out, len(active)


def sbc_ranks_chunked(posterior, thetas, xs, num_samples: int, chunk: int = 100, log=print):
    """SBC ranks and data-averaged-posterior draws, sampling ``chunk`` observations at a time.

    A drop-in for ``sbi.diagnostics.run_sbc`` that bounds memory (sbi's batched
    sampler otherwise draws all N x S samples at once, which took the Spark's
    unified memory down when a training job shared it), bounds the rejection
    sampling (sbi's loops until every observation has its quota, which stalls
    on a posterior that sits almost entirely outside the box) and prints
    progress per chunk. The same posterior is tested as before: draws outside
    the prior box are rejected, as in ``DirectPosterior.sample``.
    """
    N = thetas.shape[0]
    ranks = torch.zeros(N, thetas.shape[1])  # float, as run_sbc returns them
    dap = torch.zeros_like(thetas)
    clamped = 0
    low, high = posterior.prior.base_dist.low, posterior.prior.base_dist.high  # BoxUniform
    t0 = time.perf_counter()
    for i in range(0, N, chunk):
        s, c = _sample_in_box(posterior, xs[i : i + chunk], num_samples, low, high)
        clamped += c
        ranks[i : i + chunk] = (s < thetas[i : i + chunk]).sum(0).cpu()
        dap[i : i + chunk] = s[0]
        if (i // chunk) % 10 == 0 or i + chunk >= N:
            log(f"  sbc {min(i + chunk, N)}/{N} ({time.perf_counter() - t0:.0f}s)")
    log(f"  observations topped up with clamped out-of-box draws: {clamped} of {N}")
    return ranks, dap


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
