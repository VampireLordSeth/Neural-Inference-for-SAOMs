"""Torch backend: the same kernels as ``backend.py`` on torch tensors.

Mapping (see GETTING_STARTED §6):

    X[rows, actor, :]                    -> X[torch.arange(B), actor, :]
    (out_row[:, None, :] @ X)[:, 0, :]   -> torch.bmm(out_row.unsqueeze(1), X).squeeze(1)
    X @ stack(cols, -1)                  -> torch.bmm(X, torch.stack(cols, -1))
    softmax (hand-rolled)                -> torch.softmax
    rng.integers / rng.random            -> torch.randint / torch.rand with a torch.Generator

The Poisson ministep count is *not* drawn here: ``simulate_period`` draws it
with the numpy Generator for every backend, so a seed fixes the ministep
schedule regardless of backend. The torch Generator for the in-loop draws is
seeded from that same numpy Generator.

Default dtype is float64 so results are comparable to the numpy backend to
rounding; pass ``dtype=torch.float32`` for throughput once parity is shown.
"""

from __future__ import annotations

import numpy as np

from .backend import ALL_PRODUCTS, OUT_DTYPE, RowProducts

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("the torch backend needs torch: pip install -e '.[torch]'") from exc


class TorchBackend:
    name = "torch"

    def __init__(self, device: str | None = None, dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype is None else dtype

    # -- conversion
    # np.array (not asarray) so broadcast views become writable copies first
    def network(self, X0):
        return torch.as_tensor(np.array(X0), device=self.device).to(self.dtype)

    def array(self, a):
        return torch.as_tensor(np.array(a, dtype=np.float64), device=self.device).to(self.dtype)

    def int_array(self, a):
        return torch.as_tensor(np.array(a, dtype=np.int64), device=self.device)

    def covariate(self, v):
        return self.array(v)

    def to_numpy(self, a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def finalize(self, X):
        return X.to(torch.int8).cpu().numpy().astype(OUT_DTYPE)

    # -- construction
    def empty(self, shape):
        return torch.empty(shape, dtype=self.dtype, device=self.device)

    def arange(self, n: int):
        return torch.arange(n, device=self.device)

    def broadcast_to(self, a, shape):
        return a.expand(*shape)

    def as_float(self, a):
        return a.to(self.dtype)

    # -- randomness
    def rng_state(self, rng: np.random.Generator):
        g = torch.Generator(device=self.device)
        g.manual_seed(int(rng.integers(0, 2**63 - 1)))
        return g

    def integers(self, n: int, B: int, state):
        return torch.randint(0, n, (B,), generator=state, device=self.device)

    def random(self, B: int, state):
        return torch.rand(B, generator=state, device=self.device, dtype=self.dtype)

    # -- kernels
    def row_products(self, X, actor, needs=ALL_PRODUCTS) -> RowProducts:
        B = X.shape[0]
        rows = torch.arange(B, device=X.device)
        out_row = X[rows, actor, :]
        in_col = X[rows, :, actor]
        two_path = shared_out = back_path = None
        if "two_path" in needs:
            two_path = torch.bmm(out_row.unsqueeze(1), X).squeeze(1)
        right = []
        if "shared_out" in needs:
            right.append(out_row)
        if "back_path" in needs:
            right.append(in_col)
        if right:
            cols = torch.bmm(X, torch.stack(right, dim=-1))
            c = 0
            if "shared_out" in needs:
                shared_out = cols[:, :, c]
                c += 1
            if "back_path" in needs:
                back_path = cols[:, :, c]
        return RowProducts(out_row, in_col, two_path, shared_out, back_path)

    def softmax(self, f):
        return torch.softmax(f, dim=-1)

    def categorical_sample(self, p, u):
        cdf = torch.cumsum(p, dim=-1)
        idx = (u[:, None] > cdf).sum(dim=-1)
        return torch.clamp(idx, max=p.shape[-1] - 1)

    def toggle(self, X, actor, target, active) -> None:
        do = active & (target != actor)
        rows = torch.arange(X.shape[0], device=X.device)[do]
        a, t = actor[do], target[do]
        X[rows, a, t] = 1 - X[rows, a, t]

    def hamming(self, X, Y):
        return (X != Y).sum(dim=(1, 2))

    def __repr__(self) -> str:
        return f"TorchBackend(device={str(self.device)!r}, dtype={self.dtype})"
