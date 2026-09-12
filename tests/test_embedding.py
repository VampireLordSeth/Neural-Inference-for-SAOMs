"""Tests for the learned panel embedding (torch only)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from saomsim.embedding import PanelEmbedding, _unpack_bits, layout, pack_panel  # noqa: E402
from saomsim.population import (  # noqa: E402
    N_MAX,
    generate_m2,
    m2_model,
    sample_covariates,
    transform_m2,
    unpack,
)
from saomsim.prior import BoxPrior  # noqa: E402


def small_set(N=12, seed=0):
    rng = np.random.default_rng(seed)
    model = m2_model(sample_covariates(1, 20, rng))
    prior = BoxPrior.for_model(
        model,
        rate=(1, 12),
        **{
            "density": (-4, 0),
            "recip": (-1, 4),
            "transTrip": (-0.5, 1.5),
            "cycle3": (-1.5, 0.5),
            "altX(v)": (-1, 1),
            "egoX(v)": (-1, 1),
            "sameX(g)": (-1, 2),
        },
    )
    ts = generate_m2(prior, N, rng, n_range=(20, 30), chunk=4)
    S = transform_m2(ts.summary, ts.summary_names, model)
    x = pack_panel(ts.X0, ts.X1, ts.v, ts.g, ts.n, S)
    return ts, S, x


def test_layout_and_unpack_roundtrip():
    ts, S, x = small_set()
    L = layout(S.shape[1])
    assert x.shape == (12, L["total"])
    xt = torch.as_tensor(x)
    X0 = _unpack_bits(xt[:, L["X0"][0] : L["X0"][1]]).numpy()
    for i in range(12):
        n = int(ts.n[i])
        assert np.array_equal(X0[i, :n, :n], unpack(ts.X0[i : i + 1], n)[0])
        assert X0[i, n:, :].sum() == 0 and X0[i, :, n:].sum() == 0


def test_embedding_shapes_and_finite():
    ts, S, x = small_set()
    emb = PanelEmbedding(S.shape[1], hidden=16, out_dim=8)
    z = emb(torch.as_tensor(x))
    assert z.shape == (12, 8) and torch.all(torch.isfinite(z))


def test_embedding_is_permutation_invariant_and_padding_invariant():
    ts, S, x = small_set()
    emb = PanelEmbedding(S.shape[1], hidden=16, out_dim=8).eval()
    i = 0
    n = int(ts.n[i])
    rng = np.random.default_rng(5)
    perm = rng.permutation(n)
    X0 = unpack(ts.X0[i : i + 1], n)[0]
    X1 = unpack(ts.X1[i : i + 1], n)[0]
    v, g = ts.v[i, :n], ts.g[i, :n]

    def build(X0, X1, v, g):
        P0 = np.zeros((1, N_MAX, N_MAX), dtype=np.uint8)
        P1 = np.zeros_like(P0)
        P0[0, :n, :n], P1[0, :n, :n] = X0, X1
        vv = np.zeros((1, N_MAX), dtype=np.float32)
        gg = np.full((1, N_MAX), -1, dtype=np.int8)
        vv[0, :n], gg[0, :n] = v, g
        return pack_panel(
            np.packbits(P0, axis=-1),
            np.packbits(P1, axis=-1),
            vv,
            gg,
            ts.n[i : i + 1],
            S[i : i + 1],
        )

    with torch.no_grad():
        z_ref = emb(torch.as_tensor(build(X0, X1, v, g)))
        z_perm = emb(torch.as_tensor(build(X0[perm][:, perm], X1[perm][:, perm], v[perm], g[perm])))
        # garbage in the padded region must not matter
        junk_v = np.zeros((1, N_MAX), dtype=np.float32)
        junk_v[0, :n] = v
        junk_v[0, n:] = 99.0
        x_junk = build(X0, X1, v, g)
        L = layout(S.shape[1])
        x_junk[:, L["v"][0] : L["v"][1]] = junk_v
        z_junk = emb(torch.as_tensor(x_junk))
    assert torch.allclose(z_ref, z_perm, atol=1e-5)
    assert torch.allclose(z_ref, z_junk, atol=1e-5)


def test_embedding_trains_one_step():
    ts, S, x = small_set(N=16, seed=1)
    emb = PanelEmbedding(S.shape[1], hidden=16, out_dim=4)
    head = torch.nn.Linear(4, 8)
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=1e-3)
    xt, th = torch.as_tensor(x), torch.as_tensor(ts.theta, dtype=torch.float32)
    loss0 = ((head(emb(xt)) - th) ** 2).mean()
    loss0.backward()
    opt.step()
    opt.zero_grad()
    loss1 = ((head(emb(xt)) - th) ** 2).mean()
    assert torch.isfinite(loss1) and loss1.item() < loss0.item() * 1.5
