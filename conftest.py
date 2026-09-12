"""Shared fixtures.

``backend`` parametrises a test over every available backend: numpy always,
torch when importable. Environment knobs for the torch backend:

  SAOMSIM_TORCH_DEVICE   ``cuda`` / ``cpu`` (default: CUDA if available)
  SAOMSIM_TORCH_DTYPE    ``float64`` (default) / ``float32``
"""

import os

import numpy as np
import pytest

from saomsim.backend import NUMPY


def _backends():
    out = [pytest.param("numpy", id="numpy")]
    try:
        import torch  # noqa: F401

        from saomsim.backend_torch import TorchBackend

        device = os.environ.get("SAOMSIM_TORCH_DEVICE")
        dtype = getattr(torch, os.environ.get("SAOMSIM_TORCH_DTYPE", "float64"))
        out.append(pytest.param(TorchBackend(device=device, dtype=dtype), id="torch"))
    except ImportError:
        out.append(
            pytest.param(None, id="torch", marks=pytest.mark.skip(reason="torch not installed"))
        )
    return out


@pytest.fixture(params=_backends())
def backend(request):
    return NUMPY if request.param == "numpy" else request.param


def to_np(backend, a):
    """Bring a backend array back to numpy (helper importable from tests)."""
    return np.asarray(backend.to_numpy(a))
