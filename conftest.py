"""Shared fixtures.

``backend`` parametrises a test over every available backend: numpy always,
torch when importable. Environment knobs for the torch backend:

  SAOMSIM_TORCH_DEVICE   ``cuda`` / ``cpu`` (default: CUDA if available)
  SAOMSIM_TORCH_DTYPE    ``float64`` (default) / ``float32``
"""

import os
from pathlib import Path

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


S50_FILES = ("s501.csv", "s502.csv", "s503.csv", "s50a.csv", "s50_covariates.csv")
NEED_S50 = (
    "the s50 data are not redistributed here; run `python benchmarks/fetch_data.py s50` "
    "(needs R with RSiena installed)"
)


def s50_available() -> bool:
    d = Path(__file__).parent / "benchmarks"
    return all((d / f).exists() for f in S50_FILES)


requires_s50 = pytest.mark.skipif(not s50_available(), reason=NEED_S50)


def to_np(backend, a):
    """Bring a backend array back to numpy (helper importable from tests)."""
    return np.asarray(backend.to_numpy(a))
