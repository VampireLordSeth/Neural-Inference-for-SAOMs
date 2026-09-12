"""saomsim: batched simulator for stochastic actor-oriented models (NumPy)."""

from .backend import get_backend
from .effects import COVARIATE, KINDS, STRUCTURAL, Effect, Model
from .estimate import EstimateResult, estimate, moments
from .simulate import (
    SimulationInfo,
    random_network,
    rate_statistic,
    simulate_panel,
    simulate_period,
    statistics,
)

__version__ = "0.1.0"

__all__ = [
    "COVARIATE",
    "KINDS",
    "STRUCTURAL",
    "Effect",
    "EstimateResult",
    "Model",
    "SimulationInfo",
    "estimate",
    "get_backend",
    "moments",
    "random_network",
    "rate_statistic",
    "simulate_panel",
    "simulate_period",
    "statistics",
]
