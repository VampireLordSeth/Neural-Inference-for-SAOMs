"""saomsim: batched simulator for stochastic actor-oriented models (NumPy)."""

from .effects import COVARIATE, KINDS, STRUCTURAL, Effect, Model
from .estimate import EstimateResult, estimate, moments
from .simulate import random_network, rate_statistic, simulate_panel, simulate_period, statistics

__version__ = "0.1.0"

__all__ = [
    "COVARIATE",
    "KINDS",
    "STRUCTURAL",
    "Effect",
    "EstimateResult",
    "Model",
    "estimate",
    "moments",
    "random_network",
    "rate_statistic",
    "simulate_panel",
    "simulate_period",
    "statistics",
]
