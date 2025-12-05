"""Internal utilities for nonconform.

This package contains internal implementation details and should not be
imported directly by users. The public API is exposed through the main
nonconform package.
"""

from .config import set_params
from .constants import Aggregation, Distribution, Kernel, Pruning
from .log_utils import ensure_numpy_array, get_logger
from .math_utils import (
    aggregate,
    false_discovery_rate,
    statistical_power,
)
from .tuning import tune_kde_hyperparameters

__all__ = [
    "Aggregation",
    "Distribution",
    "Kernel",
    "Pruning",
    "aggregate",
    "ensure_numpy_array",
    "false_discovery_rate",
    "get_logger",
    "set_params",
    "statistical_power",
    "tune_kde_hyperparameters",
]
