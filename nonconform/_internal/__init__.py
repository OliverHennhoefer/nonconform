"""Internal utilities for nonconform.

This package contains internal implementation details and should not be
imported directly by users. The public API is exposed through the main
nonconform package.
"""

from .config import set_params
from .constants import (
    ConformalMode,
    Distribution,
    Kernel,
    Pruning,
    ScorePolarity,
    TieBreakMode,
)
from .log_utils import ensure_numpy_array, get_logger
from .math_utils import (
    AggregationMethod,
    BootstrapAggregationMethod,
    aggregate,
    false_discovery_rate,
    normalize_aggregation_method,
    normalize_bootstrap_aggregation_method,
    statistical_power,
)
from .tuning import tune_kde_hyperparameters

__all__ = [
    "AggregationMethod",
    "BootstrapAggregationMethod",
    "ConformalMode",
    "Distribution",
    "Kernel",
    "Pruning",
    "ScorePolarity",
    "TieBreakMode",
    "aggregate",
    "ensure_numpy_array",
    "false_discovery_rate",
    "get_logger",
    "normalize_aggregation_method",
    "normalize_bootstrap_aggregation_method",
    "set_params",
    "statistical_power",
    "tune_kde_hyperparameters",
]
