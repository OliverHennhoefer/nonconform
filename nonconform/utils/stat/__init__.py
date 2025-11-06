"""Statistical utilities for conformal anomaly detection.

This module provides statistical functions including aggregation methods,
extreme value theory functions, evaluation metrics, and general statistical
operations used in conformal prediction.
"""

from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from nonconform.utils.tune.tuning import tune_kde_hyperparameters

from .aggregation import aggregate
from .results import ConformalResult
from .statistical import calculate_p_val
from .weighted_fdr import weighted_false_discovery_control

__all__ = [
    "ConformalResult",
    "aggregate",
    "calculate_p_val",
    "false_discovery_rate",
    "statistical_power",
    "tune_kde_hyperparameters",
    "weighted_false_discovery_control",
]
