"""Statistical utilities for conformal anomaly detection.

This module provides statistical functions including aggregation methods,
extreme value theory functions, evaluation metrics, and general statistical
operations used in conformal prediction.
"""

from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power

from .aggregation import aggregate
from .statistical import calculate_p_val
from .weighted_fdr import weighted_false_discovery_control

__all__ = [
    # Aggregation functions
    "aggregate",
    # Statistical functions
    "calculate_p_val",
    # Evaluation metrics
    "false_discovery_rate",
    "statistical_power",
    # FDR control
    "weighted_false_discovery_control",
]
