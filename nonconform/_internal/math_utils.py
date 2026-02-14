"""Mathematical utilities for nonconform.

This module provides aggregation, metrics, and statistical utilities
used throughout the package.
"""

from typing import Literal, assert_never, cast, get_args

import numpy as np

AggregationMethod = Literal["mean", "median", "minimum", "maximum"]
BootstrapAggregationMethod = Literal["mean", "median"]
_AGGREGATION_METHODS = get_args(AggregationMethod)
_BOOTSTRAP_AGGREGATION_METHODS = get_args(BootstrapAggregationMethod)


def normalize_aggregation_method(method: str) -> AggregationMethod:
    """Normalize and validate a general aggregation method string."""
    if not isinstance(method, str):
        raise TypeError(
            f"aggregation method must be a string, got {type(method).__name__}."
        )
    normalized = method.strip().lower()
    if normalized not in _AGGREGATION_METHODS:
        valid_methods = ", ".join(_AGGREGATION_METHODS)
        raise ValueError(
            f"Unsupported aggregation method: {method!r}. "
            f"Valid methods are: {valid_methods}."
        )
    return cast(AggregationMethod, normalized)


def normalize_bootstrap_aggregation_method(method: str) -> BootstrapAggregationMethod:
    """Normalize and validate a bootstrap aggregation method string."""
    normalized = normalize_aggregation_method(method)
    if normalized not in _BOOTSTRAP_AGGREGATION_METHODS:
        valid_methods = ", ".join(_BOOTSTRAP_AGGREGATION_METHODS)
        raise ValueError(
            f"Unsupported bootstrap aggregation method: {method!r}. "
            f"Valid methods are: {valid_methods}."
        )
    return cast(BootstrapAggregationMethod, normalized)


def aggregate(method: str, scores: np.ndarray) -> np.ndarray:
    """Aggregate anomaly scores using a specified method.

    Applies a chosen aggregation technique to a 2D array of anomaly scores,
    where each row represents scores from a different model and each column
    corresponds to a data sample.

    Args:
        method: The aggregation method to apply.
        scores: A 2D array of anomaly scores. Rows = different models,
            columns = data samples. Aggregation is performed along axis=0.

    Returns:
        Array of aggregated anomaly scores with length equal to number
        of columns in input.

    Raises:
        ValueError: If the method is not a supported aggregation type.

    Examples:
        >>> scores = np.array([[1, 2, 3], [4, 5, 6]])
        >>> aggregate("mean", scores)
        array([2.5, 3.5, 4.5])
    """
    normalized = normalize_aggregation_method(method)
    match normalized:
        case "mean":
            return np.mean(scores, axis=0)
        case "median":
            return np.median(scores, axis=0)
        case "minimum":
            return np.min(scores, axis=0)
        case "maximum":
            return np.max(scores, axis=0)
    assert_never(normalized)


def false_discovery_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the False Discovery Rate (FDR) for binary classification.

    FDR is the proportion of false positives among all predicted positives:
    FDR = FP / (FP + TP)

    If there are no predicted positives, FDR is defined as 0.0.

    Args:
        y: True binary labels (1 = positive/anomaly, 0 = negative/normal).
        y_hat: Predicted binary labels.

    Returns:
        The calculated False Discovery Rate.

    Examples:
        >>> y = np.array([1, 0, 1, 0])
        >>> y_hat = np.array([1, 1, 0, 0])  # 1 TP, 1 FP
        >>> false_discovery_rate(y, y_hat)
        0.5
    """
    y_true = y.astype(bool)
    y_pred = y_hat.astype(bool)

    true_positives = np.sum(y_pred & y_true)
    false_positives = np.sum(y_pred & ~y_true)

    total_predicted_positives = true_positives + false_positives

    if total_predicted_positives == 0:
        return 0.0

    return false_positives / total_predicted_positives


def statistical_power(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate statistical power (recall or true positive rate).

    Power (TPR) is the proportion of actual positives correctly identified:
    Power = TP / (TP + FN)

    If there are no actual positives, power is defined as 0.0.

    Args:
        y: True binary labels (1 = positive/anomaly, 0 = negative/normal).
        y_hat: Predicted binary labels.

    Returns:
        The calculated statistical power.

    Examples:
        >>> y = np.array([1, 0, 1, 0])
        >>> y_hat = np.array([1, 1, 0, 0])  # 1 TP, 1 FN
        >>> statistical_power(y, y_hat)
        0.5
    """
    y_bool = y.astype(bool)
    y_hat_bool = y_hat.astype(bool)

    true_positives = np.sum(y_bool & y_hat_bool)
    false_negatives = np.sum(y_bool & ~y_hat_bool)
    total_actual_positives = true_positives + false_negatives

    if total_actual_positives == 0:
        return 0.0

    return true_positives / total_actual_positives


__all__ = [
    "AggregationMethod",
    "BootstrapAggregationMethod",
    "aggregate",
    "false_discovery_rate",
    "normalize_aggregation_method",
    "normalize_bootstrap_aggregation_method",
    "statistical_power",
]
