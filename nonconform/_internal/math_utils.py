"""Mathematical utilities for nonconform.

This module provides aggregation, metrics, and statistical utilities
used throughout the package.
"""

import numpy as np

from .constants import Aggregation


def aggregate(method: Aggregation, scores: np.ndarray) -> np.ndarray:
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
        >>> aggregate(Aggregation.MEAN, scores)
        array([2.5, 3.5, 4.5])
    """
    match method:
        case Aggregation.MEAN:
            return np.mean(scores, axis=0)
        case Aggregation.MEDIAN:
            return np.median(scores, axis=0)
        case Aggregation.MINIMUM:
            return np.min(scores, axis=0)
        case Aggregation.MAXIMUM:
            return np.max(scores, axis=0)
        case _:
            valid_methods = ", ".join([f"Aggregation.{a.name}" for a in Aggregation])
            raise ValueError(
                f"Unsupported aggregation method: {method}. "
                f"Valid methods are: {valid_methods}."
            )


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
    "aggregate",
    "false_discovery_rate",
    "statistical_power",
]
