import numpy as np


def false_discovery_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate the False Discovery Rate (FDR) for binary classification.

    The False Discovery Rate is the proportion of false positives among all
    instances predicted as positive. It is calculated as:
    FDR = FP / (FP + TP), where FP is false positives and TP is true positives.
    If the total number of predicted positives (FP + TP) is zero, FDR is
    defined as 0.0.

    Args:
        y (numpy.ndarray): True binary labels, where 1 indicates an actual
            positive (e.g., anomaly) and 0 indicates an actual negative
            (e.g., normal).
        y_hat (numpy.ndarray): Predicted binary labels, where 1 indicates a
            predicted positive and 0 indicates a predicted negative.

    Returns:
        float: The calculated False Discovery Rate.
    """
    y_true = y.astype(bool)
    y_pred = y_hat.astype(bool)

    true_positives = np.sum(y_pred & y_true)
    false_positives = np.sum(y_pred & ~y_true)

    total_predicted_positives = true_positives + false_positives

    if total_predicted_positives == 0:
        fdr = 0.0
    else:
        fdr = false_positives / total_predicted_positives

    return fdr


def statistical_power(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate statistical power (recall or true positive rate).

    Statistical power, also known as recall or true positive rate (TPR),
    measures the proportion of actual positives that are correctly identified
    by the classifier. It is calculated as:
    Power (TPR) = TP / (TP + FN), where TP is true positives and FN is
    false negatives.
    If the total number of actual positives (TP + FN) is zero, power is
    defined as 0.0.

    Args:
        y (numpy.ndarray): True binary labels, where 1 indicates an actual
            positive (e.g., anomaly) and 0 indicates an actual negative
            (e.g., normal).
        y_hat (numpy.ndarray): Predicted binary labels, where 1 indicates a
            predicted positive and 0 indicates a predicted negative.

    Returns:
        float: The calculated statistical power.
    """
    y_bool = y.astype(bool)  # Or y == 1
    y_hat_bool = y_hat.astype(bool)  # Or y_hat == 1

    true_positives = np.sum(y_bool & y_hat_bool)
    false_negatives = np.sum(y_bool & ~y_hat_bool)
    total_actual_positives = true_positives + false_negatives

    if total_actual_positives == 0:
        power = 0.0
    else:
        power = true_positives / total_actual_positives

    return power
