import numpy as np


def calculate_p_val(scores: np.ndarray, calibration_set: np.ndarray) -> np.ndarray:
    """Calculate p-values for scores based on a calibration set.

    This function computes a p-value for each score in the `scores` array by
    comparing it against the distribution of scores in the `calibration_set`.
    The p-value represents the proportion of calibration scores that are
    greater than or equal to the given score, with a small adjustment.

    Args:
        scores (numpy.ndarray): A 1D array of test scores for which p-values
            are to be calculated.
        calibration_set (numpy.ndarray): A 1D array of calibration scores
            used as the reference distribution.

    Returns:
        numpy.ndarray: An array of p-values, each corresponding to an input score
            from `scores`.

    Notes:
        The p-value for each score is computed using the formula:
        p_value = (1 + count(calibration_score >= score)) / (1 + N_calibration)
        where N_calibration is the total number of scores in `calibration_set`.
    """
    # sum_smaller counts how many calibration_set values are >= each score
    sum_smaller = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
    return (1.0 + sum_smaller) / (1.0 + len(calibration_set))


def calculate_weighted_p_val(
    scores: np.ndarray,
    calibration_set: np.ndarray,
    w_scores: np.ndarray,
    w_calib: np.ndarray,
) -> np.ndarray:
    """Calculate weighted p-values for scores using a weighted calibration set.

    This function computes p-values by comparing input `scores` (with
    corresponding `w_scores` weights) against a `calibration_set` (with
    `w_calib` weights). The calculation involves a weighted count of
    calibration scores exceeding each test score, incorporating the weights
    of both the test scores and calibration scores.

    Args:
        scores (numpy.ndarray): A 1D array of test scores.
        calibration_set (numpy.ndarray): A 1D array of calibration scores.
        w_scores (numpy.ndarray): A 1D array of weights corresponding to each
            score in `scores`.
        w_calib (numpy.ndarray): A 1D array of weights corresponding to each
            score in `calibration_set`.

    Returns:
        numpy.ndarray: An array of weighted p-values corresponding to the input
            `scores`.
    """
    # Create comparison matrix: True where calibration_set[j] >= scores[i]
    comparison_matrix = calibration_set >= scores[:, np.newaxis]

    # Weighted sum of calibration scores >= test score
    weighted_sum_calib_ge_score = np.sum(comparison_matrix * w_calib, axis=1)

    # Sum of weights of higher-scoring calibration items + self weight
    numerator = weighted_sum_calib_ge_score + w_scores

    # Total calibration weight + test instance weight
    denominator = np.sum(w_calib) + w_scores

    # Handle division by zero
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
