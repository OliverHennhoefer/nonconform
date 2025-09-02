import numpy as np

from nonconform.utils.func.decorator import _performance_conversion


@_performance_conversion("scores", "calibration_set")
def calculate_p_val(scores: np.ndarray, calibration_set: np.ndarray) -> list[float]:
    """Calculate p-values for scores based on a calibration set.

    This function computes a p-value for each score in the `scores` array by
    comparing it against the distribution of scores in the `calibration_set`.
    The p-value represents the proportion of calibration scores that are
    greater than or equal to the given score, with a small adjustment.

    The `@_performance_conversion` decorator ensures that `scores` and
    `calibration_set` are ``numpy.ndarray`` objects internally and that the
    returned ``numpy.ndarray`` of p-values is converted to a ``list[float]``.

    Args:
        scores (numpy.ndarray): A 1D array of test scores for which p-values
            are to be calculated. Can be passed as a list, which the
            decorator will convert.
        calibration_set (numpy.ndarray): A 1D array of calibration scores
            used as the reference distribution. Can be passed as a list,
            which the decorator will convert.

    Returns
    -------
        list[float]: A list of p-values, each corresponding to an input score
            from `scores`.

    Notes
    -----
        The p-value for each score is computed using the formula:
        p_value = (1 + count(calibration_score >= score)) / (1 + N_calibration)
        where N_calibration is the total number of scores in `calibration_set`.
    """
    # sum_smaller counts how many calibration_set values are >= each score
    sum_smaller = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
    return (1.0 + sum_smaller) / (1.0 + len(calibration_set))


@_performance_conversion("scores", "calibration_set")
def calculate_weighted_p_val(
    scores: np.ndarray,
    calibration_set: np.ndarray,
    w_scores: np.ndarray,
    w_calib: np.ndarray,
) -> list[float]:
    """Calculate weighted p-values for scores using a weighted calibration set.

    This function computes p-values by comparing input `scores` (with
    corresponding `w_scores` weights) against a `calibration_set` (with
    `w_calib` weights). The calculation involves a weighted count of
    calibration scores exceeding each test score, incorporating the weights
    of both the test scores and calibration scores.

    The `@_performance_conversion` decorator handles `scores` and
    `calibration_set`, converting them to ``numpy.ndarray`` if they are lists
    and converting the ``numpy.ndarray`` result to ``list[float]``.
    Note: `w_scores` and `w_calib` are NOT automatically converted by this
    decorator instance and must be provided as ``numpy.ndarray`` objects.

    Args:
        scores (numpy.ndarray): A 1D array of test scores. The decorator
            allows passing a list.
        calibration_set (numpy.ndarray): A 1D array of calibration scores.
            The decorator allows passing a list.
        w_scores (numpy.ndarray): A 1D array of weights corresponding to each
            score in `scores`. Must be a ``numpy.ndarray``.
        w_calib (numpy.ndarray): A 1D array of weights corresponding to each
            score in `calibration_set`. Must be a ``numpy.ndarray``.

    Returns
    -------
        list[float]: A list of weighted p-values corresponding to the input
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
    p_values = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )
    return p_values
