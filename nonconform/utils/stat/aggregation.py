import numpy as np

from nonconform.utils.func.enums import Aggregation


def aggregate(method: Aggregation, scores: np.ndarray) -> np.ndarray:
    """Aggregate anomaly scores using a specified method.

    This function applies a chosen aggregation technique to a 2D array of
    anomaly scores, where each row typically represents scores from a different
    model or source, and each column corresponds to a data sample.

    Args:
        method (Aggregation): The aggregation method to apply. Must be a
            member of the :class:`~nonconform.utils.enums.Aggregation` enum (e.g.,
            ``Aggregation.MEAN``, ``Aggregation.MEDIAN``).
        scores (numpy.ndarray): A 2D NumPy array of anomaly scores.
            It is expected that scores are arranged such that rows correspond
            to different sets of scores (e.g., from different models) and
            columns correspond to individual data points/samples.
            Aggregation is performed along ``axis=0``.

    Returns:
        numpy.ndarray: An array of aggregated anomaly scores. The length of the
            array will correspond to the number of columns in the input `scores` array.

    Raises:
        ValueError: If the `method` is not a supported aggregation type
            defined in the internal mapping.
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
                f"Valid methods are: {valid_methods}. "
                f"Example: aggregate(Aggregation.MEAN, scores)"
            )
