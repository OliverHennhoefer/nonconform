import numpy as np

from nonconform.utils.func.decorator import _performance_conversion
from nonconform.utils.func.enums import Aggregation


@_performance_conversion("scores")
def aggregate(method: Aggregation, scores: np.ndarray) -> list[float]:
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
        list[float]: A list of aggregated anomaly scores. The length of the list
            will correspond to the number of columns in the input `scores` array.

    Raises:
        ValueError: If the `method` is not a supported aggregation type
            defined in the internal mapping.
    """
    aggregation_methods = {
        Aggregation.MEAN: lambda x: np.mean(x, axis=0),
        Aggregation.MEDIAN: lambda x: np.median(x, axis=0),
        Aggregation.MINIMUM: lambda x: np.min(x, axis=0),
        Aggregation.MAXIMUM: lambda x: np.max(x, axis=0),
    }

    func = aggregation_methods.get(method)
    if not func:
        raise ValueError(f"Unsupported aggregation method: {method}")

    aggregated_scores: np.ndarray = func(scores)
    return aggregated_scores
