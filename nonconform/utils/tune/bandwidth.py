import numpy as np


def _scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule-of-thumb bandwidth estimation."""
    n = len(data)
    sigma = np.std(data, ddof=1)
    return 1.06 * sigma * n ** (-0.2)


def _silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth estimation."""
    n = len(data)
    sigma = np.std(data, ddof=1)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma_hat = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    return 0.9 * sigma_hat * n ** (-0.2)


def _sheather_jones_bandwidth(data: np.ndarray) -> float:
    """Sheather-Jones bandwidth selector with fallback to Silverman."""
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(data, bw_method="scott")
        return kde.factor * np.std(data, ddof=1)
    except Exception:
        return _silverman_bandwidth(data)


def compute_bandwidth_range(data: np.ndarray) -> tuple[float, float]:
    """Compute bandwidth search range using robust statistics.

    Uses percentile-based range and IQR-based spread to be robust against
    outliers that can cause extreme bandwidth ranges.
    """
    # Robust range: use percentiles instead of min/max to ignore outliers
    q1, q99 = np.percentile(data, [1, 99])
    robust_range = q99 - q1

    # Robust spread: IQR-based (same approach as Silverman's rule)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    robust_std = iqr / 1.349 if iqr > 0 else float(np.std(data))

    bw_min = min(robust_range * 0.001, robust_std * 0.01)
    bw_max = max(robust_range * 0.5, robust_std * 2)

    # Cap the ratio to prevent extreme search ranges
    max_ratio = 1000
    if bw_max / max(bw_min, 1e-10) > max_ratio:
        bw_max = bw_min * max_ratio

    return bw_min, bw_max
