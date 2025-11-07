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
    """Compute bandwidth search range from data characteristics."""
    data_range = np.ptp(data)
    data_std = float(np.std(data))
    bw_min = min(data_range * 0.001, data_std * 0.01)
    bw_max = max(data_range * 0.5, data_std * 2)
    return bw_min, bw_max
