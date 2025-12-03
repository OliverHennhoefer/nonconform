"""Bandwidth estimation utilities for kernel density estimation.

This module provides rule-of-thumb bandwidth selectors for KDE.
"""

import numpy as np

# Statistical constants for bandwidth estimation
# Scott (1992): h = 1.06 * sigma * n^(-1/5)
SCOTT_CONSTANT = 1.06

# Silverman (1986): h = 0.9 * min(sigma, IQR/1.349) * n^(-1/5)
SILVERMAN_CONSTANT = 0.9

# IQR to standard deviation conversion factor for normal distribution
# For N(0,1): IQR = Q3 - Q1 = 0.6745 - (-0.6745) = 1.349 * sigma
IQR_TO_STD_FACTOR = 1.349

# Maximum ratio between bandwidth bounds to prevent extreme search ranges
MAX_BANDWIDTH_RATIO = 1000


def _scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule-of-thumb bandwidth estimation.

    Reference: Scott, D.W. (1992). Multivariate Density Estimation.
    """
    n = len(data)
    sigma = np.std(data, ddof=1)
    return SCOTT_CONSTANT * sigma * n ** (-0.2)


def _silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth estimation.

    Reference: Silverman, B.W. (1986). Density Estimation for Statistics
    and Data Analysis.
    """
    n = len(data)
    sigma = np.std(data, ddof=1)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma_hat = min(sigma, iqr / IQR_TO_STD_FACTOR) if iqr > 0 else sigma
    return SILVERMAN_CONSTANT * sigma_hat * n ** (-0.2)


def _sheather_jones_bandwidth(data: np.ndarray) -> float:
    """Sheather-Jones bandwidth selector with fallback to Silverman."""
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(data, bw_method="scott")
        return kde.factor * np.std(data, ddof=1)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # Fallback to Silverman if scipy unavailable or numerical issues occur
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
    robust_std = iqr / IQR_TO_STD_FACTOR if iqr > 0 else float(np.std(data))

    bw_min = min(robust_range * 0.001, robust_std * 0.01)
    bw_max = max(robust_range * 0.5, robust_std * 2)

    # Cap the ratio to prevent extreme search ranges
    if bw_max / max(bw_min, 1e-10) > MAX_BANDWIDTH_RATIO:
        bw_max = bw_min * MAX_BANDWIDTH_RATIO

    return bw_min, bw_max
