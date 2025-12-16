"""Calibration-conditional p-value adjustment functions.

This module provides adjustment functions for converting marginal conformal p-values
to calibration-conditional p-values with high-probability validity guarantees.

Based on: Bates et al. (2023) "Testing for Outliers with Conformal P-values"
"""

import numpy as np


def _ensure_monotonic(b: np.ndarray, min_val: float = 1e-15) -> np.ndarray:
    """Ensure adjustment sequence is monotonically non-decreasing and positive.

    Args:
        b: Adjustment sequence array (modified in place).
        min_val: Minimum allowed value for any element.

    Returns:
        The modified array with enforced monotonicity and positivity.
    """
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1])
    return np.maximum(b, min_val)


def compute_simes_sequence(n: int, delta: float, k: int | None = None) -> np.ndarray:
    """Compute adjustment sequence using generalized Simes inequality.

    The generalized Simes inequality provides exact finite-sample validity
    with tight bounds for small p-values.

    Formula:
        b_{n+1-i} = 1 - delta^{1/k} * (i*(i-1)*...*(i-k+1) / n*...*(n-k+1))^{1/k}
        for i = k, ..., n (for i < k, the product is 0 so b = 1).

    Args:
        n: Number of calibration samples.
        delta: Miscoverage probability (1-delta = coverage level).
        k: Number of terms in Simes bound. Defaults to n // 2
            (optimal for small p-values).

    Returns:
        Array of length n with adjustment values b_1, ..., b_n (non-decreasing).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    if k is None:
        k = max(1, n // 2)
    k = min(k, n)  # k cannot exceed n

    b = np.ones(n, dtype=np.float64)

    # Compute b_{n+1-i} for i = k, ..., n
    # For i < k, the product includes non-positive factors, so b = 1
    # b_{n+1-i} = 1 - delta^{1/k} * (product ratio)^{1/k}
    delta_power = delta ** (1.0 / k)

    # Compute denominator (fixed): n * (n-1) * ... * (n-k+1)
    log_den = 0.0
    for j in range(k):
        log_den += np.log(n - j)

    for i in range(k, n + 1):
        # Compute numerator: i * (i-1) * ... * (i-k+1) (k factors)
        log_num = 0.0
        for j in range(k):
            log_num += np.log(i - j)

        product_ratio = np.exp((log_num - log_den) / k)
        b_idx = n - i  # Index for b_{n+1-i} in 0-based indexing
        b[b_idx] = 1.0 - delta_power * product_ratio

    return _ensure_monotonic(b)


def compute_asymptotic_sequence(n: int, delta: float) -> np.ndarray:
    """Compute adjustment sequence using asymptotic (Eicker) bound.

    Uses the asymptotic confidence band for the empirical CDF based on
    the Brownian bridge limiting distribution.

    Formula: b_i = min(i/n + c_n(delta) * sqrt(i*(n-i)) / (n*sqrt(n)), 1)
    where c_n(delta) is derived from the Eicker bound.

    Args:
        n: Number of calibration samples.
        delta: Miscoverage probability (1-delta = coverage level).

    Returns:
        Array of length n with adjustment values b_1, ..., b_n (non-decreasing).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    # Compute c_n(delta) using Eicker's formula (see docstring for full formula)
    # For small n, use fallback to avoid log(log(n)) issues
    if n <= 2:
        # Fallback to DKW-style bound for very small n
        c_n = np.sqrt(np.log(2.0 / delta) / 2.0)
    else:
        log_n = np.log(n)
        log_log_n = np.log(log_n)

        if log_log_n <= 0:
            # n <= e, use DKW fallback
            c_n = np.sqrt(np.log(2.0 / delta) / 2.0)
        else:
            log_log_log_n = np.log(max(log_log_n, 1e-10))
            numerator = (
                -np.log(-np.log(1 - delta))
                + 2.0 * log_log_n
                + 0.5 * log_log_log_n
                - 0.5 * np.log(np.pi)
            )
            denominator = np.sqrt(2.0 * log_log_n)
            c_n = numerator / denominator

    # Compute b_i for i = 1, ..., n
    i_vals = np.arange(1, n + 1, dtype=np.float64)
    base = i_vals / n
    correction = c_n * np.sqrt(i_vals * (n - i_vals)) / (n * np.sqrt(n))

    b = np.minimum(base + correction, 1.0)

    return _ensure_monotonic(b)


def compute_monte_carlo_sequence(
    n: int,
    delta: float,
    n_simulations: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Compute adjustment sequence via Monte Carlo calibration.

    Combines Simes and asymptotic bounds, then calibrates via simulation
    to achieve exact (1-delta) coverage. This provides exact finite-sample
    validity with efficiency close to the asymptotic bound.

    Args:
        n: Number of calibration samples.
        delta: Miscoverage probability (1-delta = coverage level).
        n_simulations: Number of Monte Carlo samples for calibration.
        seed: Random seed for reproducibility.

    Returns:
        Array of length n with adjustment values b_1, ..., b_n (non-decreasing).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if n_simulations < 100:
        raise ValueError(f"n_simulations must be at least 100, got {n_simulations}")

    rng = np.random.default_rng(seed)

    # Get Simes sequence (exact, tight for small p-values)
    b_simes = compute_simes_sequence(n, delta)

    # For very small n, just use Simes
    if n <= 10:
        return b_simes

    # Function to compute combined sequence for a given delta_hat
    def compute_combined_sequence(delta_hat: float) -> np.ndarray:
        b_asym = compute_asymptotic_sequence(n, delta_hat)
        # Take element-wise minimum (tighter bound at each position)
        b_combined = np.minimum(b_simes, b_asym)
        return _ensure_monotonic(b_combined)

    # Function to estimate coverage probability via Monte Carlo
    def estimate_coverage(b_seq: np.ndarray) -> float:
        # Simulate uniform order statistics
        uniforms = rng.random((n_simulations, n))
        uniforms.sort(axis=1)  # Order statistics U_(1), ..., U_(n)

        # Check if all U_(i) <= b_i for each simulation
        # Coverage = fraction of simulations where all constraints hold
        violations = np.any(uniforms > b_seq, axis=1)
        coverage = 1.0 - np.mean(violations)
        return coverage

    # Binary search for delta_hat that achieves target coverage (1-delta)
    target_coverage = 1.0 - delta
    delta_low, delta_high = 1e-10, 1.0 - 1e-10
    tolerance = 0.001  # Stop when coverage is within this tolerance

    # Initial check with Simes (should achieve at least 1-delta by theory)
    simes_coverage = estimate_coverage(b_simes)
    if simes_coverage >= target_coverage - tolerance:
        # Simes alone achieves target, try to tighten with asymptotic
        pass

    best_b = b_simes.copy()

    for _ in range(20):  # Max 20 bisection iterations
        delta_mid = (delta_low + delta_high) / 2.0
        b_mid = compute_combined_sequence(delta_mid)
        coverage_mid = estimate_coverage(b_mid)

        if coverage_mid >= target_coverage:
            # We have enough coverage, try to tighten (increase delta_hat)
            delta_low = delta_mid
            best_b = b_mid.copy()
        else:
            # Not enough coverage, need to be more conservative (decrease delta_hat)
            delta_high = delta_mid

        if abs(coverage_mid - target_coverage) < tolerance:
            best_b = b_mid.copy()
            break

    return best_b


def apply_adjustment(
    p_marginal: np.ndarray,
    adjustment_sequence: np.ndarray,
    n_calib: int,
) -> np.ndarray:
    """Map marginal p-values to calibration-conditional p-values.

    Uses the step function: p_cond = b[ceil((n+1) * p_marg)]
    where b is the adjustment sequence indexed from 0 to n.

    Args:
        p_marginal: Array of marginal conformal p-values.
        adjustment_sequence: Array b_1, ..., b_n from adjustment method.
        n_calib: Number of calibration samples.

    Returns:
        Array of calibration-conditional p-values (always >= marginal p-values).
    """
    p_marginal = np.asarray(p_marginal, dtype=np.float64)

    # Marginal p-values are in {1/(n+1), 2/(n+1), ..., (n+1)/(n+1)}
    # For p_marg = k/(n+1), we want b_k (using 1-based indexing)
    # Index into adjustment_sequence (0-based): k - 1

    # Compute indices: ceil((n+1) * p_marg) - 1 for 0-based indexing
    indices = np.ceil((n_calib + 1) * p_marginal).astype(int) - 1

    # Clip indices to valid range [0, n-1]
    indices = np.clip(indices, 0, len(adjustment_sequence) - 1)

    # Look up adjusted p-values
    p_conditional = adjustment_sequence[indices]

    # Ensure p_cond >= p_marg (should always hold by construction)
    p_conditional = np.maximum(p_conditional, p_marginal)

    # Clip to [0, 1]
    p_conditional = np.clip(p_conditional, 0.0, 1.0)

    return p_conditional


__all__ = [
    "apply_adjustment",
    "compute_asymptotic_sequence",
    "compute_monte_carlo_sequence",
    "compute_simes_sequence",
]
