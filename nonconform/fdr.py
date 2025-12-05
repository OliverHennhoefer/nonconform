"""False Discovery Rate control for conformal prediction.

This module implements Weighted Conformalized Selection (WCS) for FDR control
under covariate shift. For standard BH/BY procedures, use
scipy.stats.false_discovery_control.

Functions:
    weighted_false_discovery_control: Perform Weighted Conformalized Selection.
    weighted_bh: Apply weighted Benjamini-Hochberg procedure.
"""

import logging

import numpy as np
from tqdm import tqdm

from nonconform.scoring import calculate_weighted_p_val
from nonconform.structures import ConformalResult

from ._internal import Pruning, get_logger


def _bh_rejection_indices(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return indices of BH rejection set for given p-values."""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return np.array([], dtype=int)
    k = below[-1]
    return sorted_idx[: k + 1]


def _bh_rejection_count(p_values: np.ndarray, thresholds: np.ndarray) -> int:
    """Return size of BH rejection set for given p-values."""
    sorted_p = np.sort(p_values)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _calib_weight_mass_below(
    calib_scores: np.ndarray, w_calib: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute weighted calibration mass strictly below each target score."""
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = w_calib[order]
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="left")
    return cum_weights[positions]


def _compute_r_star(metrics: np.ndarray) -> int:
    """Return the largest r s.t. #{j : metrics_j <= r} >= r."""
    if metrics.size == 0:
        return 0
    sorted_metrics = np.sort(metrics)
    for k in range(sorted_metrics.size, 0, -1):
        if sorted_metrics[k - 1] <= k:
            return k
    return 0


def _select_with_metrics(first_sel_idx: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    """Select indices whose metric satisfies the r_* threshold."""
    r_star = _compute_r_star(metrics)
    if r_star == 0:
        return np.array([], dtype=int)
    selected = first_sel_idx[metrics <= r_star]
    return np.sort(selected)


def _prune_heterogeneous(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Heterogeneous pruning with independent random variables."""
    xi = rng.uniform(size=len(first_sel_idx))
    metrics = xi * sizes_sel
    return _select_with_metrics(first_sel_idx, metrics)


def _prune_homogeneous(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Homogeneous pruning with shared random variable."""
    xi = rng.uniform()
    metrics = xi * sizes_sel
    return _select_with_metrics(first_sel_idx, metrics)


def _prune_deterministic(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray
) -> np.ndarray:
    """Deterministic pruning based on rejection set sizes."""
    metrics = sizes_sel.astype(float)
    return _select_with_metrics(first_sel_idx, metrics)


def _compute_rejection_set_size_for_instance(
    j: int,
    test_scores: np.ndarray,
    w_test: np.ndarray,
    sum_calib_weight: float,
    bh_thresholds: np.ndarray,
    calib_mass_below: np.ndarray,
    scratch: np.ndarray,
    include_self_weight: bool,
) -> int:
    """Compute rejection set size |R_j^{(0)}| for test instance j."""
    np.copyto(scratch, calib_mass_below)
    if include_self_weight:
        scratch += w_test[j] * (test_scores[j] < test_scores)
        denominator = sum_calib_weight + w_test[j]
    else:
        denominator = sum_calib_weight
    scratch[j] = 0.0
    scratch /= denominator
    scratch[j] = 0.0
    return _bh_rejection_count(scratch, bh_thresholds)


def weighted_false_discovery_control(
    result: ConformalResult | None = None,
    *,
    p_values: np.ndarray | None = None,
    alpha: float = 0.05,
    test_scores: np.ndarray | None = None,
    calib_scores: np.ndarray | None = None,
    test_weights: np.ndarray | None = None,
    calib_weights: np.ndarray | None = None,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Perform Weighted Conformalized Selection (WCS).

    Args:
        result: Optional conformal result bundle (from ConformalDetector.last_result).
            When provided, remaining parameters default to the contents of this object.
        p_values: Weighted conformal p-values. If None, computed internally.
        alpha: Target false discovery rate (0 < alpha < 1). Defaults to 0.05.
        test_scores: Non-conformity scores for test data.
        calib_scores: Non-conformity scores for calibration data.
        test_weights: Importance weights for test data.
        calib_weights: Importance weights for calibration data.
        pruning: Pruning method. Defaults to Pruning.DETERMINISTIC.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        Boolean mask of test points retained after pruning.

    Raises:
        ValueError: If alpha is outside (0, 1) or required inputs are missing.

    Note:
        The procedure follows Algorithm 1 in Jin & Candes (2023).
        Computational cost is O(m^2) in the number of test points.

    References:
        Jin, Y., & Candes, E. (2023). Model-free selective inference under
        covariate shift via weighted conformal p-values. arXiv preprint
        arXiv:2307.09291.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if result is not None:
        if result.p_values is not None and p_values is None:
            p_values = result.p_values
        if result.test_scores is not None and test_scores is None:
            test_scores = result.test_scores
        if result.calib_scores is not None and calib_scores is None:
            calib_scores = result.calib_scores
        if result.test_weights is not None and test_weights is None:
            test_weights = result.test_weights
        if result.calib_weights is not None and calib_weights is None:
            calib_weights = result.calib_weights

    kde_support: tuple[np.ndarray, np.ndarray, float] | None = None
    use_self_weight = True
    if result is not None and result.metadata:
        kde_meta = result.metadata.get("kde")
        if kde_meta is not None:
            try:
                eval_grid = np.asarray(kde_meta["eval_grid"])
                cdf_values = np.asarray(kde_meta["cdf_values"])
                total_weight = float(kde_meta["total_weight"])
                if (
                    eval_grid.ndim == 1
                    and cdf_values.ndim == 1
                    and eval_grid.size == cdf_values.size
                    and eval_grid.size > 1
                ):
                    kde_support = (eval_grid, cdf_values, total_weight)
                    use_self_weight = False
            except KeyError:
                kde_support = None

    required_arrays = (test_scores, calib_scores, test_weights, calib_weights)
    if any(arr is None for arr in required_arrays):
        raise ValueError(
            "test_scores, calib_scores, test_weights, and calib_weights "
            "must all be provided."
        )
    test_scores = np.asarray(test_scores)
    calib_scores = np.asarray(calib_scores)
    test_weights = np.asarray(test_weights)
    calib_weights = np.asarray(calib_weights)

    if p_values is None:
        p_vals = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
    else:
        p_vals = np.asarray(p_values)
        if p_vals.ndim != 1:
            raise ValueError(f"p_values must be a 1D array, got shape {p_vals.shape}.")

    m = len(test_scores)
    if len(test_weights) != m or len(p_vals) != m:
        raise ValueError(
            "test_scores, test_weights, and p_values must have the same length."
        )
    if len(calib_scores) != len(calib_weights):
        raise ValueError("calib_scores and calib_weights must have the same length.")

    rng = np.random.default_rng(seed)

    if kde_support is not None:
        eval_grid, cdf_values, total_weight = kde_support
        sum_calib_weight = total_weight
        calib_mass_below = sum_calib_weight * np.interp(
            test_scores,
            eval_grid,
            cdf_values,
            left=0.0,
            right=1.0,
        )
    else:
        sum_calib_weight = np.sum(calib_weights)
        calib_mass_below = _calib_weight_mass_below(
            calib_scores, calib_weights, test_scores
        )

    # Compute R_j^{(0)} sizes and thresholds
    r_sizes = np.zeros(m, dtype=float)
    bh_thresholds = alpha * (np.arange(1, m + 1) / m)
    scratch = np.empty(m, dtype=float)
    logger = get_logger("fdr")
    j_iterator = (
        tqdm(range(m), desc="Weighted FDR Control")
        if logger.isEnabledFor(logging.INFO)
        else range(m)
    )
    for j in j_iterator:
        r_sizes[j] = _compute_rejection_set_size_for_instance(
            j,
            test_scores,
            test_weights,
            sum_calib_weight,
            bh_thresholds,
            calib_mass_below,
            scratch,
            include_self_weight=use_self_weight,
        )

    thresholds = alpha * r_sizes / m
    first_sel_idx = np.flatnonzero(p_vals <= thresholds)

    if len(first_sel_idx) == 0:
        return np.zeros(m, dtype=bool)

    sizes_sel = r_sizes[first_sel_idx]
    if pruning == Pruning.HETEROGENEOUS:
        final_sel_idx = _prune_heterogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == Pruning.HOMOGENEOUS:
        final_sel_idx = _prune_homogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == Pruning.DETERMINISTIC:
        final_sel_idx = _prune_deterministic(first_sel_idx, sizes_sel)
    else:
        raise ValueError(f"Unknown pruning method '{pruning}'.")

    final_sel_mask = np.zeros(m, dtype=bool)
    final_sel_mask[final_sel_idx] = True

    return final_sel_mask


def weighted_bh(
    result: ConformalResult,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply weighted Benjamini-Hochberg procedure.

    Uses estimator-supplied weighted p-values when available and falls back on
    recomputing them with the standard weighted conformal formula otherwise.

    Args:
        result: Conformal result bundle with test/calib scores and weights.
        alpha: Target false discovery rate (0 < alpha < 1). Defaults to 0.05.

    Returns:
        Boolean array indicating discoveries for each test point.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if result is None:
        raise ValueError("weighted_bh requires a ConformalResult instance.")

    p_values = result.p_values
    if p_values is not None:
        p_values = np.asarray(p_values, dtype=float)
    else:
        required = {
            "test_scores": result.test_scores,
            "calib_scores": result.calib_scores,
            "test_weights": result.test_weights,
            "calib_weights": result.calib_weights,
        }
        missing = [name for name, arr in required.items() if arr is None]
        if missing:
            raise ValueError(
                "Cannot recompute weighted p-values; missing: " + ", ".join(missing)
            )
        p_values = calculate_weighted_p_val(
            np.asarray(required["test_scores"]),
            np.asarray(required["calib_scores"]),
            np.asarray(required["test_weights"]),
            np.asarray(required["calib_weights"]),
        )

    if p_values.ndim != 1:
        raise ValueError(f"p_values must be a 1D array, got shape {p_values.shape!r}.")

    m = len(p_values)
    if m == 0:
        return np.zeros(0, dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[
        ::-1
    ]

    adjusted_p_values = np.empty(m)
    adjusted_p_values[sorted_idx] = adjusted_sorted

    return adjusted_p_values <= alpha


__all__ = [
    "Pruning",
    "weighted_bh",
    "weighted_false_discovery_control",
]
