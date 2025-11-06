"""False Discovery Rate control for conformal prediction.

This module implements Weighted Conformalized Selection (WCS) for FDR control
under covariate shift. For standard BH/BY procedures, use
scipy.stats.false_discovery_control.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from nonconform.utils.func.enums import Pruning
from nonconform.utils.func.logger import get_logger
from nonconform.utils.stat.results import ConformalResult
from nonconform.utils.stat.statistical import calculate_weighted_p_val


def _bh_rejection_indices(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return indices of BH rejection set for given p-values.

    This helper mimics the Benjamini-Hochberg procedure: sort p-values,
    find the largest k such that p_(k) ≤ q*k/m, and return the first k
    indices in the sorted order.

    Args:
        p_values: Array of p-values to apply BH procedure on.
        q: Target false discovery rate threshold.

    Returns:
        Array of indices in the rejection set. Returns empty array if no
        p-value meets the criterion.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=int)
    # Sort indices by p-value
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    # Thresholds q * (1:m) / m
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return np.array([], dtype=int)
    k = below[-1]
    return sorted_idx[: k + 1]


def _bh_rejection_count(p_values: np.ndarray, thresholds: np.ndarray) -> int:
    """Return size of BH rejection set for given p-values."""
    if p_values.size == 0:
        return 0
    sorted_p = np.sort(p_values)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _calib_weight_mass_below(
    calib_scores: np.ndarray, w_calib: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute weighted calibration mass strictly below each target score."""
    if len(calib_scores) == 0:
        return np.zeros_like(targets, dtype=float)
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = w_calib[order]
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="left")
    return cum_weights[positions]


def _compute_r_star(metrics: np.ndarray) -> int:
    """Return the largest r s.t. #{j : metrics_j ≤ r} ≥ r."""
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
    """Heterogeneous pruning with independent random variables.

    Uses independent ξ_j ∈ [0,1] for each candidate and applies the
    thresholding rule from eq. (7) of the weighted conformal paper.
    """
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
    """Compute rejection set size |R_j^{(0)}| for test instance j.

    For a given test instance j, computes auxiliary p-values for all other
    test instances and applies the Benjamini-Hochberg procedure to determine
    the rejection set size.

    Args:
        j: Index of the test instance to compute rejection set size for.
        test_scores: Non-conformity scores for all test instances.
        w_test: Importance weights for all test instances.
        sum_calib_weight: Sum of all calibration weights (precomputed).
        bh_thresholds: Precomputed BH thresholds q * (1:m) / m.
        calib_mass_below: Weighted calibration mass strictly below each test score.
        scratch: Workspace array for computing auxiliary p-values.
        include_self_weight: Whether to include the test instance's own weight
            when computing auxiliary p-values. If True, incorporates w_test[j]
            into both numerator and denominator; if False, uses only calibration
            weights.

    Returns:
        Size of the rejection set |R_j^{(0)}| for instance j.
    """
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
        result: Optional conformal result bundle carrying cached arrays (as
            produced by ``ConformalDetector.last_result``). When provided,
            remaining parameters default to the contents of this object.
        p_values: Weighted conformal p-values for the test data. If None, the
            values are computed internally using `test_scores`, `calib_scores`,
            `test_weights`, and `calib_weights`.
        alpha: Target false discovery rate (0 < alpha < 1). Defaults to 0.05.
        test_scores: Non-conformity scores for the test data (length m).
        calib_scores: Non-conformity scores for the calibration data (length n).
        test_weights: Importance weights for the test data (length m).
        calib_weights: Importance weights for the calibration data (length n).
        pruning: Pruning method. ``'hete'`` (heterogeneous pruning) uses
            independent random variables l_j; ``'homo'`` (homogeneous
            pruning) uses a single random variable l shared across
            candidates; ``'dtm'`` (deterministic) performs deterministic
            pruning based on |R_j^{(0)}|. Defaults to ``'dtm'``.
        seed: Random seed for reproducibility. Defaults to None
            (non-deterministic).

    Returns:
        Boolean mask of test points retained after pruning (final selection).
        For deterministic pruning (``'dtm'``), this may coincide with the
        first selection step.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1) or required inputs are missing.

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

    need_scores = p_values is None

    if need_scores:
        if any(
            arr is None
            for arr in (test_scores, calib_scores, test_weights, calib_weights)
        ):
            raise ValueError(
                "test_scores, calib_scores, test_weights, and calib_weights "
                "must be provided when p_values is None."
            )
        test_scores = np.asarray(test_scores)
        calib_scores = np.asarray(calib_scores)
        test_weights = np.asarray(test_weights)
        calib_weights = np.asarray(calib_weights)
        p_vals = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
    else:
        p_vals = np.asarray(p_values)
        if p_vals.ndim != 1:
            raise ValueError(f"p_values must be a 1D array, got shape {p_vals.shape}.")
        if any(
            arr is None
            for arr in (test_scores, calib_scores, test_weights, calib_weights)
        ):
            raise ValueError(
                "test_scores, calib_scores, test_weights, and calib_weights "
                "must be provided when supplying p_values."
            )
        test_scores = np.asarray(test_scores)
        calib_scores = np.asarray(calib_scores)
        test_weights = np.asarray(test_weights)
        calib_weights = np.asarray(calib_weights)

    m = len(test_scores)
    if len(test_weights) != m or len(p_vals) != m:
        raise ValueError(
            "test_scores, test_weights, and p_values must have the same length."
        )
    if len(calib_scores) != len(calib_weights):
        raise ValueError("calib_scores and calib_weights must have the same length.")

    if seed is None:
        # Draw entropy from OS to seed the generator explicitly for lint compliance.
        seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)

    # Precompute constants
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

    # Step 2: compute R_j^{(0)} sizes and thresholds s_j
    r_sizes = np.zeros(m, dtype=float)
    bh_thresholds = alpha * (np.arange(1, m + 1) / m)
    scratch = np.empty(m, dtype=float)
    logger = get_logger("utils.stat.weighted_fdr")
    j_iterator = (
        tqdm(
            range(m),
            desc="Weighted FDR Control",
        )
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

    # Compute thresholds s_j = q * |R_j^{(0)}| / m
    thresholds = alpha * r_sizes / m

    # Step 3: first selection set R^{(1)}
    first_sel_idx = np.flatnonzero(p_vals <= thresholds)

    # If no points selected, return early with empty boolean mask
    if len(first_sel_idx) == 0:
        final_sel_mask = np.zeros(m, dtype=bool)
        return final_sel_mask

    # Step 4: pruning
    # For pruning, we need |R_j^{(0)}| for each j in first_sel_idx
    sizes_sel = r_sizes[first_sel_idx]
    if pruning == Pruning.HETEROGENEOUS:
        final_sel_idx = _prune_heterogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == Pruning.HOMOGENEOUS:
        final_sel_idx = _prune_homogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == Pruning.DETERMINISTIC:
        final_sel_idx = _prune_deterministic(first_sel_idx, sizes_sel)
    else:
        raise ValueError(
            f"Unknown pruning method '{pruning}'. "
            f"Use 'heterogeneous', 'homogeneous' or 'deterministic'."
        )

    # Convert indices to boolean mask
    final_sel_mask = np.zeros(m, dtype=bool)
    final_sel_mask[final_sel_idx] = True

    return final_sel_mask


def weighted_bh(
    result: ConformalResult,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply weighted Benjamini-Hochberg procedure.

    Computes weighted conformal p-values and applies the Benjamini-Hochberg
    procedure to return discovery decisions for FDR control under covariate shift.
    Similar to scipy.stats.false_discovery_control but with importance weighting.

    Args:
        result: Conformal result bundle with test/calib scores and weights.
        alpha: Target false discovery rate (0 < alpha < 1). Defaults to 0.05.

    Returns:
        Boolean array indicating discoveries for each test point.

    References:
        Jin, Y., & Candes, E. (2023). Model-free selective inference under
        covariate shift via weighted conformal p-values. arXiv preprint
        arXiv:2307.09291.
    """
    test_scores = result.test_scores
    calib_scores = result.calib_scores
    test_weights = result.test_weights
    calib_weights = result.calib_weights

    m = len(test_scores)
    sum_calib_weight = np.sum(calib_weights)

    # Compute weighted conformal p-values (strict <, no tie-breaking)
    weighted_mass_below = np.sum(
        (calib_scores >= test_scores[:, np.newaxis]) * calib_weights,
        axis=1,
    )

    p_values = weighted_mass_below / (sum_calib_weight + test_weights)

    # Apply BH procedure to get adjusted p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Compute adjusted p-values: p_adj[k] = min(p[j] * m / (j+1) for j >= k)
    adjusted_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[
        ::-1
    ]

    # Reorder back to original order
    adjusted_p_values = np.empty(m)
    adjusted_p_values[sorted_idx] = adjusted_sorted

    # Return boolean decisions
    return adjusted_p_values <= alpha
