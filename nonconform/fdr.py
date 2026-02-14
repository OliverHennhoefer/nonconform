"""False Discovery Rate control for weighted conformal prediction.

This module exposes explicit, non-polymorphic entrypoints for:
- Weighted Conformalized Selection (WCS)
- weighted Benjamini-Hochberg (BH)
"""

import logging

import numpy as np
from tqdm import tqdm

from nonconform.scoring import calculate_weighted_p_val
from nonconform.structures import ConformalResult

from ._internal import Pruning, TieBreakMode, get_logger

_KDE_MONOTONICITY_TOL = 1e-12


def _validate_alpha(alpha: float) -> None:
    """Validate FDR target level."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _as_1d_numeric(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize array-like input into a strict 1D float ndarray."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape!r}.")
    return arr


def _bh_rejection_count(p_values: np.ndarray, thresholds: np.ndarray) -> int:
    """Return size of BH rejection set for given p-values."""
    sorted_p = np.sort(p_values)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _validate_non_negative_finite(name: str, values: np.ndarray) -> None:
    """Validate that an array is finite and non-negative."""
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative.")


def _validate_finite(name: str, values: np.ndarray) -> None:
    """Validate that an array has only finite entries."""
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")


def _validate_p_values(p_values: np.ndarray) -> None:
    """Validate that p-values are finite and within [0, 1]."""
    if p_values.size == 0:
        return
    if not np.all(np.isfinite(p_values)):
        raise ValueError("p_values must be finite.")
    eps = 1e-10
    if np.any((p_values < -eps) | (p_values > 1 + eps)):
        raise ValueError("p_values must be within [0, 1].")


def _validate_pruning(pruning: Pruning) -> None:
    """Validate pruning mode type."""
    if not isinstance(pruning, Pruning):
        raise TypeError(
            f"pruning must be an instance of Pruning. Got {type(pruning).__name__}."
        )


def _calib_weight_mass_at_or_above(
    calib_scores: np.ndarray, w_calib: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute weighted calibration mass at or above each target score."""
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = w_calib[order]
    total_weight = np.sum(sorted_weights)
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="left")
    return total_weight - cum_weights[positions]


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
    calib_mass_at_or_above: np.ndarray,
    scratch: np.ndarray,
    include_self_weight: bool,
    sorted_test_idx: np.ndarray | None,
    le_cutoffs: np.ndarray | None,
) -> int:
    """Compute rejection set size |R_j^{(0)}| for test instance j."""
    np.copyto(scratch, calib_mass_at_or_above)
    if include_self_weight:
        if sorted_test_idx is None or le_cutoffs is None:
            raise ValueError("Internal error: missing score-rank cache for WCS.")
        scratch[sorted_test_idx[: le_cutoffs[j]]] += w_test[j]
        denominator = sum_calib_weight + w_test[j]
    else:
        denominator = sum_calib_weight
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError(
            "Weighted FDR requires positive finite effective calibration mass."
        )
    scratch[j] = 0.0
    scratch /= denominator
    scratch[j] = 0.0
    return _bh_rejection_count(scratch, bh_thresholds)


def _extract_required_wcs_fields(
    result: ConformalResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract required WCS arrays from a result bundle."""
    required = {
        "p_values": result.p_values,
        "test_scores": result.test_scores,
        "calib_scores": result.calib_scores,
        "test_weights": result.test_weights,
        "calib_weights": result.calib_weights,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            "result is missing required WCS fields: "
            f"{missing_list}. Run weighted compute_p_values(...) first."
        )

    p_values = np.asarray(required["p_values"])
    test_scores = np.asarray(required["test_scores"])
    calib_scores = np.asarray(required["calib_scores"])
    test_weights = np.asarray(required["test_weights"])
    calib_weights = np.asarray(required["calib_weights"])
    return p_values, test_scores, calib_scores, test_weights, calib_weights


def _extract_required_bh_p_values(result: ConformalResult) -> np.ndarray:
    """Extract required p-values from a result bundle for weighted BH."""
    if result.p_values is None:
        raise ValueError(
            "result.p_values is required for weighted_bh_from_result(...). "
            "Run compute_p_values(...) first."
        )
    return np.asarray(result.p_values, dtype=float)


def _extract_kde_support(
    result: ConformalResult,
) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, bool]:
    """Extract optional KDE support metadata for probabilistic estimation."""
    if not result.metadata:
        return None, True

    kde_meta = result.metadata.get("kde")
    if kde_meta is None:
        return None, True
    if not isinstance(kde_meta, dict):
        raise ValueError("result.metadata['kde'] must be a dictionary.")

    required_keys = ("eval_grid", "cdf_values", "total_weight")
    missing_keys = [key for key in required_keys if key not in kde_meta]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"result.metadata['kde'] is malformed: missing keys {missing}."
        )

    eval_grid = _as_1d_numeric(
        "result.metadata['kde']['eval_grid']", kde_meta["eval_grid"]
    )
    cdf_values = _as_1d_numeric(
        "result.metadata['kde']['cdf_values']", kde_meta["cdf_values"]
    )
    try:
        total_weight = float(kde_meta["total_weight"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "result.metadata['kde']['total_weight'] must be a finite positive float."
        ) from exc

    if eval_grid.size <= 1:
        raise ValueError(
            "result.metadata['kde']['eval_grid'] must contain at least 2 points."
        )
    if eval_grid.size != cdf_values.size:
        raise ValueError(
            "result.metadata['kde']['eval_grid'] and ['cdf_values'] "
            "must have equal length."
        )
    _validate_finite("result.metadata['kde']['eval_grid']", eval_grid)
    _validate_finite("result.metadata['kde']['cdf_values']", cdf_values)
    if np.any(np.diff(eval_grid) <= 0):
        raise ValueError(
            "result.metadata['kde']['eval_grid'] must be strictly increasing."
        )
    if np.any(np.diff(cdf_values) < -_KDE_MONOTONICITY_TOL):
        raise ValueError("result.metadata['kde']['cdf_values'] must be non-decreasing.")
    eps = 1e-10
    if np.any((cdf_values < -eps) | (cdf_values > 1 + eps)):
        raise ValueError("result.metadata['kde']['cdf_values'] must be within [0, 1].")
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError(
            "result.metadata['kde']['total_weight'] must be a finite positive value."
        )

    return (eval_grid, cdf_values, total_weight), False


def _run_wcs(
    *,
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float,
    pruning: Pruning,
    seed: int | None,
    kde_support: tuple[np.ndarray, np.ndarray, float] | None = None,
    include_self_weight: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run Weighted Conformalized Selection from explicit arrays."""
    _validate_alpha(alpha)
    _validate_pruning(pruning)

    p_vals = _as_1d_numeric("p_values", p_values)
    test_scores_arr = _as_1d_numeric("test_scores", test_scores)
    calib_scores_arr = _as_1d_numeric("calib_scores", calib_scores)
    test_weights_arr = _as_1d_numeric("test_weights", test_weights)
    calib_weights_arr = _as_1d_numeric("calib_weights", calib_weights)

    _validate_p_values(p_vals)
    _validate_finite("test_scores", test_scores_arr)
    _validate_finite("calib_scores", calib_scores_arr)
    _validate_non_negative_finite("test_weights", test_weights_arr)
    _validate_non_negative_finite("calib_weights", calib_weights_arr)

    m = len(test_scores_arr)
    if len(test_weights_arr) != m or len(p_vals) != m:
        raise ValueError(
            "test_scores, test_weights, and p_values must have the same length."
        )
    if len(calib_scores_arr) != len(calib_weights_arr):
        raise ValueError("calib_scores and calib_weights must have the same length.")

    if rng is None:
        rng = np.random.default_rng(seed)

    if kde_support is not None:
        eval_grid, cdf_values, total_weight = kde_support
        sum_calib_weight = total_weight
        calib_mass_at_or_above = sum_calib_weight * (
            1.0
            - np.interp(
                test_scores_arr,
                eval_grid,
                cdf_values,
                left=0.0,
                right=1.0,
            )
        )
    else:
        sum_calib_weight = float(np.sum(calib_weights_arr, dtype=float))
        calib_mass_at_or_above = _calib_weight_mass_at_or_above(
            calib_scores_arr, calib_weights_arr, test_scores_arr
        )
    if not np.isfinite(sum_calib_weight) or sum_calib_weight <= 0.0:
        raise ValueError(
            "Weighted FDR requires positive finite total calibration weight."
        )

    r_sizes = np.zeros(m, dtype=float)
    bh_thresholds = alpha * (np.arange(1, m + 1) / m)
    scratch = np.empty(m, dtype=float)
    sorted_test_idx: np.ndarray | None = None
    le_cutoffs: np.ndarray | None = None
    if include_self_weight:
        sorted_test_idx = np.argsort(test_scores_arr, kind="mergesort")
        sorted_scores = test_scores_arr[sorted_test_idx]
        le_cutoffs = np.searchsorted(sorted_scores, test_scores_arr, side="right")
    logger = get_logger("fdr")
    j_iterator = (
        tqdm(range(m), desc="Weighted FDR Control")
        if logger.isEnabledFor(logging.INFO)
        else range(m)
    )
    for j in j_iterator:
        r_sizes[j] = _compute_rejection_set_size_for_instance(
            j,
            test_scores_arr,
            test_weights_arr,
            sum_calib_weight,
            bh_thresholds,
            calib_mass_at_or_above,
            scratch,
            include_self_weight=include_self_weight,
            sorted_test_idx=sorted_test_idx,
            le_cutoffs=le_cutoffs,
        )

    thresholds = alpha * r_sizes / m
    first_sel_idx = np.flatnonzero(p_vals <= thresholds)
    if len(first_sel_idx) == 0:
        return np.zeros(m, dtype=bool)

    sizes_sel = r_sizes[first_sel_idx]
    if pruning is Pruning.HETEROGENEOUS:
        final_sel_idx = _prune_heterogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning is Pruning.HOMOGENEOUS:
        final_sel_idx = _prune_homogeneous(first_sel_idx, sizes_sel, rng)
    else:
        final_sel_idx = _prune_deterministic(first_sel_idx, sizes_sel)

    final_sel_mask = np.zeros(m, dtype=bool)
    final_sel_mask[final_sel_idx] = True
    return final_sel_mask


def weighted_false_discovery_control(
    result: ConformalResult,
    *,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Perform WCS from a strict ConformalResult bundle."""
    p_values, test_scores, calib_scores, test_weights, calib_weights = (
        _extract_required_wcs_fields(result)
    )
    kde_support, use_self_weight = _extract_kde_support(result)
    return _run_wcs(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=pruning,
        seed=seed,
        kde_support=kde_support,
        include_self_weight=use_self_weight,
    )


def weighted_false_discovery_control_from_arrays(
    *,
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Perform WCS from explicit weighted arrays and precomputed p-values."""
    return _run_wcs(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=pruning,
        seed=seed,
    )


def weighted_false_discovery_control_empirical(
    *,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Perform WCS from explicit arrays with empirical randomized p-values."""
    rng = np.random.default_rng(seed)
    p_values = calculate_weighted_p_val(
        scores=np.asarray(test_scores),
        calibration_set=np.asarray(calib_scores),
        test_weights=np.asarray(test_weights),
        calib_weights=np.asarray(calib_weights),
        tie_break=TieBreakMode.RANDOMIZED,
        rng=rng,
    )
    return _run_wcs(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=pruning,
        seed=seed,
        rng=rng,
    )


def weighted_bh(p_values: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    """Apply weighted Benjamini-Hochberg to precomputed p-values."""
    _validate_alpha(alpha)

    p_values_arr = _as_1d_numeric("p_values", p_values)
    _validate_p_values(p_values_arr)

    m = len(p_values_arr)
    if m == 0:
        return np.zeros(0, dtype=bool)

    sorted_idx = np.argsort(p_values_arr)
    sorted_p = p_values_arr[sorted_idx]
    adjusted_sorted = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[
        ::-1
    ]

    adjusted_p_values = np.empty(m)
    adjusted_p_values[sorted_idx] = adjusted_sorted
    return adjusted_p_values <= alpha


def weighted_bh_from_result(
    result: ConformalResult, *, alpha: float = 0.05
) -> np.ndarray:
    """Apply weighted BH from a strict ConformalResult bundle."""
    p_values = _extract_required_bh_p_values(result)
    return weighted_bh(p_values, alpha=alpha)


def weighted_bh_empirical(
    *,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Apply weighted BH after empirical randomized p-value computation."""
    p_values = calculate_weighted_p_val(
        scores=np.asarray(test_scores),
        calibration_set=np.asarray(calib_scores),
        test_weights=np.asarray(test_weights),
        calib_weights=np.asarray(calib_weights),
        tie_break=TieBreakMode.RANDOMIZED,
        rng=np.random.default_rng(seed),
    )
    return weighted_bh(p_values, alpha=alpha)


__all__ = [
    "Pruning",
    "weighted_bh",
    "weighted_bh_empirical",
    "weighted_bh_from_result",
    "weighted_false_discovery_control",
    "weighted_false_discovery_control_empirical",
    "weighted_false_discovery_control_from_arrays",
]
