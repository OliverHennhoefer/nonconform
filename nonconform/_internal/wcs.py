"""Private extraction, validation, and pruning for weighted conformal selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from nonconform.structures import ConformalResult

from .constants import Pruning
from .log_utils import get_logger
from .validation import (
    as_1d_numeric,
    validate_finite,
    validate_non_negative_finite,
    validate_p_values,
)

_KDE_MONOTONICITY_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class _WCSInputs:
    """Validated arrays used by the WCS computation."""

    p_values: np.ndarray
    test_scores: np.ndarray
    calib_scores: np.ndarray
    test_weights: np.ndarray
    calib_weights: np.ndarray


def extract_required_fields(
    result: ConformalResult | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract required WCS arrays from a result bundle."""
    result = _require_result_bundle(result)
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

    return (
        np.asarray(required["p_values"]),
        np.asarray(required["test_scores"]),
        np.asarray(required["calib_scores"]),
        np.asarray(required["test_weights"]),
        np.asarray(required["calib_weights"]),
    )


def extract_kde_support(
    result: ConformalResult | None,
) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, bool]:
    """Extract optional KDE support metadata for probabilistic estimation."""
    result = _require_result_bundle(result)
    kde_metadata = _kde_metadata(result)
    if kde_metadata is None:
        return None, True

    eval_grid = as_1d_numeric(
        "result.metadata['kde']['eval_grid']",
        kde_metadata["eval_grid"],
    )
    cdf_values = as_1d_numeric(
        "result.metadata['kde']['cdf_values']",
        kde_metadata["cdf_values"],
    )
    total_weight = _as_total_weight(kde_metadata["total_weight"])
    _validate_kde_arrays(eval_grid, cdf_values, total_weight)
    return (eval_grid, cdf_values, total_weight), False


def run(
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
    inputs = _normalize_inputs(
        p_values,
        test_scores,
        calib_scores,
        test_weights,
        calib_weights,
    )
    if rng is None:
        rng = np.random.default_rng(seed)

    sum_calib_weight, calib_mass = _calibration_mass(inputs, kde_support)
    rejection_sizes = _rejection_sizes(
        inputs,
        alpha=alpha,
        sum_calib_weight=sum_calib_weight,
        calib_mass_strictly_above=calib_mass,
        include_self_weight=include_self_weight,
    )
    first_selection = np.flatnonzero(
        inputs.p_values <= alpha * rejection_sizes / inputs.test_scores.size
    )
    if first_selection.size == 0:
        return np.zeros(inputs.test_scores.size, dtype=bool)

    final_indices = _prune(
        first_selection,
        rejection_sizes[first_selection],
        pruning,
        rng,
    )
    selected = np.zeros(inputs.test_scores.size, dtype=bool)
    selected[final_indices] = True
    return selected


def _require_result_bundle(result: ConformalResult | None) -> ConformalResult:
    """Require a result bundle for the result-aware WCS API."""
    if result is None:
        raise ValueError(
            "result must be a ConformalResult, got None. Run compute_p_values(...) "
            "before calling weighted_false_discovery_control()."
        )
    return result


def _kde_metadata(result: ConformalResult) -> dict[str, object] | None:
    """Return a structurally valid KDE metadata mapping, if present."""
    if not result.metadata:
        return None
    kde_metadata = result.metadata.get("kde")
    if kde_metadata is None:
        return None
    if not isinstance(kde_metadata, dict):
        raise ValueError("result.metadata['kde'] must be a dictionary.")

    required_keys = ("eval_grid", "cdf_values", "total_weight")
    missing_keys = [key for key in required_keys if key not in kde_metadata]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"result.metadata['kde'] is malformed: missing keys {missing}."
        )
    return kde_metadata


def _as_total_weight(value: object) -> float:
    """Normalize KDE calibration mass."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "result.metadata['kde']['total_weight'] must be a finite positive float."
        ) from exc


def _validate_kde_arrays(
    eval_grid: np.ndarray,
    cdf_values: np.ndarray,
    total_weight: float,
) -> None:
    """Validate cached KDE support used by WCS."""
    if eval_grid.size <= 1:
        raise ValueError(
            "result.metadata['kde']['eval_grid'] must contain at least 2 points."
        )
    if eval_grid.size != cdf_values.size:
        raise ValueError(
            "result.metadata['kde']['eval_grid'] and ['cdf_values'] "
            "must have equal length."
        )
    validate_finite("result.metadata['kde']['eval_grid']", eval_grid)
    validate_finite("result.metadata['kde']['cdf_values']", cdf_values)
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


def _validate_alpha(alpha: float) -> None:
    """Validate FDR target level."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _validate_pruning(pruning: Pruning) -> None:
    """Validate pruning mode type."""
    if not isinstance(pruning, Pruning):
        raise TypeError(
            f"pruning must be an instance of Pruning. Got {type(pruning).__name__}."
        )


def _normalize_inputs(
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
) -> _WCSInputs:
    """Normalize and validate WCS arrays."""
    inputs = _WCSInputs(
        p_values=as_1d_numeric("p_values", p_values),
        test_scores=as_1d_numeric("test_scores", test_scores),
        calib_scores=as_1d_numeric("calib_scores", calib_scores),
        test_weights=as_1d_numeric("test_weights", test_weights),
        calib_weights=as_1d_numeric("calib_weights", calib_weights),
    )
    validate_p_values(inputs.p_values)
    validate_finite("test_scores", inputs.test_scores)
    validate_finite("calib_scores", inputs.calib_scores)
    validate_non_negative_finite("test_weights", inputs.test_weights)
    validate_non_negative_finite("calib_weights", inputs.calib_weights)
    if (
        inputs.test_weights.size != inputs.test_scores.size
        or inputs.p_values.size != inputs.test_scores.size
    ):
        raise ValueError(
            "test_scores, test_weights, and p_values must have the same length."
        )
    if inputs.calib_scores.size != inputs.calib_weights.size:
        raise ValueError("calib_scores and calib_weights must have the same length.")
    return inputs


def _calibration_mass(
    inputs: _WCSInputs,
    kde_support: tuple[np.ndarray, np.ndarray, float] | None,
) -> tuple[float, np.ndarray]:
    """Return total and score-specific calibration weight masses."""
    if kde_support is not None:
        eval_grid, cdf_values, total_weight = kde_support
        mass = total_weight * (
            1.0
            - np.interp(
                inputs.test_scores,
                eval_grid,
                cdf_values,
                left=0.0,
                right=1.0,
            )
        )
    else:
        total_weight = float(np.sum(inputs.calib_weights, dtype=float))
        mass = _calib_weight_mass_strictly_above(
            inputs.calib_scores,
            inputs.calib_weights,
            inputs.test_scores,
        )
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "Weighted FDR requires positive finite total calibration weight."
        )
    return total_weight, mass


def _calib_weight_mass_strictly_above(
    calib_scores: np.ndarray,
    calib_weights: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute calibration weight mass strictly above each target."""
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = calib_weights[order]
    total_weight = np.sum(sorted_weights)
    cumulative_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="right")
    return total_weight - cumulative_weights[positions]


def _rejection_sizes(
    inputs: _WCSInputs,
    *,
    alpha: float,
    sum_calib_weight: float,
    calib_mass_strictly_above: np.ndarray,
    include_self_weight: bool,
) -> np.ndarray:
    """Compute every leave-one-out WCS rejection-set size."""
    n_test = inputs.test_scores.size
    rejection_sizes = np.zeros(n_test, dtype=float)
    bh_thresholds = alpha * (np.arange(1, n_test + 1) / n_test)
    scratch = np.empty(n_test, dtype=float)
    sorted_test_idx, lt_cutoffs = _score_rank_cache(
        inputs.test_scores,
        include_self_weight,
    )
    logger = get_logger("fdr")
    iterator = (
        tqdm(range(n_test), desc="Weighted FDR Control")
        if logger.isEnabledFor(logging.INFO)
        else range(n_test)
    )
    for index in iterator:
        rejection_sizes[index] = _rejection_set_size_for_instance(
            index,
            inputs,
            sum_calib_weight,
            bh_thresholds,
            calib_mass_strictly_above,
            scratch,
            include_self_weight=include_self_weight,
            sorted_test_idx=sorted_test_idx,
            lt_cutoffs=lt_cutoffs,
        )
    return rejection_sizes


def _score_rank_cache(
    test_scores: np.ndarray,
    include_self_weight: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build the strict test-score ordering cache when required."""
    if not include_self_weight:
        return None, None
    sorted_test_idx = np.argsort(test_scores, kind="mergesort")
    sorted_scores = test_scores[sorted_test_idx]
    lt_cutoffs = np.searchsorted(sorted_scores, test_scores, side="left")
    return sorted_test_idx, lt_cutoffs


def _rejection_set_size_for_instance(
    index: int,
    inputs: _WCSInputs,
    sum_calib_weight: float,
    bh_thresholds: np.ndarray,
    calib_mass_strictly_above: np.ndarray,
    scratch: np.ndarray,
    *,
    include_self_weight: bool,
    sorted_test_idx: np.ndarray | None,
    lt_cutoffs: np.ndarray | None,
) -> int:
    """Compute the WCS auxiliary rejection-set size for one test instance."""
    np.copyto(scratch, calib_mass_strictly_above)
    if include_self_weight:
        if sorted_test_idx is None or lt_cutoffs is None:
            raise ValueError("Internal error: missing score-rank cache for WCS.")
        scratch[sorted_test_idx[: lt_cutoffs[index]]] += inputs.test_weights[index]
        denominator = sum_calib_weight + inputs.test_weights[index]
    else:
        denominator = sum_calib_weight
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError(
            "Weighted FDR requires positive finite effective calibration mass."
        )
    scratch[index] = 0.0
    scratch /= denominator
    return _bh_rejection_count(scratch, bh_thresholds)


def _bh_rejection_count(p_values: np.ndarray, thresholds: np.ndarray) -> int:
    """Return the size of a BH rejection set."""
    sorted_p_values = np.sort(p_values)
    below = np.nonzero(sorted_p_values <= thresholds)[0]
    return 0 if below.size == 0 else int(below[-1] + 1)


def _prune(
    first_selection: np.ndarray,
    selected_sizes: np.ndarray,
    pruning: Pruning,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the configured WCS pruning rule."""
    if pruning is Pruning.HETEROGENEOUS:
        metrics = rng.uniform(size=first_selection.size) * selected_sizes
    elif pruning is Pruning.HOMOGENEOUS:
        metrics = rng.uniform() * selected_sizes
    else:
        metrics = selected_sizes.astype(float)
    return _select_with_metrics(first_selection, metrics)


def _select_with_metrics(
    first_selection: np.ndarray,
    metrics: np.ndarray,
) -> np.ndarray:
    """Select indices whose metric satisfies the r-star threshold."""
    r_star = _compute_r_star(metrics)
    if r_star == 0:
        return np.array([], dtype=int)
    return np.sort(first_selection[metrics <= r_star])


def _compute_r_star(metrics: np.ndarray) -> int:
    """Return the largest r such that #{j: metrics_j <= r} >= r."""
    if metrics.size == 0:
        return 0
    sorted_metrics = np.sort(metrics)
    for k in range(sorted_metrics.size, 0, -1):
        if sorted_metrics[k - 1] <= k:
            return k
    return 0


__all__ = ["extract_kde_support", "extract_required_fields", "run"]
