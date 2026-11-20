"""Private BH and Storey-BH primitives for unweighted conformal p-values."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from scipy.stats import false_discovery_control

from .constants import TieBreakMode
from .provenance import EstimationFamily, ResultProvenance, parse_result_provenance
from .validation import as_1d_numeric, validate_p_values

if TYPE_CHECKING:
    from nonconform.structures import ConformalResult

SelectionMethod = Literal["bh", "storey"]

_METHODS = ("bh", "storey")
_CLASSICAL_FAMILIES = frozenset(
    {EstimationFamily.EMPIRICAL, EstimationFamily.CONDITIONAL_EMPIRICAL}
)
_logger = logging.getLogger("nonconform.fdr")


def normalize_method(method: str) -> SelectionMethod:
    """Normalize and validate a p-value selection method."""
    if not isinstance(method, str):
        raise TypeError("method must be 'bh' or 'storey'.")
    normalized = method.strip().lower()
    if normalized not in _METHODS:
        raise ValueError(f"method must be 'bh' or 'storey', got {method!r}.")
    return cast(SelectionMethod, normalized)


def validate_storey_scope(provenance: ResultProvenance) -> None:
    """Require the classical rank-based p-values covered by Storey-BH theory."""
    if provenance.grouped:
        raise ValueError(
            "method='storey' does not support group-conditional calibration. "
            "Use method='bh' with groups=."
        )
    if (
        provenance.estimation_family not in _CLASSICAL_FAMILIES
        or provenance.tie_break is not TieBreakMode.CLASSICAL
    ):
        raise ValueError(
            "method='storey' requires classical conformal p-values from "
            "Empirical() or ConditionalEmpirical() with tie_break='classical'. "
            "Its guarantee does not cover randomized or KDE-based values."
        )


def _storey_grid_index(n_calibration: int) -> int:
    """Return ``k = ceil((n + 1) / 2)``, so that ``lambda = k / (n + 1)``."""
    return (n_calibration + 2) // 2


def storey_lambda(n_calibration: int) -> float:
    """Return the classical p-value grid point at or just above one half.

    Classical conformal p-values lie on the grid ``k / (n + 1)``. Restricting
    Storey's tuning parameter to that grid is part of the conformal FDR results.
    """
    return _storey_grid_index(n_calibration) / (n_calibration + 1)


def storey_null_proportion(p_values: np.ndarray, *, n_calibration: int) -> float:
    """Estimate the null proportion with Storey's finite-sample estimator.

    Uses ``(1 + #{p >= lambda}) / (m (1 - lambda))`` without capping at one;
    the leading one and the uncapped value are needed for FDR control. An
    empty family returns one, so Storey-BH, like BH, selects nothing.
    """
    if p_values.size == 0:
        return 1.0
    k = _storey_grid_index(n_calibration)
    n_large = int(np.count_nonzero(p_values >= storey_lambda(n_calibration)))
    # 1 - lambda = (n + 1 - k) / (n + 1); kept in integers for exactness.
    numerator = (1 + n_large) * (n_calibration + 1)
    return numerator / (p_values.size * (n_calibration + 1 - k))


def null_proportion(
    p_values: np.ndarray,
    *,
    method: SelectionMethod,
    n_calibration: int,
) -> float:
    """Return the null proportion used to scale BH thresholds."""
    if method == "bh":
        return 1.0
    return storey_null_proportion(p_values, n_calibration=n_calibration)


def adjusted_p_values(
    p_values: np.ndarray,
    *,
    method: SelectionMethod,
    n_calibration: int,
) -> np.ndarray:
    """Return adjusted p-values whose ``<= alpha`` mask is the method's selection."""
    adjusted = false_discovery_control(p_values, method="bh")
    if method == "bh":
        return adjusted
    proportion = storey_null_proportion(p_values, n_calibration=n_calibration)
    return np.minimum(1.0, proportion * adjusted)


def has_classical_floor(provenance: ResultProvenance | None) -> bool:
    """Whether p-values are classical empirical values bounded below by 1/(n+1)."""
    return (
        provenance is not None
        and not provenance.weighted
        and provenance.estimation_family is EstimationFamily.EMPIRICAL
        and provenance.tie_break is TieBreakMode.CLASSICAL
    )


def warn_if_floor_blocks_discovery(
    p_values: np.ndarray,
    *,
    alpha: float,
    method: SelectionMethod,
    n_calibration: int,
    grouped: bool = False,
) -> bool:
    """Warn when the classical p-value floor rules out a single discovery.

    Call only after an empty selection of classical empirical p-values. With
    ``n`` calibration scores those p-values are at least ``1 / (n + 1)``, so
    the step-up rule cannot select one isolated point when that floor exceeds
    its first threshold ``alpha / (m * pi0)``. For group-conditional p-values,
    ``n`` is the largest calibration group among the test points.

    Returns:
        Whether the warning was logged.
    """
    n_test = p_values.size
    proportion = null_proportion(p_values, method=method, n_calibration=n_calibration)
    # The first step-up threshold is alpha / (m * pi0); compare it with the
    # floor 1 / (n + 1) without dividing by a possibly tiny threshold.
    scaled_floor = n_test * proportion / (n_calibration + 1)
    if scaled_floor <= alpha * (1.0 + 1e-12):
        return False
    if method == "bh":
        rule = (
            f"a single BH discovery among m = {n_test} tests needs "
            f"p <= alpha/m = {alpha / n_test:.3g}"
        )
    else:
        rule = (
            f"a single Storey-BH discovery among m = {n_test} tests needs "
            f"p <= alpha/(m*pi0) = {alpha / (n_test * proportion):.3g} at the "
            f"estimated null proportion pi0 = {proportion:.3g}"
        )
    k_min = math.ceil(scaled_floor / alpha * (1.0 - 1e-12))
    if k_min > n_test:
        consequence = (
            "No selection is possible for this batch, even if every test point "
            "attains that minimum."
        )
    else:
        consequence = (
            f"Any discovery requires at least {k_min} test points that jointly "
            "attain that minimum."
        )
    required = math.ceil(n_test * proportion / alpha * (1.0 - 1e-12)) - 1
    if grouped:
        calibration = f"at most {n_calibration} calibration scores per test group"
        target = "calibration scores per group"
    else:
        calibration = f"{n_calibration} calibration scores"
        target = "calibration scores"
    _logger.warning(
        "select() made no discoveries. With %s, classical conformal p-values "
        "cannot fall below 1/%d = %.3g, but %s. %s Use at least %d %s, or "
        "smaller test batches, so that a single anomaly can be discovered. "
        "If anomalies should stand out clearly, also check that "
        "score_polarity matches the detector's score direction.",
        calibration,
        n_calibration + 1,
        1.0 / (n_calibration + 1),
        rule,
        consequence,
        required,
        target,
    )
    return True


def warn_no_discoveries(
    p_values: np.ndarray,
    *,
    alpha: float,
    method: SelectionMethod,
    n_calibration: int,
    suggest_fpr_bounds: bool,
    calibration_ruled_out: bool,
) -> None:
    """Explain an empty standard selection the p-value floor does not explain.

    Call only after an empty standard selection for which
    ``warn_if_floor_blocks_discovery`` did not warn. With
    ``calibration_ruled_out``, the p-values are classical and their floor
    permitted a single discovery, so the evidence in the batch was the limit.
    Otherwise (randomized or calibration-conditional p-values) the calibration
    size remains one of the possible causes.
    """
    n_test = p_values.size
    proportion = null_proportion(p_values, method=method, n_calibration=n_calibration)
    threshold = alpha / (n_test * proportion)
    rule = "alpha/m" if method == "bh" else "alpha/(m*pi0)"
    if calibration_ruled_out:
        causes = (
            "Calibration size is not the limit here: either the batch holds no "
            "anomalies, or they score too close to normal points for the "
            "detector to tell them apart."
        )
    else:
        causes = (
            "Either the batch holds no anomalies, they score too close to "
            "normal points for the detector to tell them apart, or there are "
            "too few calibration scores for this batch (aim for n_calib >= "
            "batch_size / alpha)."
        )
    fallback = (
        " If anomalies are expected but separate poorly, "
        "detector.fpr_bounds() gives a score threshold that caps the "
        "false-alarm rate instead of the share of false flags in a list."
        if suggest_fpr_bounds
        else ""
    )
    _logger.warning(
        "select() made no discoveries. The smallest p-value in this batch is "
        "%.3g, above %s = %.3g that a single discovery needs (m = %d tests), "
        "and no larger group of points met its step-up threshold either. "
        "%s Check that score_polarity matches the detector's score "
        "direction.%s",
        float(np.min(p_values)),
        rule,
        threshold,
        n_test,
        causes,
        fallback,
    )


def validate_result_scope(
    result: ConformalResult,
    *,
    method: SelectionMethod,
) -> tuple[np.ndarray, int]:
    """Validate a native snapshot; return its p-values and calibration size."""
    if result.p_values is None:
        raise ValueError("result is missing p_values. Run compute_p_values() first.")
    if result.calib_scores is None:
        raise ValueError(
            "result is missing calib_scores. Run compute_p_values() first."
        )
    if result.test_weights is not None or result.calib_weights is not None:
        raise ValueError(
            "adjusted_p_values() supports only unweighted p-values. Weighted "
            "select() uses weighted conformalized selection instead."
        )
    provenance = parse_result_provenance(result)
    if provenance is None:
        raise ValueError(
            "adjusted_p_values() requires a native detector snapshot. For "
            "external p-values, use scipy.stats.false_discovery_control()."
        )
    if provenance.weighted:
        raise ValueError(
            "adjusted_p_values() supports only unweighted p-values. Weighted "
            "select() uses weighted conformalized selection instead."
        )
    if method == "storey":
        validate_storey_scope(provenance)
    p_values = as_1d_numeric("result.p_values", result.p_values).astype(
        float, copy=True
    )
    validate_p_values(p_values)
    calib_scores = as_1d_numeric("result.calib_scores", result.calib_scores)
    if calib_scores.size == 0:
        raise ValueError("result.calib_scores must contain at least one score.")
    signature = provenance.test_batch_signature
    if signature is None or not signature.shape or signature.shape[0] != p_values.size:
        raise ValueError(
            "result.p_values are inconsistent with recorded batch dimensions."
        )
    return np.clip(p_values, 0.0, 1.0), int(calib_scores.size)
