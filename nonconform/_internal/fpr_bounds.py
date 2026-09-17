"""Private mathematics and validation for raw-score FPR certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nonconform.structures import ConformalResult

from .certificates import conservative_mc_quantile, validate_scope
from .provenance import parse_result_provenance
from .validation import (
    as_1d_numeric,
    validate_finite,
    validate_optional_seed,
    validate_positive_integer,
    validate_probability,
)


def as_calibration_scores(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize a nonempty, finite one-dimensional score array."""
    scores = as_1d_numeric(name, values).astype(float, copy=True)
    if scores.size == 0:
        raise ValueError(f"{name} must contain at least one score.")
    validate_finite(name, scores)
    return scores


def as_scores(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize a possibly empty, finite one-dimensional score array."""
    scores = as_1d_numeric(name, values).astype(float, copy=True)
    validate_finite(name, scores)
    return scores


def as_threshold_query(
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Normalize scalar or vector score thresholds, allowing infinities."""
    try:
        values = np.asarray(threshold, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a numeric scalar or 1D array.") from exc
    scalar_input = values.ndim == 0
    values = np.atleast_1d(values)
    if values.ndim != 1:
        raise ValueError(
            f"threshold must be a scalar or 1D array, got shape {values.shape!r}."
        )
    if np.any(np.isnan(values)):
        raise ValueError("threshold must not contain NaN values.")
    return values.astype(float, copy=True), scalar_input


def validate_target_fpr(target_fpr: float) -> float:
    """Validate an operational target false-positive rate."""
    return validate_probability("target_fpr", target_fpr)


def validate_result_scope(result: ConformalResult) -> np.ndarray:
    """Validate a native result and return its calibration scores."""
    if result.calib_scores is None:
        raise ValueError(
            "result is missing calib_scores. Run score_samples() or "
            "compute_p_values() first."
        )
    if result.test_weights is not None or result.calib_weights is not None:
        raise ValueError("fpr_bounds() supports only unweighted calibration scores.")
    provenance = parse_result_provenance(result)
    validate_scope(provenance, procedure="fpr_bounds")
    return as_calibration_scores("result.calib_scores", result.calib_scores)


@dataclass(frozen=True, slots=True)
class KSBand:
    """Prepared one-sided Monte Carlo KS configuration."""

    n_calibration: int
    confidence: float
    n_resamples: int
    seed: int | None
    critical_value: float


def prepare_ks_band(
    *,
    n_calibration: int,
    confidence: float,
    n_resamples: int | None,
    seed: int | None,
) -> KSBand:
    """Prepare a conservative one-sided KS critical value by simulation."""
    n_calibration = validate_positive_integer("n_calibration", n_calibration)
    confidence = validate_probability("confidence", confidence)
    n_resamples = validate_positive_integer(
        "n_resamples", 1000 if n_resamples is None else n_resamples
    )
    seed = validate_optional_seed("seed", seed)

    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_calibration + 1, dtype=float) / n_calibration
    statistics = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        uniforms = np.sort(rng.random(n_calibration))
        statistics[index] = np.max(ranks - uniforms)

    critical_value = conservative_mc_quantile(statistics, confidence)
    return KSBand(
        n_calibration=n_calibration,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        critical_value=critical_value,
    )


def upper_bound(
    alarm_counts: np.ndarray,
    *,
    n_calibration: int,
    critical_value: float,
) -> np.ndarray:
    """Evaluate the simultaneous upper bound from inclusive tail counts."""
    empirical_fpr = alarm_counts.astype(float) / n_calibration
    return np.minimum(1.0, empirical_fpr + critical_value)


__all__ = [
    "KSBand",
    "as_calibration_scores",
    "as_scores",
    "as_threshold_query",
    "prepare_ks_band",
    "upper_bound",
    "validate_result_scope",
    "validate_target_fpr",
]
