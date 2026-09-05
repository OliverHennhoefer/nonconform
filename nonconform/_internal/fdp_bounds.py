"""Private mathematics and validation for simultaneous FDP bounds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nonconform.structures import ConformalResult

from .provenance import (
    EstimationFamily,
    StrategyFamily,
    parse_result_provenance,
)
from .validation import (
    as_1d_numeric,
    validate_optional_seed,
    validate_p_values,
    validate_positive_finite,
    validate_positive_integer,
    validate_probability,
)

DEFAULT_METHOD = "mc_thc"
_METHODS = frozenset({"mc_thc", "mc_hc", "mc_ks", "ks", "mc_bj"})


def validate_method(method: str) -> str:
    """Normalize and validate the FDP-bound envelope method."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    normalized = "_".join(method.strip().lower().replace("-", " ").split())
    if normalized not in _METHODS:
        supported = ", ".join(sorted(_METHODS))
        raise ValueError(f"method must be one of {{{supported}}}.")
    return normalized


def validate_truncation(
    lower: float,
    upper: float,
    beta: float,
) -> tuple[float, float, float]:
    """Validate truncated higher-criticism shape parameters."""
    lower_value = validate_probability("lower", lower)
    upper_value = validate_probability("upper", upper)
    if lower_value >= upper_value:
        raise ValueError("lower must be strictly smaller than upper.")
    beta_value = validate_positive_finite("beta", beta)
    return lower_value, upper_value, beta_value


def as_p_values(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize p-values into a non-empty, validated 1D float array."""
    p_values = as_1d_numeric(name, values).astype(float, copy=True)
    if p_values.size == 0:
        raise ValueError(f"{name} must contain at least one p-value.")
    validate_p_values(p_values)
    return np.clip(p_values, 0.0, 1.0)


def as_thresholds(
    thresholds: np.ndarray | None,
    p_values: np.ndarray,
) -> np.ndarray:
    """Return evaluated thresholds, preserving explicit user order."""
    if thresholds is None:
        return np.unique(np.sort(p_values)).astype(float, copy=True)

    arr = as_1d_numeric("thresholds", thresholds).astype(float, copy=True)
    validate_p_values(arr)
    return np.clip(arr, 0.0, 1.0)


def as_threshold_query(threshold: float | np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize scalar or vector threshold query values."""
    try:
        arr = np.asarray(threshold, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a numeric scalar or 1D array.") from exc
    scalar_input = arr.ndim == 0
    arr = np.atleast_1d(arr)
    if arr.ndim != 1:
        raise ValueError(
            f"threshold must be a scalar or 1D array, got shape {arr.shape!r}."
        )
    validate_p_values(arr)
    return np.clip(arr, 0.0, 1.0), scalar_input


def validate_result_scope(result: ConformalResult) -> None:
    """Reject result scopes known to fall outside the FDP-bound guarantee."""
    provenance = parse_result_provenance(result, allow_legacy_metadata=True)
    if provenance is None:
        return
    if provenance.weighted:
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports only unweighted "
            "conformal p-values in this release."
        )
    if provenance.estimation_family is not EstimationFamily.EMPIRICAL:
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports empirical conformal "
            "p-values only."
        )
    if provenance.strategy_family is not StrategyFamily.SPLIT:
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports split or detached "
            "calibration results only."
        )


def build_ecdf_upper_bound(
    *,
    method: str,
    n_calibration: int,
    n_test: int,
    confidence: float,
    n_resamples: int,
    seed: int | None,
    lower: float,
    upper: float,
    beta: float,
    precision: float,
) -> tuple[float, np.ndarray | None]:
    """Build method-specific ECDF envelope state."""
    bj_lower_bounds = None
    if method == "mc_thc":
        summary_quantile = _mc_summary_quantile(
            n_calibration=n_calibration,
            n_test=n_test,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            statistic=lambda sampled: _higher_criticism_statistic(
                sampled,
                lower=lower,
                upper=upper,
                beta=beta,
            ),
        )
    elif method == "mc_hc":
        summary_quantile = _mc_summary_quantile(
            n_calibration=n_calibration,
            n_test=n_test,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            statistic=_higher_criticism_statistic,
        )
    elif method == "mc_ks":
        summary_quantile = _mc_summary_quantile(
            n_calibration=n_calibration,
            n_test=n_test,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            statistic=_ks_statistic,
        )
    elif method == "ks":
        summary_quantile = _dkw_lambda(
            n_calibration=n_calibration,
            n_test=n_test,
            confidence=confidence,
        )
    elif method == "mc_bj":
        summary_quantile = _mc_summary_quantile(
            n_calibration=n_calibration,
            n_test=n_test,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            statistic=_berk_jones_statistic,
        )
        targets = np.arange(1, n_test // 2 + 1, dtype=float) / n_test
        bj_lower_bounds = _solve_bernoulli_kl_lower_bounds(
            targets,
            summary_quantile,
            n_test=n_test,
            precision=precision,
        )
    else:
        raise RuntimeError(f"Internal error: unsupported FDP method {method!r}.")
    return summary_quantile, bj_lower_bounds


def ecdf_upper_bound_from_params(
    x: np.ndarray,
    *,
    method: str,
    summary_quantile: float,
    lower: float,
    upper: float,
    beta: float,
    bj_lower_bounds: np.ndarray | None,
    n_test: int,
) -> np.ndarray:
    """Evaluate the configured ECDF upper envelope."""
    if method == "mc_thc":
        return _thc_ecdf_upper_bound(
            x,
            summary_quantile=summary_quantile,
            lower=lower,
            upper=upper,
            beta=beta,
        )
    if method == "mc_hc":
        return _hc_ecdf_upper_bound(x, summary_quantile=summary_quantile)
    if method in {"mc_ks", "ks"}:
        return _ks_ecdf_upper_bound(x, summary_quantile=summary_quantile)
    if method == "mc_bj":
        if bj_lower_bounds is None:
            raise RuntimeError("Internal error: missing Berk-Jones lower bounds.")
        return _bj_ecdf_upper_bound(
            x,
            lower_bounds=bj_lower_bounds,
            n_test=n_test,
        )
    raise RuntimeError(f"Internal error: unsupported FDP method {method!r}.")


def evaluate_fdp_upper_bound(
    p_values: np.ndarray,
    thresholds: np.ndarray,
    *,
    ecdf_upper_bound: Callable[[np.ndarray], np.ndarray],
    boost: bool,
) -> np.ndarray:
    """Evaluate simultaneous FDP upper bounds at thresholds."""
    sorted_p_values = np.sort(p_values)
    n_test = sorted_p_values.size

    if boost:
        max_p_under_threshold = np.zeros(thresholds.size, dtype=float)
        numerator = np.full(thresholds.size, fill_value=n_test, dtype=float)

        for p_value in sorted_p_values:
            mask = p_value <= thresholds
            max_p_under_threshold[mask] = np.maximum(
                max_p_under_threshold[mask],
                p_value,
            )
            ecdf_bound = float(ecdf_upper_bound(np.array([p_value], dtype=float))[0])
            second_term = n_test * ecdf_bound - np.count_nonzero(
                sorted_p_values <= p_value
            )
            numerator[mask] = np.minimum(numerator[mask], second_term)

        numerator += np.searchsorted(
            sorted_p_values,
            max_p_under_threshold,
            side="right",
        )
    else:
        numerator = n_test * ecdf_upper_bound(thresholds)

    denominator = np.searchsorted(sorted_p_values, thresholds, side="right")
    raw = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )
    return np.clip(raw, 0.0, 1.0)


def _custom_quantile(values: np.ndarray, q: float) -> float:
    """Return the Monte Carlo quantile used by Song, Jin, and Candes."""
    n = len(values)
    sorted_values = np.sort(values)
    index = q * (n + 1)
    if index <= n:
        return float(sorted_values[int(np.ceil(index)) - 1])
    return float("inf")


def _sample_conformal_null_p_values(
    *,
    n_calibration: int,
    n_test: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample null conformal p-values from their finite-sample rank law."""
    uniforms = np.sort(rng.random(n_calibration))
    interval_probabilities = np.diff(np.concatenate(([0.0], uniforms, [1.0])))
    cell_indices = rng.choice(
        np.arange(n_calibration + 1),
        size=n_test,
        p=interval_probabilities,
    )
    return (cell_indices + rng.random(n_test)) / (n_calibration + 1)


def _higher_criticism_statistic(
    p_values: np.ndarray,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    beta: float = 0.5,
) -> float:
    """Compute the higher-criticism summary statistic."""
    n_test = len(p_values)
    sorted_p_values = np.sort(p_values)
    eps = np.finfo(float).eps
    safe_p_values = np.clip(sorted_p_values, eps, 1.0 - eps)
    grid = np.arange(1, n_test + 1, dtype=float) / n_test
    scaled_diffs = (grid - sorted_p_values) / np.power(
        safe_p_values * (1.0 - safe_p_values),
        beta,
    )

    lower_idx = int(np.count_nonzero(p_values <= lower))
    upper_idx = int(np.count_nonzero(p_values <= upper))
    lower_term = 0.0
    if lower > 0.0:
        lower_term = (lower_idx / n_test - lower) / np.power(
            lower * (1.0 - lower),
            beta,
        )
    if lower_idx < upper_idx:
        return float(max(np.max(scaled_diffs[lower_idx:upper_idx]), lower_term))
    return float(lower_term)


def _ks_statistic(p_values: np.ndarray) -> float:
    """Compute the one-sided KS summary statistic."""
    n_test = len(p_values)
    sorted_p_values = np.sort(p_values)
    grid = np.arange(1, n_test + 1, dtype=float) / n_test
    return float(np.max(grid - sorted_p_values))


def _bernoulli_kl(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Compute Bernoulli KL divergence with endpoint-safe probabilities."""
    eps = np.finfo(float).eps
    p0_safe = np.clip(p0, eps, 1.0 - eps)
    p1_safe = np.clip(p1, eps, 1.0 - eps)
    return p0_safe * np.log(p0_safe / p1_safe) + (1.0 - p0_safe) * np.log(
        (1.0 - p0_safe) / (1.0 - p1_safe)
    )


def _berk_jones_statistic(p_values: np.ndarray) -> float:
    """Compute the Berk-Jones summary statistic."""
    n_test = len(p_values)
    half = n_test // 2
    if half == 0:
        return 0.0
    sorted_p_values = np.sort(p_values)
    grid = np.arange(1, half + 1, dtype=float) / n_test
    return float(n_test * np.max(_bernoulli_kl(sorted_p_values[:half], grid)))


def _solve_bernoulli_kl_lower_bounds(
    targets: np.ndarray,
    statistic: float,
    *,
    n_test: int,
    precision: float,
) -> np.ndarray:
    """Solve KL(x, target) = statistic / n_test below each target."""
    target_level = statistic / n_test
    solutions = np.zeros_like(targets, dtype=float)
    for i, target in enumerate(targets):
        lower = 0.0
        upper = float(target)
        while upper - lower > precision:
            midpoint = (lower + upper) / 2.0
            divergence = float(
                _bernoulli_kl(
                    np.array([midpoint], dtype=float),
                    np.array([target], dtype=float),
                )[0]
            )
            if divergence < target_level:
                upper = midpoint
            else:
                lower = midpoint
        solutions[i] = (lower + upper) / 2.0
    return solutions


def _dkw_tau(n_calibration: int, n_test: int) -> float:
    """Return the transductive DKW effective sample size."""
    return n_test * n_calibration / (n_test + n_calibration)


def _dkw_psi(
    x: float,
    *,
    n_calibration: int,
    n_test: int,
    delta: float,
) -> float:
    """Return one DKW fixed-point update."""
    tau = _dkw_tau(n_calibration, n_test)
    numerator = np.log(1.0 / delta) + np.log(
        1.0 + np.sqrt(2.0 * np.pi) * 2.0 * x * tau / np.sqrt(n_calibration + n_test)
    )
    return float(min(1.0, np.sqrt(numerator / (2.0 * tau))))


def _dkw_lambda(
    *,
    n_calibration: int,
    n_test: int,
    confidence: float,
    iterations: int = 1000,
) -> float:
    """Compute the deterministic KS envelope offset from the author code."""
    delta = 1.0 - confidence
    value = 1.0
    for _ in range(iterations):
        value = _dkw_psi(
            value,
            n_calibration=n_calibration,
            n_test=n_test,
            delta=delta,
        )
    return value


def _mc_summary_quantile(
    *,
    n_calibration: int,
    n_test: int,
    confidence: float,
    n_resamples: int,
    seed: int | None,
    statistic: Callable[[np.ndarray], float],
) -> float:
    """Estimate an envelope cutoff from conformal null samples."""
    rng = np.random.default_rng(seed)
    summary_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = _sample_conformal_null_p_values(
            n_calibration=n_calibration,
            n_test=n_test,
            rng=rng,
        )
        summary_stats[i] = statistic(sampled)
    return _custom_quantile(summary_stats, confidence)


def _hc_ecdf_upper_bound(
    x: np.ndarray,
    *,
    summary_quantile: float,
) -> np.ndarray:
    """Evaluate the HC upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    return np.clip(
        x_arr + np.sqrt(np.clip(x_arr * (1.0 - x_arr), 0.0, None)) * summary_quantile,
        0.0,
        1.0,
    )


def _thc_ecdf_upper_bound(
    x: np.ndarray,
    *,
    summary_quantile: float,
    lower: float,
    upper: float,
    beta: float,
) -> np.ndarray:
    """Evaluate the MC-THC upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)

    lower_value = min(
        1.0,
        lower + np.power(lower * (1.0 - lower), beta) * summary_quantile,
    )
    lower_mask = x_arr < lower
    upper_mask = x_arr > upper
    middle_mask = ~(lower_mask | upper_mask)

    out[lower_mask] = lower_value
    out[upper_mask] = 1.0
    out[middle_mask] = np.minimum(
        1.0,
        x_arr[middle_mask]
        + np.power(x_arr[middle_mask] * (1.0 - x_arr[middle_mask]), beta)
        * summary_quantile,
    )
    return np.clip(out, 0.0, 1.0)


def _ks_ecdf_upper_bound(
    x: np.ndarray,
    *,
    summary_quantile: float,
) -> np.ndarray:
    """Evaluate a KS-style upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    return np.clip(x_arr + summary_quantile, 0.0, 1.0)


def _bj_ecdf_upper_bound(
    x: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    n_test: int,
) -> np.ndarray:
    """Evaluate the Berk-Jones upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    if lower_bounds.size == 0:
        return np.ones_like(x_arr, dtype=float)
    indices = np.searchsorted(lower_bounds, x_arr, side="left")
    return np.where(indices == lower_bounds.size, 1.0, indices / n_test)


__all__ = [
    "DEFAULT_METHOD",
    "as_1d_numeric",
    "as_p_values",
    "as_threshold_query",
    "as_thresholds",
    "build_ecdf_upper_bound",
    "ecdf_upper_bound_from_params",
    "evaluate_fdp_upper_bound",
    "validate_method",
    "validate_optional_seed",
    "validate_positive_finite",
    "validate_positive_integer",
    "validate_probability",
    "validate_result_scope",
    "validate_truncation",
]
