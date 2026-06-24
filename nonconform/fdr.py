"""False Discovery Rate control utilities for conformal prediction.

This module provides explicit entry points for:

- Post-hoc simultaneous FDP upper bounds for conformal p-values.
- Weighted Conformalized Selection (WCS) under covariate shift.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.structures import ConformalResult

from ._internal import Pruning, get_logger

_KDE_MONOTONICITY_TOL = 1e-12
_FDP_BOUND_METHOD = "mc_thc"
_FDP_BOUND_METHODS = frozenset({"mc_thc", "mc_hc", "mc_ks", "ks", "mc_bj"})
_RESULT_SCOPE_METADATA_KEY = "nonconform"


def _validate_alpha(alpha: float) -> None:
    """Validate FDR target level."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _require_result_bundle(
    result: ConformalResult | None, *, caller: str
) -> ConformalResult:
    """Ensure a non-empty result bundle is provided."""
    if result is None:
        raise ValueError(
            f"result must be a ConformalResult, got None. "
            f"Run compute_p_values(...) before calling {caller}()."
        )
    return result


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


def _validate_positive_integer(name: str, value: int) -> int:
    """Validate a positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_seed(seed: int | None) -> int | None:
    """Validate an optional random seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a non-negative integer or None.")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer or None.")
    return seed


def _validate_probability(name: str, value: float) -> float:
    """Validate a scalar probability in (0, 1)."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value in (0, 1).") from exc
    if not np.isfinite(scalar) or not (0.0 < scalar < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")
    return scalar


def _validate_positive_finite(name: str, value: float) -> float:
    """Validate a positive finite scalar."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value.") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return scalar


def _validate_fdp_method(method: str) -> str:
    """Normalize and validate the FDP-bound envelope method."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    normalized = "_".join(method.strip().lower().replace("-", " ").split())
    if normalized not in _FDP_BOUND_METHODS:
        supported = ", ".join(sorted(_FDP_BOUND_METHODS))
        raise ValueError(f"method must be one of {{{supported}}}.")
    return normalized


def _validate_truncation(
    lower: float, upper: float, beta: float
) -> tuple[float, float, float]:
    """Validate truncated higher-criticism shape parameters."""
    lower_value = _validate_probability("lower", lower)
    upper_value = _validate_probability("upper", upper)
    if lower_value >= upper_value:
        raise ValueError("lower must be strictly smaller than upper.")
    beta_value = _validate_positive_finite("beta", beta)
    return lower_value, upper_value, beta_value


def _as_p_values(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize p-values into a non-empty, validated 1D float array."""
    p_values = _as_1d_numeric(name, values).astype(float, copy=True)
    if p_values.size == 0:
        raise ValueError(f"{name} must contain at least one p-value.")
    _validate_p_values(p_values)
    return np.clip(p_values, 0.0, 1.0)


def _as_thresholds(thresholds: np.ndarray | None, p_values: np.ndarray) -> np.ndarray:
    """Return evaluated thresholds, preserving explicit user order."""
    if thresholds is None:
        return np.unique(np.sort(p_values)).astype(float, copy=True)

    arr = _as_1d_numeric("thresholds", thresholds).astype(float, copy=True)
    _validate_p_values(arr)
    return np.clip(arr, 0.0, 1.0)


def _as_threshold_query(threshold: float | np.ndarray) -> tuple[np.ndarray, bool]:
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
    _validate_p_values(arr)
    return np.clip(arr, 0.0, 1.0), scalar_input


def _validate_fdp_result_scope(result: ConformalResult) -> None:
    """Reject cached result scopes known to fall outside the FDP-bound guarantee."""
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if "kde" in metadata:
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports empirical conformal "
            "p-values, not probabilistic/KDE p-values."
        )

    scope = metadata.get(_RESULT_SCOPE_METADATA_KEY)
    if scope is None:
        return
    if not isinstance(scope, dict):
        raise ValueError("result.metadata['nonconform'] must be a dictionary.")

    strategy = scope.get("strategy")
    estimation = scope.get("estimation")
    if scope.get("weighted"):
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports only unweighted "
            "conformal p-values in this release."
        )
    if strategy != "Split":
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports split or detached "
            "calibration results only."
        )
    if estimation != "Empirical":
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports empirical conformal "
            "p-values only."
        )


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


def _dkw_psi(x: float, *, n_calibration: int, n_test: int, delta: float) -> float:
    """Return one DKW fixed-point update."""
    tau = _dkw_tau(n_calibration, n_test)
    numerator = np.log(1.0 / delta) + np.log(
        1.0 + np.sqrt(2.0 * np.pi) * 2.0 * x * tau / np.sqrt(n_calibration + n_test)
    )
    return float(min(1.0, np.sqrt(numerator / (2.0 * tau))))


def _dkw_lambda(
    *, n_calibration: int, n_test: int, confidence: float, iterations: int = 1000
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


def _hc_ecdf_upper_bound(x: np.ndarray, *, summary_quantile: float) -> np.ndarray:
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


def _ks_ecdf_upper_bound(x: np.ndarray, *, summary_quantile: float) -> np.ndarray:
    """Evaluate a KS-style upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    return np.clip(x_arr + summary_quantile, 0.0, 1.0)


def _bj_ecdf_upper_bound(
    x: np.ndarray, *, lower_bounds: np.ndarray, n_test: int
) -> np.ndarray:
    """Evaluate the Berk-Jones upper envelope for the null p-value ECDF."""
    x_arr = np.asarray(x, dtype=float)
    if lower_bounds.size == 0:
        return np.ones_like(x_arr, dtype=float)
    indices = np.searchsorted(lower_bounds, x_arr, side="left")
    return np.where(indices == lower_bounds.size, 1.0, indices / n_test)


def _ecdf_upper_bound_from_params(
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
        return _bj_ecdf_upper_bound(x, lower_bounds=bj_lower_bounds, n_test=n_test)
    raise RuntimeError(f"Internal error: unsupported FDP method {method!r}.")


def _build_ecdf_upper_bound(
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
                sampled, lower=lower, upper=upper, beta=beta
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


def _evaluate_fdp_upper_bound(
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


@dataclass(slots=True)
class FDPBoundResult:
    """Post-hoc simultaneous FDP upper-bound certificate.

    The result evaluates a high-confidence upper bound on the realized false
    discovery proportion (FDP) for threshold selections of conformal p-values.
    Use this as an exploratory certificate, not as a replacement for
    ``ConformalDetector.select(...)``.
    """

    p_values: np.ndarray
    thresholds: np.ndarray
    rejection_counts: np.ndarray
    fdp_upper_bounds: np.ndarray
    n_calibration: int
    n_test: int
    confidence: float
    method: str
    n_resamples: int
    boost: bool
    seed: int | None
    _summary_quantile: float = field(repr=False)
    _lower: float = field(repr=False)
    _upper: float = field(repr=False)
    _beta: float = field(repr=False)
    _precision: float = field(repr=False)
    _bj_lower_bounds: np.ndarray | None = field(default=None, repr=False)
    precision_lower_bounds: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Copy array fields so the result is independent of caller inputs."""
        self.p_values = np.asarray(self.p_values, dtype=float).copy()
        self.thresholds = np.asarray(self.thresholds, dtype=float).copy()
        self.rejection_counts = np.asarray(self.rejection_counts, dtype=int).copy()
        self.fdp_upper_bounds = np.asarray(self.fdp_upper_bounds, dtype=float).copy()
        if self._bj_lower_bounds is not None:
            self._bj_lower_bounds = np.asarray(
                self._bj_lower_bounds, dtype=float
            ).copy()
        self.precision_lower_bounds = 1.0 - self.fdp_upper_bounds

    def _ecdf_upper_bound(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this result's ECDF envelope."""
        return _ecdf_upper_bound_from_params(
            x,
            method=self.method,
            summary_quantile=self._summary_quantile,
            lower=self._lower,
            upper=self._upper,
            beta=self._beta,
            bj_lower_bounds=self._bj_lower_bounds,
            n_test=self.n_test,
        )

    def bound_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the FDP upper bound at one or more thresholds."""
        threshold_arr, scalar_input = _as_threshold_query(threshold)
        bounds = _evaluate_fdp_upper_bound(
            self.p_values,
            threshold_arr,
            ecdf_upper_bound=self._ecdf_upper_bound,
            boost=self.boost,
        )
        if scalar_input:
            return float(bounds[0])
        return bounds

    def precision_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the certified precision lower bound at thresholds."""
        bounds = self.bound_at(threshold)
        return 1.0 - bounds

    def to_frame(self, thresholds: np.ndarray | None = None) -> pd.DataFrame:
        """Return threshold-level FDP certificates as a DataFrame."""
        if thresholds is None:
            threshold_arr = self.thresholds
            fdp_bounds = self.fdp_upper_bounds
            precision_bounds = self.precision_lower_bounds
        else:
            threshold_arr = _as_thresholds(thresholds, self.p_values)
            fdp_bounds = self.bound_at(threshold_arr)
            precision_bounds = 1.0 - fdp_bounds
        rejection_counts = np.searchsorted(
            np.sort(self.p_values),
            threshold_arr,
            side="right",
        )
        return pd.DataFrame(
            {
                "threshold": threshold_arr,
                "discoveries": rejection_counts,
                "fdp_upper_bound": fdp_bounds,
                "precision_lower_bound": precision_bounds,
            }
        )

    def select(self, threshold: float) -> np.ndarray:
        """Return the selection mask induced by ``p_values <= threshold``."""
        threshold_arr, scalar_input = _as_threshold_query(threshold)
        if not scalar_input:
            raise ValueError("threshold must be a scalar for select().")
        return self.p_values <= threshold_arr[0]


def conformal_fdp_upper_bound(
    p_values: np.ndarray,
    *,
    n_calibration: int,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    method: str = _FDP_BOUND_METHOD,
    seed: int | None = None,
    boost: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    beta: float = 0.5,
    precision: float = 1e-8,
    thresholds: np.ndarray | None = None,
) -> FDPBoundResult:
    """Compute post-hoc simultaneous FDP upper bounds for conformal p-values.

    This implements simultaneous FDP envelopes from Song, Jin, and Candes for
    unweighted conformal p-values from a fixed scoring map. Choose ``method``
    before inspecting the resulting curve. The returned certificate is valid
    for threshold exploration under that scope; it does not cover detector/model
    selection or weighted conformal p-values.
    """
    p_values_arr = _as_p_values("p_values", p_values)
    n_calibration = _validate_positive_integer("n_calibration", n_calibration)
    n_resamples = _validate_positive_integer("n_resamples", n_resamples)
    confidence = _validate_probability("confidence", confidence)
    method = _validate_fdp_method(method)
    seed = _validate_seed(seed)
    if not isinstance(boost, bool):
        raise TypeError("boost must be a boolean value.")
    lower, upper, beta = _validate_truncation(lower, upper, beta)
    precision = _validate_positive_finite("precision", precision)

    evaluated_thresholds = _as_thresholds(thresholds, p_values_arr)
    summary_quantile, bj_lower_bounds = _build_ecdf_upper_bound(
        method=method,
        n_calibration=n_calibration,
        n_test=p_values_arr.size,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        lower=lower,
        upper=upper,
        beta=beta,
        precision=precision,
    )

    def ecdf_upper_bound(x: np.ndarray) -> np.ndarray:
        return _ecdf_upper_bound_from_params(
            x,
            method=method,
            summary_quantile=summary_quantile,
            lower=lower,
            upper=upper,
            beta=beta,
            bj_lower_bounds=bj_lower_bounds,
            n_test=p_values_arr.size,
        )

    fdp_bounds = _evaluate_fdp_upper_bound(
        p_values_arr,
        evaluated_thresholds,
        ecdf_upper_bound=ecdf_upper_bound,
        boost=boost,
    )
    sorted_p_values = np.sort(p_values_arr)
    rejection_counts = np.searchsorted(
        sorted_p_values,
        evaluated_thresholds,
        side="right",
    )

    return FDPBoundResult(
        p_values=p_values_arr,
        thresholds=evaluated_thresholds,
        rejection_counts=rejection_counts,
        fdp_upper_bounds=fdp_bounds,
        n_calibration=n_calibration,
        n_test=p_values_arr.size,
        confidence=confidence,
        method=method,
        n_resamples=n_resamples,
        boost=boost,
        seed=seed,
        _summary_quantile=summary_quantile,
        _lower=lower,
        _upper=upper,
        _beta=beta,
        _precision=precision,
        _bj_lower_bounds=bj_lower_bounds,
    )


def conformal_fdp_upper_bound_from_result(
    result: ConformalResult | None,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    method: str = _FDP_BOUND_METHOD,
    seed: int | None = None,
    boost: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    beta: float = 0.5,
    precision: float = 1e-8,
    thresholds: np.ndarray | None = None,
) -> FDPBoundResult:
    """Compute simultaneous FDP bounds from a ``ConformalResult`` bundle."""
    result = _require_result_bundle(
        result,
        caller="conformal_fdp_upper_bound_from_result",
    )
    if result.p_values is None:
        raise ValueError(
            "result is missing p_values. Run compute_p_values(...) before calling "
            "conformal_fdp_upper_bound_from_result()."
        )
    if result.calib_scores is None:
        raise ValueError(
            "result is missing calib_scores. The FDP bound requires the number "
            "of calibration scores."
        )
    if result.test_weights is not None or result.calib_weights is not None:
        raise ValueError(
            "conformal_fdp_upper_bound_from_result() supports only unweighted "
            "conformal p-values in this release."
        )
    _validate_fdp_result_scope(result)

    calib_scores = _as_1d_numeric("result.calib_scores", result.calib_scores)
    if calib_scores.size == 0:
        raise ValueError("result.calib_scores must contain at least one score.")

    return conformal_fdp_upper_bound(
        result.p_values,
        n_calibration=calib_scores.size,
        confidence=confidence,
        n_resamples=n_resamples,
        method=method,
        seed=seed,
        boost=boost,
        lower=lower,
        upper=upper,
        beta=beta,
        precision=precision,
        thresholds=thresholds,
    )


def _validate_pruning(pruning: Pruning) -> None:
    """Validate pruning mode type."""
    if not isinstance(pruning, Pruning):
        raise TypeError(
            f"pruning must be an instance of Pruning. Got {type(pruning).__name__}."
        )


def _calib_weight_mass_strictly_above(
    calib_scores: np.ndarray, w_calib: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute weighted calibration mass strictly above each target score."""
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = w_calib[order]
    total_weight = np.sum(sorted_weights)
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="right")
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
    calib_mass_strictly_above: np.ndarray,
    scratch: np.ndarray,
    include_self_weight: bool,
    sorted_test_idx: np.ndarray | None,
    lt_cutoffs: np.ndarray | None,
) -> int:
    """Compute rejection set size |R_j^{(0)}| for test instance j."""
    np.copyto(scratch, calib_mass_strictly_above)
    if include_self_weight:
        if sorted_test_idx is None or lt_cutoffs is None:
            raise ValueError("Internal error: missing score-rank cache for WCS.")
        scratch[sorted_test_idx[: lt_cutoffs[j]]] += w_test[j]
        denominator = sum_calib_weight + w_test[j]
    else:
        denominator = sum_calib_weight
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError(
            "Weighted FDR requires positive finite effective calibration mass."
        )
    scratch[j] = 0.0
    scratch /= denominator
    return _bh_rejection_count(scratch, bh_thresholds)


def _extract_required_wcs_fields(
    result: ConformalResult | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract required WCS arrays from a result bundle."""
    result = _require_result_bundle(result, caller="weighted_false_discovery_control")
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


def _extract_kde_support(
    result: ConformalResult | None,
) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, bool]:
    """Extract optional KDE support metadata for probabilistic estimation."""
    result = _require_result_bundle(result, caller="weighted_false_discovery_control")
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
        calib_mass_strictly_above = sum_calib_weight * (
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
        calib_mass_strictly_above = _calib_weight_mass_strictly_above(
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
    lt_cutoffs: np.ndarray | None = None
    if include_self_weight:
        sorted_test_idx = np.argsort(test_scores_arr, kind="mergesort")
        sorted_scores = test_scores_arr[sorted_test_idx]
        lt_cutoffs = np.searchsorted(sorted_scores, test_scores_arr, side="left")
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
            calib_mass_strictly_above,
            scratch,
            include_self_weight=include_self_weight,
            sorted_test_idx=sorted_test_idx,
            lt_cutoffs=lt_cutoffs,
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
    result: ConformalResult | None,
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


__all__ = [
    "FDPBoundResult",
    "Pruning",
    "conformal_fdp_upper_bound",
    "conformal_fdp_upper_bound_from_result",
    "weighted_false_discovery_control",
    "weighted_false_discovery_control_from_arrays",
]
