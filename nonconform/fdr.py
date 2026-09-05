"""Public false-discovery procedures for conformal anomaly evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nonconform.structures import ConformalResult

from ._internal import Pruning
from ._internal import fdp_bounds as _fdp_bounds
from ._internal import wcs as _wcs
from ._internal.validation import (
    as_1d_numeric as _as_1d_numeric,
)
from ._internal.validation import (
    validate_finite as _validate_finite,
)
from ._internal.validation import (
    validate_non_negative_finite as _validate_non_negative_finite,
)
from ._internal.validation import (
    validate_probability as _validate_probability,
)


def _as_1d_or_2d_numeric(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize array-like input into a strict 1D or 2D float ndarray."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if arr.ndim not in {1, 2}:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape!r}.")
    return arr


def _as_e_values(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize e-values into a non-empty, validated 1D float array."""
    e_values = _as_1d_numeric(name, values).astype(float, copy=True)
    if e_values.size == 0:
        raise ValueError(f"{name} must contain at least one e-value.")
    _validate_non_negative_finite(name, e_values)
    return e_values


def _as_repeated_scores(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return score arrays as ``(n_repetitions, n_points)`` matrices."""
    test_arr = _as_1d_or_2d_numeric("test_scores", test_scores).astype(float, copy=True)
    calib_arr = _as_1d_or_2d_numeric("calib_scores", calib_scores).astype(
        float, copy=True
    )
    _validate_finite("test_scores", test_arr)
    _validate_finite("calib_scores", calib_arr)

    if test_arr.ndim != calib_arr.ndim:
        raise ValueError("test_scores and calib_scores must have the same dimension.")
    if test_arr.ndim == 1:
        if test_arr.size == 0:
            raise ValueError("test_scores must contain at least one score.")
        if calib_arr.size == 0:
            raise ValueError("calib_scores must contain at least one score.")
        return test_arr.reshape(1, -1), calib_arr.reshape(1, -1)

    if test_arr.shape[0] == 0:
        raise ValueError("test_scores must contain at least one repetition.")
    if test_arr.shape[1] == 0:
        raise ValueError("test_scores must contain at least one score.")
    if calib_arr.shape[1] == 0:
        raise ValueError("calib_scores must contain at least one score.")
    if test_arr.shape[0] != calib_arr.shape[0]:
        raise ValueError(
            "test_scores and calib_scores must have the same number of repetitions."
        )
    return test_arr, calib_arr


def _conformal_e_value_threshold(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
) -> float:
    """Return the score threshold used for one conformal e-value split."""
    n_test = test_scores.size
    n_calib = calib_scores.size
    candidates = np.unique(np.concatenate([test_scores, calib_scores]))
    for threshold in np.sort(candidates):
        test_count = int(np.count_nonzero(test_scores >= threshold))
        if test_count == 0:
            continue
        calib_count = int(np.count_nonzero(calib_scores >= threshold))
        fdp_hat = (n_test / n_calib) * (calib_count / test_count)
        if fdp_hat <= alpha_bh:
            return float(threshold)
    return float("inf")


def _e_bh_selection(e_values: np.ndarray, *, alpha: float) -> tuple[np.ndarray, float]:
    """Return e-BH selection mask and selected e-value threshold."""
    m = e_values.size
    order = np.argsort(-e_values, kind="mergesort")
    sorted_e_values = e_values[order]
    thresholds = m / (alpha * np.arange(1, m + 1, dtype=float))
    passing = np.nonzero(sorted_e_values >= thresholds)[0]
    if passing.size == 0:
        return np.zeros(m, dtype=bool), float("inf")

    n_selected = int(passing[-1] + 1)
    selected = np.zeros(m, dtype=bool)
    selected[order[:n_selected]] = True
    return selected, float(sorted_e_values[n_selected - 1])


@dataclass(slots=True, frozen=True)
class EValueSelectionResult:
    """Batch e-value FDR selection result.

    The result bundles aggregated conformal e-values with the e-BH discoveries
    selected at the requested FDR level.
    """

    e_values: np.ndarray
    selected: np.ndarray
    alpha: float
    alpha_bh: float
    e_threshold: float
    n_repetitions: int

    def __post_init__(self) -> None:
        """Copy array fields so the result is independent of caller inputs."""
        object.__setattr__(
            self,
            "e_values",
            np.asarray(self.e_values, dtype=float).copy(),
        )
        object.__setattr__(
            self,
            "selected",
            np.asarray(self.selected, dtype=bool).copy(),
        )


@dataclass(slots=True)
class FDPBoundResult:
    """Post-hoc simultaneous FDP upper-bound certificate.

    The result evaluates a high-confidence bound on the realized false discovery
    proportion for threshold selections of conformal p-values.

    Attributes:
        p_values: Empirical conformal p-values used for the certificate.
        thresholds: Thresholds evaluated when the certificate was built.
        rejection_counts: Number of p-values at or below each threshold.
        fdp_upper_bounds: Simultaneous realized-FDP upper bounds.
        n_calibration: Calibration sample size.
        n_test: Testing-family size.
        confidence: Requested simultaneous coverage probability.
        method: ECDF-envelope method.
        n_resamples: Requested Monte Carlo sample size.
        boost: Whether threshold-specific sharpening was enabled.
        seed: Monte Carlo seed, or None.
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
                self._bj_lower_bounds,
                dtype=float,
            ).copy()
        self.precision_lower_bounds = 1.0 - self.fdp_upper_bounds

    def _ecdf_upper_bound(self, x: np.ndarray) -> np.ndarray:
        """Evaluate this result's ECDF envelope."""
        return _fdp_bounds.ecdf_upper_bound_from_params(
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
        """Evaluate the simultaneous FDP upper envelope at thresholds."""
        threshold_arr, scalar_input = _fdp_bounds.as_threshold_query(threshold)
        bounds = _fdp_bounds.evaluate_fdp_upper_bound(
            self.p_values,
            threshold_arr,
            ecdf_upper_bound=self._ecdf_upper_bound,
            boost=self.boost,
        )
        if scalar_input:
            return float(bounds[0])
        return bounds

    def precision_at(self, threshold: float | np.ndarray) -> float | np.ndarray:
        """Return ``1 - bound_at(threshold)`` as a precision lower bound."""
        return 1.0 - self.bound_at(threshold)

    def to_frame(self, thresholds: np.ndarray | None = None) -> pd.DataFrame:
        """Return threshold-level FDP certificates as a DataFrame."""
        if thresholds is None:
            threshold_arr = self.thresholds
            fdp_bounds = self.fdp_upper_bounds
            precision_bounds = self.precision_lower_bounds
        else:
            threshold_arr = _fdp_bounds.as_thresholds(thresholds, self.p_values)
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
        """Return the mask induced by ``p_values <= threshold``."""
        threshold_arr, scalar_input = _fdp_bounds.as_threshold_query(threshold)
        if not scalar_input:
            raise ValueError("threshold must be a scalar for select().")
        return self.p_values <= threshold_arr[0]


def conformal_e_values(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
) -> np.ndarray:
    """Compute derandomized conformal e-values from split-conformal scores.

    Args:
        test_scores: Test anomaly scores. Shape ``(n_test,)`` for one split or
            ``(n_repetitions, n_test)`` for repeated splits.
        calib_scores: Calibration anomaly scores with matching split dimension.
        alpha_bh: Inner BH-style threshold level used to construct each split's
            conformal e-values.

    Returns:
        Aggregated e-values of shape ``(n_test,)``. Repeated splits are
        aggregated with uniform weights.
    """
    alpha_bh_value = _validate_probability("alpha_bh", alpha_bh)
    test_arr, calib_arr = _as_repeated_scores(test_scores, calib_scores)

    e_values_by_split = np.zeros_like(test_arr, dtype=float)
    for idx in range(test_arr.shape[0]):
        threshold = _conformal_e_value_threshold(
            test_arr[idx],
            calib_arr[idx],
            alpha_bh=alpha_bh_value,
        )
        calib_count = int(np.count_nonzero(calib_arr[idx] >= threshold))
        denominator = (1.0 + calib_count) / (1.0 + calib_arr.shape[1])
        e_values_by_split[idx] = (test_arr[idx] >= threshold).astype(float) / (
            denominator
        )

    return np.mean(e_values_by_split, axis=0)


def e_value_false_discovery_control(
    e_values: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply the e-BH procedure to e-values.

    Args:
        e_values: Non-negative e-values; larger values are stronger evidence.
        alpha: Target FDR level in ``(0, 1)``.

    Returns:
        Boolean selection mask of shape ``(n_test,)``.
    """
    alpha_value = _validate_probability("alpha", alpha)
    e_values_arr = _as_e_values("e_values", e_values)
    selected, _ = _e_bh_selection(e_values_arr, alpha=alpha_value)
    return selected


def conformal_e_value_selection(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha: float = 0.05,
    alpha_bh: float | None = None,
) -> EValueSelectionResult:
    """Compute conformal e-values and apply e-BH selection.

    This is an expert batch workflow for stabilizing repeated split-conformal
    anomaly decisions. Existing p-value, BH, and weighted FDR workflows are
    unchanged.
    """
    alpha_value = _validate_probability("alpha", alpha)
    alpha_bh_value = alpha_value / 10.0 if alpha_bh is None else alpha_bh
    alpha_bh_value = _validate_probability("alpha_bh", alpha_bh_value)
    test_arr, calib_arr = _as_repeated_scores(test_scores, calib_scores)

    e_values = conformal_e_values(
        test_arr,
        calib_arr,
        alpha_bh=alpha_bh_value,
    )
    selected, e_threshold = _e_bh_selection(e_values, alpha=alpha_value)
    return EValueSelectionResult(
        e_values=e_values,
        selected=selected,
        alpha=alpha_value,
        alpha_bh=alpha_bh_value,
        e_threshold=e_threshold,
        n_repetitions=int(test_arr.shape[0]),
    )


def conformal_fdp_upper_bound(
    p_values: np.ndarray,
    *,
    n_calibration: int,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    method: str = _fdp_bounds.DEFAULT_METHOD,
    seed: int | None = None,
    boost: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    beta: float = 0.5,
    precision: float = 1e-8,
    thresholds: np.ndarray | None = None,
) -> FDPBoundResult:
    """Compute post-hoc simultaneous FDP upper bounds for conformal p-values.

    This implements simultaneous FDP envelopes from Song, Jin, and Candès for
    unweighted conformal p-values from a fixed scoring map. Choose ``method``
    before inspecting the resulting curve. This array-level entry point cannot
    inspect how its p-values were produced, so the caller is responsible for the
    reference method's assumptions.

    Args:
        p_values: Non-empty one-dimensional testing-family p-values in
            ``[0, 1]``.
        n_calibration: Number of calibration scores used for every p-value.
        confidence: Simultaneous coverage probability in ``(0, 1)``.
        n_resamples: Positive Monte Carlo sample size for ``"mc_"`` methods.
        method: One of ``"mc_thc"``, ``"mc_hc"``, ``"mc_ks"``, ``"ks"``, or
            ``"mc_bj"``.
        seed: Non-negative Monte Carlo seed, or None.
        boost: Whether to apply threshold-specific envelope sharpening.
        lower: Lower truncation point for ``method="mc_thc"``.
        upper: Upper truncation point for ``method="mc_thc"``.
        beta: Positive truncation exponent for ``method="mc_thc"``.
        precision: Positive numerical tolerance for ``"mc_bj"`` inversion.
        thresholds: Optional one-dimensional thresholds in ``[0, 1]``.

    Returns:
        A simultaneous threshold-indexed FDP certificate.

    References:
        Song, Jin, and Candès, "Everywhere Valid Bounds on False Discovery
        Proportions in Conformal Inference" (2026), arXiv:2605.20726.
    """
    p_values_arr = _fdp_bounds.as_p_values("p_values", p_values)
    n_calibration = _fdp_bounds.validate_positive_integer(
        "n_calibration",
        n_calibration,
    )
    n_resamples = _fdp_bounds.validate_positive_integer(
        "n_resamples",
        n_resamples,
    )
    confidence = _fdp_bounds.validate_probability("confidence", confidence)
    method = _fdp_bounds.validate_method(method)
    seed = _fdp_bounds.validate_optional_seed("seed", seed)
    if not isinstance(boost, bool):
        raise TypeError("boost must be a boolean value.")
    lower, upper, beta = _fdp_bounds.validate_truncation(lower, upper, beta)
    precision = _fdp_bounds.validate_positive_finite("precision", precision)

    evaluated_thresholds = _fdp_bounds.as_thresholds(thresholds, p_values_arr)
    summary_quantile, bj_lower_bounds = _fdp_bounds.build_ecdf_upper_bound(
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
        return _fdp_bounds.ecdf_upper_bound_from_params(
            x,
            method=method,
            summary_quantile=summary_quantile,
            lower=lower,
            upper=upper,
            beta=beta,
            bj_lower_bounds=bj_lower_bounds,
            n_test=p_values_arr.size,
        )

    fdp_bounds = _fdp_bounds.evaluate_fdp_upper_bound(
        p_values_arr,
        evaluated_thresholds,
        ecdf_upper_bound=ecdf_upper_bound,
        boost=boost,
    )
    rejection_counts = np.searchsorted(
        np.sort(p_values_arr),
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
    method: str = _fdp_bounds.DEFAULT_METHOD,
    seed: int | None = None,
    boost: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    beta: float = 0.5,
    precision: float = 1e-8,
    thresholds: np.ndarray | None = None,
) -> FDPBoundResult:
    """Compute simultaneous FDP bounds from a compatible result bundle.

    The result must contain unweighted ``Empirical`` p-values from ``Split`` or
    detached calibration. Weighted, KDE, conditionally calibrated, and
    resampling-strategy bundles are rejected.

    Args:
        result: Result produced by ``compute_p_values()`` or ``select()``.
        confidence: Simultaneous coverage probability in ``(0, 1)``.
        n_resamples: Positive Monte Carlo sample size for ``"mc_"`` methods.
        method: ECDF-envelope method accepted by
            :func:`conformal_fdp_upper_bound`.
        seed: Non-negative Monte Carlo seed, or None.
        boost: Whether to apply threshold-specific envelope sharpening.
        lower: Lower truncation point for ``method="mc_thc"``.
        upper: Upper truncation point for ``method="mc_thc"``.
        beta: Positive truncation exponent for ``method="mc_thc"``.
        precision: Positive numerical tolerance for ``"mc_bj"`` inversion.
        thresholds: Optional one-dimensional thresholds in ``[0, 1]``.

    Returns:
        A simultaneous threshold-indexed FDP certificate.
    """
    if result is None:
        raise ValueError(
            "result must be a ConformalResult, got None. Run compute_p_values(...) "
            "before calling conformal_fdp_upper_bound_from_result()."
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
    _fdp_bounds.validate_result_scope(result)

    calib_scores = _fdp_bounds.as_1d_numeric(
        "result.calib_scores",
        result.calib_scores,
    )
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


def weighted_false_discovery_control(
    result: ConformalResult | None,
    *,
    alpha: float = 0.05,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Apply weighted conformalized selection to a result bundle.

    The result must contain p-values, test and calibration scores, and matching
    non-negative weights for the same complete testing family. Validity also
    depends on the weighted-conformal covariate-shift assumptions.

    Args:
        result: Weighted detector result for the target family.
        alpha: Nominal FDR target in ``(0, 1)``.
        pruning: Deterministic, homogeneous-randomized, or
            heterogeneous-randomized WCS pruning rule.
        seed: Non-negative seed for randomized pruning, or None.

    Returns:
        Boolean selection mask aligned with the result's test rows.
    """
    p_values, test_scores, calib_scores, test_weights, calib_weights = (
        _wcs.extract_required_fields(result)
    )
    kde_support, use_self_weight = _wcs.extract_kde_support(result)
    return _wcs.run(
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
    """Apply weighted conformalized selection to explicit arrays.

    This low-level API cannot verify provenance. All arrays must come from the
    same calibration construction and complete target family.

    Args:
        p_values: One p-value per test observation.
        test_scores: One anomalous-higher score per test observation.
        calib_scores: Calibration scores in the same orientation.
        test_weights: Non-negative target-density weights for test observations.
        calib_weights: Non-negative target-density weights for calibration
            observations.
        alpha: Nominal FDR target in ``(0, 1)``.
        pruning: WCS pruning rule.
        seed: Non-negative seed for randomized pruning, or None.

    Returns:
        Boolean selection mask aligned with the test arrays.
    """
    return _wcs.run(
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
    "EValueSelectionResult",
    "FDPBoundResult",
    "Pruning",
    "conformal_e_value_selection",
    "conformal_e_values",
    "conformal_fdp_upper_bound",
    "conformal_fdp_upper_bound_from_result",
    "e_value_false_discovery_control",
    "weighted_false_discovery_control",
    "weighted_false_discovery_control_from_arrays",
]
