"""P-value estimation strategies for conformal prediction.

This module provides strategies for computing p-values from calibration scores.

Classes:
    BaseEstimation: Abstract base class for p-value estimation.
    Empirical: Classical empirical p-value estimation using discrete CDF.
    Probabilistic: KDE-based probabilistic p-value estimation.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._internal import Kernel


class BaseEstimation(ABC):
    """Abstract base for p-value estimation strategies."""

    @abstractmethod
    def compute_p_values(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute p-values for test scores.

        Args:
            scores: Test instance anomaly scores (1D array).
            calibration_set: Calibration anomaly scores (1D array).
            weights: Optional (w_calib, w_test) tuple for weighted conformal.

        Returns:
            Array of p-values for each test instance.
        """
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Optional auxiliary data exposed after compute_p_values."""
        return {}

    def set_seed(self, seed: int | None) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value or None.
        """
        if hasattr(self, "_seed"):
            self._seed = seed


class Empirical(BaseEstimation):
    """Classical empirical p-value estimation using discrete CDF.

    Computes p-values using the standard empirical distribution:
    - Standard: p = (1 + #{calib >= score}) / (1 + n_calib)
    - Weighted: p = (w_calib[calib >= score] + w_score) / (sum(w_calib) + w_score)

    Examples:
        ```python
        estimation = Empirical()
        p_values = estimation.compute_p_values(test_scores, calib_scores)
        ```
    """

    def compute_p_values(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute empirical p-values from calibration set."""
        if weights is not None:
            return self._compute_weighted(scores, calibration_set, weights)
        return self._compute_standard(scores, calibration_set)

    def _compute_standard(
        self, scores: np.ndarray, calibration_set: np.ndarray
    ) -> np.ndarray:
        """Standard conformal p-value computation."""
        return calculate_p_val(scores, calibration_set)

    def _compute_weighted(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Weighted conformal p-value computation."""
        w_calib, w_scores = weights
        return calculate_weighted_p_val(scores, calibration_set, w_scores, w_calib)


# Standalone p-value functions (consolidated from utils/stat/statistical.py)
def calculate_p_val(scores: np.ndarray, calibration_set: np.ndarray) -> np.ndarray:
    """Calculate empirical p-values (standalone function).

    Args:
        scores: Test instance anomaly scores (1D array).
        calibration_set: Calibration anomaly scores (1D array).

    Returns:
        Array of p-values for each test instance.
    """
    sorted_cal = np.sort(calibration_set)
    n_cal = len(calibration_set)
    # Count calibration scores >= test score using binary search
    ranks = n_cal - np.searchsorted(sorted_cal, scores, side="left")
    return (1.0 + ranks) / (1.0 + n_cal)


def calculate_weighted_p_val(
    scores: np.ndarray,
    calibration_set: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
) -> np.ndarray:
    """Calculate weighted empirical p-values (standalone function).

    Args:
        scores: Test instance anomaly scores (1D array).
        calibration_set: Calibration anomaly scores (1D array).
        test_weights: Test instance weights (1D array).
        calib_weights: Calibration weights (1D array).

    Returns:
        Array of weighted p-values for each test instance.
    """
    w_calib, w_scores = calib_weights, test_weights
    comparison_matrix = calibration_set >= scores[:, np.newaxis]
    weighted_sum_ge = np.sum(comparison_matrix * w_calib, axis=1)
    numerator = weighted_sum_ge + w_scores
    denominator = np.sum(w_calib) + w_scores
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


class Probabilistic(BaseEstimation):
    """KDE-based probabilistic p-value estimation with continuous values.

    Provides smooth p-values in [0,1] via kernel density estimation.
    Supports automatic hyperparameter tuning and weighted conformal prediction.

    Args:
        kernel: Kernel function or list (list triggers kernel tuning).
            Bandwidth is always auto-tuned. Defaults to Kernel.GAUSSIAN.
        n_trials: Number of Optuna trials for tuning. Defaults to 100.
        cv_folds: CV folds for tuning (-1 for leave-one-out). Defaults to -1.

    Examples:
        ```python
        # Basic usage
        estimation = Probabilistic()
        p_values = estimation.compute_p_values(test_scores, calib_scores)

        # With custom kernel
        estimation = Probabilistic(kernel=Kernel.EPANECHNIKOV)
        ```
    """

    def __init__(
        self,
        kernel: Kernel | Sequence[Kernel] = Kernel.GAUSSIAN,
        n_trials: int = 100,
        cv_folds: int = -1,
    ) -> None:
        self._kernel = kernel
        self._n_trials = n_trials
        self._cv_folds = cv_folds
        self._seed = None

        self._tuned_params: dict | None = None
        self._kde_model = None
        self._calibration_hash: int | None = None
        self._kde_eval_grid: np.ndarray | None = None
        self._kde_cdf_values: np.ndarray | None = None
        self._kde_total_weight: float | None = None

    def compute_p_values(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute continuous p-values using KDE.

        Lazy fitting: tunes and fits KDE on first call or when calibration changes.
        """
        if weights is not None:
            w_calib, w_test = weights
        else:
            w_calib, w_test = None, None

        if weights is None:
            current_hash = hash(calibration_set.tobytes())
        else:
            current_hash = hash((calibration_set.tobytes(), w_calib.tobytes()))

        if self._kde_model is None or self._calibration_hash != current_hash:
            self._fit_kde(calibration_set, w_calib)
            self._calibration_hash = current_hash

        sum_calib_weight = (
            float(np.sum(w_calib))
            if w_calib is not None
            else float(len(calibration_set))
        )

        return self._compute_p_values_from_kde(scores, w_test, sum_calib_weight)

    def _fit_kde(self, calibration_set: np.ndarray, weights: np.ndarray | None) -> None:
        """Fit KDE with automatic hyperparameter tuning."""
        # Lazy import to avoid circular dependency
        try:
            from KDEpy import FFTKDE

            from ._internal import tune_kde_hyperparameters
        except ImportError:
            raise ImportError(
                "Probabilistic estimation requires KDEpy. "
                "Install with: pip install nonconform[all]"
            )

        calibration_set = calibration_set.ravel()

        if weights is not None:
            sort_idx = np.argsort(calibration_set)
            calibration_set = calibration_set[sort_idx]
            weights = weights[sort_idx]
        else:
            calibration_set = np.sort(calibration_set)

        tuning_result = tune_kde_hyperparameters(
            calibration_set=calibration_set,
            kernel_options=self._kernel,
            n_trials=self._n_trials,
            cv_folds=self._cv_folds,
            weights=weights,
            seed=self._seed,
        )
        self._tuned_params = tuning_result
        kernel = tuning_result["kernel"]
        bandwidth = tuning_result["bandwidth"]

        kde = FFTKDE(kernel=kernel.value, bw=bandwidth)
        if weights is not None:
            kde.fit(calibration_set, weights=weights)
        else:
            kde.fit(calibration_set)

        self._kde_model = kde

    def _compute_p_values_from_kde(
        self,
        scores: np.ndarray,
        w_test: np.ndarray | None,
        sum_calib_weight: float,
    ) -> np.ndarray:
        """Compute P(X >= score) from fitted KDE via numerical integration."""
        from scipy import integrate
        from scipy.interpolate import interp1d

        scores = scores.ravel()
        eval_grid, pdf_values = self._kde_model.evaluate(2**14)

        cdf_values = integrate.cumulative_trapezoid(pdf_values, eval_grid, initial=0)
        cdf_values = cdf_values / cdf_values[-1]  # Normalize
        cdf_values = np.clip(cdf_values, 0, 1)  # Safety clipping

        cdf_func = interp1d(
            eval_grid,
            cdf_values,
            kind="linear",
            bounds_error=False,
            fill_value=(0, 1),
        )
        survival = 1.0 - cdf_func(scores)  # P(X >= score)

        self._kde_eval_grid = eval_grid.copy()
        self._kde_cdf_values = cdf_values.copy()
        self._kde_total_weight = float(sum_calib_weight)

        if w_test is None or sum_calib_weight <= 0:
            return np.clip(survival, 0, 1)

        weighted_mass_above = sum_calib_weight * survival
        p_values = np.divide(
            weighted_mass_above,
            sum_calib_weight,
            out=np.zeros_like(weighted_mass_above),
            where=sum_calib_weight != 0,
        )

        return np.clip(p_values, 0, 1)

    def get_metadata(self) -> dict[str, Any]:
        """Return KDE metadata after p-value computation."""
        if (
            self._kde_eval_grid is None
            or self._kde_cdf_values is None
            or self._kde_total_weight is None
        ):
            return {}
        return {
            "kde": {
                "eval_grid": self._kde_eval_grid.copy(),
                "cdf_values": self._kde_cdf_values.copy(),
                "total_weight": float(self._kde_total_weight),
            }
        }


__all__ = [
    "BaseEstimation",
    "Empirical",
    "Kernel",
    "Probabilistic",
    "calculate_p_val",
    "calculate_weighted_p_val",
]
