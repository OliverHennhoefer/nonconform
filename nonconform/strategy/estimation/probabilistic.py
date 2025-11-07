from collections.abc import Sequence
from typing import Any

import numpy as np
from KDEpy import FFTKDE
from scipy import integrate
from scipy.interpolate import interp1d

from nonconform.strategy.estimation.base import BaseEstimation
from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.tuning import tune_kde_hyperparameters


class Probabilistic(BaseEstimation):
    """KDE-based probabilistic p-value estimation with continuous values.

    Provides smooth p-values in [0,1] via kernel density estimation.
    Supports automatic hyperparameter tuning and weighted conformal prediction.
    """

    def __init__(
        self,
        kernel: Kernel | Sequence[Kernel] = Kernel.GAUSSIAN,
        n_trials: int = 100,
        cv_folds: int = -1,
    ):
        """Initialize Probabilistic estimation strategy.

        Args:
            kernel: Kernel function or list (list triggers kernel tuning).
                Bandwidth is always auto-tuned.
            n_trials: Number of Optuna trials for tuning.
            cv_folds: CV folds for tuning (-1 for leave-one-out).
        """
        self._kernel = kernel
        self._n_trials = n_trials
        self._cv_folds = cv_folds
        self._seed = None

        self._tuned_params: dict | None = None
        self._kde_model: FFTKDE | None = None
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
        For weighted mode, uses w_calib for tuning and fitting,
        and both weights for p-value computation.
        """
        # Extract calibration and test weights
        if weights is not None:
            w_calib, w_test = weights
        else:
            w_calib, w_test = None, None

        # Default case: no weights (standard conformal prediction)
        if weights is None:
            current_hash = hash(calibration_set.tobytes())
        else:
            # Weighted case: covariate shift handling
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

    def _fit_kde(self, calibration_set: np.ndarray, weights: np.ndarray | None):
        """Fit KDE with automatic hyperparameter tuning."""
        calibration_set = calibration_set.ravel()

        # Sort data and weights together to maintain correspondence
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
        """Compute P(X >= score) from fitted KDE via numerical integration.

        For standard conformal: returns survival function from KDE.
        For weighted conformal: applies weighted conformal formula using test weights.
        """
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
            fill_value=(0, 1),  # CDF=0 below grid_min, CDF=1 above grid_max
        )
        survival = 1.0 - cdf_func(scores)  # P(X >= score) from KDE

        # Cache metadata for downstream consumers
        self._kde_eval_grid = eval_grid.copy()
        self._kde_cdf_values = cdf_values.copy()
        self._kde_total_weight = float(sum_calib_weight)

        # Standard conformal: return survival function directly
        if w_test is None:
            return np.clip(survival, 0, 1)

        weighted_mass_above = sum_calib_weight * survival
        denominator = sum_calib_weight
        p_values = np.divide(
            weighted_mass_above,
            denominator,
            out=np.zeros_like(weighted_mass_above),
            where=denominator != 0,
        )

        return np.clip(p_values, 0, 1)

    def get_metadata(self) -> dict[str, Any]:
        """Return KDE metadata after p-value computation.

        Exposes the fitted KDE's evaluation grid, CDF values, and total weight
        for downstream analysis. Returns empty dict if KDE hasn't been fitted yet.

        Returns:
            Dictionary with 'kde' key containing eval_grid, cdf_values, and
            total_weight. Empty dict if compute_p_values() hasn't been called.
        """
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
