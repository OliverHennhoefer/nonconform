from collections.abc import Sequence

import numpy as np
from KDEpy import FFTKDE

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
        seed: int | None = None,
    ):
        """Initialize Probabilistic estimation strategy.

        Args:
            kernel: Kernel function or list (list triggers kernel tuning).
                Bandwidth is always auto-tuned.
            n_trials: Number of Optuna trials for tuning.
            cv_folds: CV folds for tuning (-1 for leave-one-out).
            seed: Random seed for reproducibility.
        """
        self._kernel = kernel
        self._n_trials = n_trials
        self._cv_folds = cv_folds
        self._seed = seed

        self._tuned_params: dict | None = None
        self._kde_model: FFTKDE | None = None
        self._calibration_hash: int | None = None

    def compute_p_values(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute continuous p-values using KDE.

        Lazy fitting: tunes and fits KDE on first call or when calibration changes.
        For weighted mode, uses w_calib for tuning and fitting.
        """
        current_hash = hash(calibration_set.tobytes())
        w_calib = weights[0] if weights is not None else None

        if self._kde_model is None or self._calibration_hash != current_hash:
            self._fit_kde(calibration_set, w_calib)
            self._calibration_hash = current_hash

        return self._compute_p_values_from_kde(scores)

    def _fit_kde(self, calibration_set: np.ndarray, weights: np.ndarray | None):
        """Fit KDE with automatic hyperparameter tuning."""
        calibration_set = np.sort(calibration_set.ravel())

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

    def _compute_p_values_from_kde(self, scores: np.ndarray) -> np.ndarray:
        """Compute P(X >= score) from fitted KDE via numerical integration."""
        scores = scores.ravel()
        calib_min, calib_max = self._kde_model.data.min(), self._kde_model.data.max()
        data_range = calib_max - calib_min
        margin = max(0.5 * data_range, 0.1)

        grid_min = calib_min - margin
        grid_max = calib_max + margin
        n_grid_points = 2**14

        eval_grid = np.linspace(grid_min, grid_max, n_grid_points)
        pdf_values = self._kde_model.evaluate(eval_grid)

        dx = (grid_max - grid_min) / (n_grid_points - 1)
        cdf_values = np.cumsum(pdf_values) * dx
        cdf_values = np.clip(cdf_values, 0, 1)

        p_values = []
        for score in scores:
            if score <= grid_min:
                p_value = 1.0
            elif score >= grid_max:
                p_value = 0.0
            else:
                idx = np.searchsorted(eval_grid, score, side="left")
                idx = min(idx, len(cdf_values) - 1)
                p_value = 1.0 - cdf_values[idx]
            p_values.append(p_value)

        return np.clip(np.array(p_values), 0, 1)
