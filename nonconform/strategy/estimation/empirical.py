import numpy as np

from nonconform.strategy.estimation.base import BaseEstimation


class Empirical(BaseEstimation):
    """Classical empirical p-value estimation using discrete CDF."""

    def compute_p_values(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute empirical p-values from calibration set.

        Standard: p = (1 + #{calib >= score}) / (1 + n_calib)
        Weighted: p = (w_calib[calib >= score] + w_score) / (sum(w_calib) + w_score)
        """
        if weights is not None:
            return self._compute_weighted(scores, calibration_set, weights)
        return self._compute_standard(scores, calibration_set)

    def _compute_standard(
        self, scores: np.ndarray, calibration_set: np.ndarray
    ) -> np.ndarray:
        sum_ge = np.sum(calibration_set >= scores[:, np.newaxis], axis=1)
        return (1.0 + sum_ge) / (1.0 + len(calibration_set))

    def _compute_weighted(
        self,
        scores: np.ndarray,
        calibration_set: np.ndarray,
        weights: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        w_calib, w_scores = weights
        comparison_matrix = calibration_set >= scores[:, np.newaxis]
        weighted_sum_ge = np.sum(comparison_matrix * w_calib, axis=1)
        numerator = weighted_sum_ge + w_scores
        denominator = np.sum(w_calib) + w_scores
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )
