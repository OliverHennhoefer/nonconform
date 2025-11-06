from abc import ABC, abstractmethod
from typing import Any

import numpy as np


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
