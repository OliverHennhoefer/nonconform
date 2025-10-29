from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from nonconform.utils.func.decorator import _ensure_numpy_array


class BaseConformalDetector(ABC):
    """Abstract base class for all conformal anomaly detectors.

    Defines the core interface that all conformal anomaly detection implementations
    must provide. This ensures consistent behavior across different conformal methods
    (standard, weighted, etc.) while maintaining flexibility.

    **Design Pattern:**
        All conformal detectors follow a two-phase workflow:
        1. **Calibration Phase**: `fit()` trains detector, computes calibration scores
        2. **Inference Phase**: `predict()` converts new data scores to valid p-values

    **Implementation Requirements:**
        Subclasses must implement both abstract methods to provide:
        - Training/calibration logic in `fit()`
        - P-value generation logic in `predict()`

    Note:
        This is an abstract class and cannot be instantiated directly. Use concrete
        implementations like `StandardConformalDetector` or `WeightedConformalDetector`.
    """

    @_ensure_numpy_array
    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray) -> None:
        """Fit the detector model(s) and compute calibration scores.

        Args:
            x (pd.DataFrame | np.ndarray): The dataset used for
                fitting the model(s) and determining calibration scores.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @_ensure_numpy_array
    @abstractmethod
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates or p-values for new data.

        Args:
            x (pd.DataFrame | np.ndarray): The new data instances
                for which to make anomaly estimates.
            raw (bool, optional): Whether to return raw anomaly scores or
                processed anomaly estimates (e.g., p-values). Defaults to False.

        Returns:
            np.ndarray: An array containing the anomaly estimates.
        """
        raise NotImplementedError("Subclasses must implement predict()")
