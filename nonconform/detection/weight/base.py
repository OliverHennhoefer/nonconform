from abc import ABC, abstractmethod

import numpy as np


class BaseWeightEstimator(ABC):
    """Abstract base class for weight estimators in weighted conformal prediction.

    Weight estimators compute importance weights to correct for covariate shift
    between calibration and test distributions. They estimate density ratios
    w(x) = p_test(x) / p_calib(x) which are used to reweight conformal scores
    for better coverage guarantees under distribution shift.

    Subclasses must implement the fit() and get_weights() methods to provide
    specific weight estimation strategies (e.g., logistic regression, random forest).
    """

    @abstractmethod
    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray):
        """Estimate density ratio weights"""
        pass

    @abstractmethod
    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (calib_weights, test_weights)"""
        pass
