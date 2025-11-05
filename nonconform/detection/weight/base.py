from abc import ABC, abstractmethod

import numpy as np


class BaseWeightEstimator(ABC):
    """Abstract base class for weight estimators in weighted conformal prediction.

    Weight estimators compute importance weights to correct for covariate shift
    between calibration and test distributions. They estimate density ratios
    w(x) = p_test(x) / p_calib(x) which are used to reweight conformal scores
    for better coverage guarantees under distribution shift.

    Subclasses must implement fit(), _get_stored_weights(), and _score_new_data()
    to provide specific weight estimation strategies (e.g., logistic regression,
    random forest). The base class handles all validation in get_weights().
    """

    @abstractmethod
    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray):
        """Estimate density ratio weights"""
        pass

    def get_weights(
        self,
        calibration_samples: np.ndarray | None = None,
        test_samples: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return density ratio weights for calibration and test data.

        Args:
            calibration_samples: Optional calibration data to score. If provided,
                computes weights for this data using the fitted model. If None,
                returns stored weights from fit(). Must provide both or neither.
            test_samples: Optional test data to score. If provided, computes
                weights for this data using the fitted model. If None, returns
                stored weights from fit(). Must provide both or neither.

        Returns:
            Tuple of (calibration_weights, test_weights) as numpy arrays.

        Raises:
            RuntimeError: If fit() has not been called.
            ValueError: If only one of calibration_samples/test_samples is provided.
        """
        # Validation: must be fitted
        if not hasattr(self, "_is_fitted") or not self._is_fitted:
            raise RuntimeError("Must call fit() before get_weights()")

        # Validation: both or neither (not one)
        if (calibration_samples is None) != (test_samples is None):
            raise ValueError(
                "Must provide both calibration_samples and test_samples, or neither. "
                "Cannot score only one set."
            )

        # Dispatch to subclass implementation
        if calibration_samples is None:
            return self._get_stored_weights()
        else:
            return self._score_new_data(calibration_samples, test_samples)

    @abstractmethod
    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights from fit().

        Returns:
            Tuple of (calibration_weights, test_weights) as numpy arrays.
        """
        pass

    @abstractmethod
    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score new data using the fitted model.

        Args:
            calibration_samples: Calibration data to score.
            test_samples: Test data to score.

        Returns:
            Tuple of (calibration_weights, test_weights) as numpy arrays.
        """
        pass
