import numpy as np

from nonconform.estimation.weight.base import BaseWeightEstimator


class IdentityWeightEstimator(BaseWeightEstimator):
    """Identity weight estimator that returns uniform weights.

    This estimator assumes no covariate shift and returns weights of 1.0
    for all samples. Useful as a baseline or when covariate shift is known
    to be minimal.

    This effectively makes weighted conformal prediction equivalent to
    standard conformal prediction.
    """

    def __init__(self):
        self._n_calib = 0
        self._n_test = 0
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the identity weight estimator.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.
        """
        self._n_calib = calibration_samples.shape[0]
        self._n_test = test_samples.shape[0]
        self._is_fitted = True

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return uniform weights of 1.0 for all samples.

        Returns:
            Tuple of (calibration_weights, test_weights) with all values = 1.0.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before get_weights()")

        calib_weights = np.ones(self._n_calib, dtype=np.float64)
        test_weights = np.ones(self._n_test, dtype=np.float64)

        return calib_weights, test_weights
