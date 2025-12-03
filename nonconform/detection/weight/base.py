"""Base classes for weight estimation in weighted conformal prediction."""

from abc import ABC, abstractmethod

import numpy as np

# Default clipping bounds when adaptive clipping is disabled
DEFAULT_CLIP_BOUNDS = (0.35, 45.0)

# Small epsilon to prevent division by zero in probability ratios
EPSILON = 1e-9


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

    # --------------------------------------------------------------------------
    # Helper methods for subclasses (shared logic)
    # --------------------------------------------------------------------------

    @staticmethod
    def _prepare_training_data(
        calibration_samples: np.ndarray,
        test_samples: np.ndarray,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare labeled and shuffled training data for classifier-based estimation.

        Labels calibration samples as 0 and test samples as 1, then shuffles.

        Args:
            calibration_samples: Calibration data samples.
            test_samples: Test data samples.
            seed: Random seed for shuffling.

        Returns:
            Tuple of (X, y) arrays ready for classifier training.
        """
        # Label calibration samples as 0, test samples as 1
        calib_labeled = np.hstack(
            (calibration_samples, np.zeros((calibration_samples.shape[0], 1)))
        )
        test_labeled = np.hstack((test_samples, np.ones((test_samples.shape[0], 1))))

        # Combine and shuffle
        joint_labeled = np.vstack((calib_labeled, test_labeled))
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        return x_joint, y_joint

    @staticmethod
    def _compute_density_ratios(
        calib_prob: np.ndarray, test_prob: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute density ratios from classifier probabilities.

        Args:
            calib_prob: Predicted probabilities for calibration samples (n_calib, 2).
            test_prob: Predicted probabilities for test samples (n_test, 2).

        Returns:
            Tuple of (w_calib, w_test) density ratio weights.
        """
        # w(z) = p_test(z) / p_calib(z) = P(label=1|z) / P(label=0|z)
        w_calib = calib_prob[:, 1] / (calib_prob[:, 0] + EPSILON)
        w_test = test_prob[:, 1] / (test_prob[:, 0] + EPSILON)
        return w_calib, w_test

    @staticmethod
    def _compute_clip_bounds(
        w_calib: np.ndarray,
        w_test: np.ndarray,
        clip_quantile: float | None,
    ) -> tuple[float, float]:
        """Compute clipping bounds for weight stabilization.

        Args:
            w_calib: Calibration weights.
            w_test: Test weights.
            clip_quantile: Quantile for adaptive clipping (e.g., 0.05 clips to
                5th-95th percentile). If None, uses default fixed bounds.

        Returns:
            Tuple of (lower_bound, upper_bound) for clipping.
        """
        if clip_quantile is not None:
            all_weights = np.concatenate([w_calib, w_test])
            lower_bound = np.percentile(all_weights, clip_quantile * 100)
            upper_bound = np.percentile(all_weights, (1 - clip_quantile) * 100)
            return (lower_bound, upper_bound)
        return DEFAULT_CLIP_BOUNDS

    @staticmethod
    def _clip_weights(
        w_calib: np.ndarray,
        w_test: np.ndarray,
        clip_bounds: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply clipping to stabilize weights.

        Args:
            w_calib: Calibration weights.
            w_test: Test weights.
            clip_bounds: Tuple of (lower, upper) bounds.

        Returns:
            Tuple of clipped (w_calib, w_test) arrays.
        """
        return np.clip(w_calib, *clip_bounds), np.clip(w_test, *clip_bounds)
