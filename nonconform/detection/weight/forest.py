import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nonconform.detection.weight.base import BaseWeightEstimator


class ForestWeightEstimator(BaseWeightEstimator):
    """Random Forest-based weight estimator for covariate shift.

    Uses Random Forest classifier to estimate density ratios between calibration
    and test distributions. Random Forest can capture non-linear relationships
    and complex interactions between features, making it suitable for handling
    more complex covariate shift patterns than logistic regression.

    The Random Forest is trained to distinguish between calibration and test samples,
    and the predicted probabilities are used to compute importance weights
    w(x) = p_test(x) / p_calib(x).

    Args:
        n_estimators (int): Number of trees in the forest. Defaults to 100.
        max_depth (int, optional): Maximum depth of trees. If None, nodes are
            expanded until all leaves are pure. Defaults to 5 to prevent overfitting.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
            Defaults to 10 to prevent overfitting.
        clip_quantile (float): Quantile for weight clipping. If 0.05, clips to
            5th and 95th percentiles. If None, uses fixed [0.35, 45.0] range.
        seed (int, optional): Random seed for reproducible results.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 5,
        min_samples_leaf: int = 10,
        clip_quantile: float = 0.05,
        seed: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.clip_quantile = clip_quantile
        self.seed = seed
        self._w_calib = None
        self._w_test = None
        self._model = None
        self._clip_bounds = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the Random Forest weight estimator on calibration and test samples.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.

        Raises:
            ValueError: If calibration_samples is empty.
        """
        if calibration_samples.shape[0] == 0:
            raise ValueError("Calibration samples are empty. Cannot compute weights.")

        # Prepare training data using shared helper
        x_joint, y_joint = self._prepare_training_data(
            calibration_samples, test_samples, self.seed
        )

        # Build Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            class_weight="balanced",
            n_jobs=-1,  # Use all available cores
        )
        model.fit(x_joint, y_joint)
        self._model = model

        # Compute weights using shared helpers
        calib_prob = model.predict_proba(calibration_samples)
        test_prob = model.predict_proba(test_samples)
        w_calib, w_test = self._compute_density_ratios(calib_prob, test_prob)

        # Compute and apply clipping
        self._clip_bounds = self._compute_clip_bounds(
            w_calib, w_test, self.clip_quantile
        )
        self._w_calib, self._w_test = self._clip_weights(
            w_calib, w_test, self._clip_bounds
        )
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights from fit()."""
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score new data using fitted model."""
        calib_prob = self._model.predict_proba(calibration_samples)
        test_prob = self._model.predict_proba(test_samples)
        w_calib, w_test = self._compute_density_ratios(calib_prob, test_prob)
        return self._clip_weights(w_calib, w_test, self._clip_bounds)
