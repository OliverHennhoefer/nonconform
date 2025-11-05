import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nonconform.detection.weight.base import BaseWeightEstimator


class LogisticWeightEstimator(BaseWeightEstimator):
    """Logistic regression-based weight estimator for covariate shift.

    Uses logistic regression to estimate density ratios between calibration
    and test distributions by training a classifier to distinguish between
    the two samples. The predicted probabilities are used to compute
    importance weights w(x) = p_test(x) / p_calib(x).

    Args:
        regularization (str or float): Regularization parameter for logistic regression.
            If 'auto', uses default sklearn parameter. If float, uses as C parameter.
        clip_quantile (float): Quantile for weight clipping. If 0.05, clips to
            5th and 95th percentiles. If None, uses fixed [0.35, 45.0] range.
        seed (int, optional): Random seed for reproducible results.
        class_weight (str or dict, optional): Weights associated with classes like
            {class_label: weight}.
            If 'balanced', uses n_samples / (n_classes * np.bincount(y)).
            Defaults to 'balanced'.
        max_iter (int, optional): Max. number of iterations for the solver to converge.
            Defaults to 1000.
    """

    def __init__(
        self,
        regularization="auto",
        clip_quantile=0.05,
        seed=None,
        class_weight="balanced",
        max_iter=1_000,
    ):
        self.regularization = regularization
        self.clip_quantile = clip_quantile
        self.seed = seed
        self.class_weight = class_weight
        self.max_iter = max_iter
        self._w_calib = None
        self._w_test = None
        self._model = None
        self._clip_bounds = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        """Fit the weight estimator on calibration and test samples.

        Args:
            calibration_samples: Array of calibration data samples.
            test_samples: Array of test data samples.

        Raises:
            ValueError: If calibration_samples is empty.
        """
        if calibration_samples.shape[0] == 0:
            raise ValueError("Calibration samples are empty. Cannot compute weights.")

        # Label calibration samples as 0, test samples as 1
        calib_labeled = np.hstack(
            (
                calibration_samples,
                np.zeros((calibration_samples.shape[0], 1)),
            )
        )
        test_labeled = np.hstack((test_samples, np.ones((test_samples.shape[0], 1))))

        # Combine and shuffle
        joint_labeled = np.vstack((calib_labeled, test_labeled))
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        # Build logistic regression pipeline
        c_param = 1.0 if self.regularization == "auto" else float(self.regularization)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_param,
                max_iter=self.max_iter,
                random_state=self.seed,
                verbose=0,
                class_weight=self.class_weight,
            ),
            memory=None,
        )
        model.fit(x_joint, y_joint)
        self._model = model

        # Compute probabilities
        calib_prob = model.predict_proba(calibration_samples)
        test_prob = model.predict_proba(test_samples)

        # Compute density ratios w(z) = p_test(z) / p_calib(z)
        # p_calib(z) = P(label=0 | z) ; p_test(z) = P(label=1 | z)
        w_calib = calib_prob[:, 1] / (calib_prob[:, 0] + 1e-9)
        w_test = test_prob[:, 1] / (test_prob[:, 0] + 1e-9)

        if self.clip_quantile is not None:
            # Adaptive clipping based on percentiles
            all_weights = np.concatenate([w_calib, w_test])
            lower_bound = np.percentile(all_weights, self.clip_quantile * 100)
            upper_bound = np.percentile(all_weights, (1 - self.clip_quantile) * 100)
            self._clip_bounds = (lower_bound, upper_bound)
        else:
            # Fixed clipping fallback mirroring ForestWeightEstimator behaviour
            self._clip_bounds = (0.35, 45.0)

        self._w_calib = np.clip(w_calib, *self._clip_bounds)
        self._w_test = np.clip(w_test, *self._clip_bounds)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return stored weights from fit()."""
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score new data using fitted model."""
        # Score provided data using fitted model
        calib_prob = self._model.predict_proba(calibration_samples)
        test_prob = self._model.predict_proba(test_samples)

        # Compute density ratios
        w_calib = calib_prob[:, 1] / (calib_prob[:, 0] + 1e-9)
        w_test = test_prob[:, 1] / (test_prob[:, 0] + 1e-9)

        # Apply same clipping bounds as fit()
        w_calib_clipped = np.clip(w_calib, *self._clip_bounds)
        w_test_clipped = np.clip(w_test, *self._clip_bounds)

        return w_calib_clipped, w_test_clipped
