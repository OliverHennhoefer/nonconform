import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nonconform.estimation.base import BaseConformalDetector
from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.decorator import _ensure_numpy_array
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import _set_params
from nonconform.utils.stat.aggregation import aggregate
from nonconform.utils.stat.statistical import calculate_weighted_p_val
from pyod.models.base import BaseDetector


class WeightedConformalDetector(BaseConformalDetector):
    """Weighted conformal anomaly detector for handling covariate shift.

    Adapts conformal prediction to scenarios where test data distribution differs from
    training data (covariate shift) by incorporating importance weights. This ensures
    valid p-values and FDR control even when the exchangeability assumption is violated.

    The detector automatically estimates density ratios between calibration and test
    distributions using logistic regression, then applies these weights to maintain
    statistical validity under distribution shift.

    Example:
        Using weighted conformal for data with suspected distribution shift:

        ```python
        from pyod.models.lof import LOF
        from nonconform.estimation import WeightedConformalDetector
        from nonconform.strategy import Split

        # Create weighted conformal detector
        detector = WeightedConformalDetector(
            detector=LOF(n_neighbors=20), strategy=Split(n_calib=0.2), seed=42
        )

        # Fit on training data (normal samples only)
        detector.fit(X_train)

        # Get weighted p-values that account for distribution shift
        p_values = detector.predict(X_test)

        # Apply FDR control as usual
        from scipy.stats import false_discovery_control

        decisions = false_discovery_control(p_values, method="bh") <= 0.1
        ```

    Attributes:
        detector: The underlying PyOD anomaly detection model.
        strategy: The calibration strategy for computing weighted p-values.
        aggregation: Method for combining scores from multiple models.
        seed: Random seed for reproducible results.
        detector_set: List of trained detector models (populated after fit).
        calibration_set: Calibration scores for p-value computation (populated by fit).
        is_fitted: Whether the detector has been fitted.
        calibration_samples: Data instances used for calibration (+ weight computation).
    """

    def __init__(
        self,
        detector: BaseDetector,
        strategy: BaseStrategy,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
    ):
        """Initialize the WeightedConformalDetector.

        Args:
            detector (BaseDetector): A PyOD anomaly detector instance. It will
                be configured with the specified seed.
            strategy (BaseStrategy): A calibration strategy instance.
            aggregation (Aggregation, optional): Method used for aggregating
                scores from multiple detector models. Defaults to Aggregation.MEDIAN.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.

        Raises:
            ValueError: If seed is negative.
            TypeError: If aggregation is not an Aggregation enum.
        """
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(aggregation, Aggregation):
            valid_methods = ", ".join([f"Aggregation.{a.name}" for a in Aggregation])
            raise TypeError(
                f"aggregation must be an Aggregation enum, "
                f"got {type(aggregation).__name__}. "
                f"Valid options: {valid_methods}. "
                f"Example: WeightedConformalDetector(detector=model,"
                f" strategy=strategy, aggregation=Aggregation.MEDIAN)"
            )

        self.detector: BaseDetector = _set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        self._detector_set: list[BaseDetector] = []
        self._calibration_set: list[float] = []
        self._calibration_samples: np.ndarray = np.array([])  # Initialize as empty

    @_ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector and prepares for conformal prediction.

        This method uses the provided strategy to fit the underlying detector(s)
        and generate a set of calibration scores. It also identifies and stores
        the data samples used for calibration. The `weighted` flag is passed
        as ``True`` to the strategy's `fit_calibrate` method, signaling that
        calibration sample identification is required.

        Args:
            x (pd.DataFrame | np.ndarray): The input data used for
                training/fitting the detector(s) and for calibration. The
                `@_ensure_numpy_array` decorator converts `x` to a
                ``numpy.ndarray`` internally.
            iteration_callback (callable | None): Optional callback function
                for strategies that support iteration tracking. Defaults to None.
        """
        self._detector_set, self._calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=True,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )
        if (
            self.strategy.calibration_ids is not None
            and len(self.strategy.calibration_ids) > 0
        ):
            self._calibration_samples = x[self.strategy.calibration_ids]
        else:
            # Handle case where calibration_ids might be empty or None
            # This might happen if the strategy doesn't yield IDs or x is too small
            self._calibration_samples = np.array([])

    @_ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate weighted anomaly estimates (p-values or raw scores) for new data.

        For each test instance in `x`:
        1. Anomaly scores are obtained from each detector in `detector_set`.
        2. These scores are aggregated using the method specified in `self.aggregation`.
        3. Importance weights are computed for calibration and test instances
           to account for covariate shift, using `_compute_weights`.
        4. Based on the `raw` parameter, either returns the aggregated scores
           or weighted p-values calculated using the aggregated scores,
           calibration scores, and computed weights.

        Args:
            x (pd.DataFrame | np.ndarray): The input data for which
                anomaly estimates are to be generated. The `@_ensure_numpy_array`
                decorator converts `x` to a ``numpy.ndarray`` internally.
            raw (bool, optional): Whether to return raw anomaly scores or
                weighted p-values. Defaults to False.
                * If True: Returns the aggregated anomaly scores from the
                  detector set for each data point.
                * If False: Returns the weighted p-values for each data point,
                  accounting for covariate shift between calibration and test data.

        Returns:
            numpy.ndarray: An array containing the anomaly estimates. The content of the
            array depends on the `raw` argument:
            - If raw=True, an array of anomaly scores (float).
            - If raw=False, an array of weighted p-values (float).
        """
        logger = get_logger("estimation.weighted_conformal")
        iterable = (
            tqdm(
                self._detector_set,
                total=len(self._detector_set),
                desc=f"Aggregating {len(self._detector_set)} models",
            )
            if logger.isEnabledFor(logging.DEBUG)
            else self._detector_set
        )
        scores_list = [model.decision_function(x) for model in iterable]

        w_cal, w_x = self._compute_weights(x)
        estimates = aggregate(self.aggregation, np.array(scores_list))

        return (
            estimates
            if raw
            else calculate_weighted_p_val(
                np.array(estimates),
                np.array(self._calibration_set),
                np.array(w_x),
                np.array(w_cal),
            )
        )

    def _compute_weights(
        self, test_instances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute importance weights for calibration and test instances.

        This method trains a logistic regression classifier to distinguish
        between samples from the calibration distribution and samples from the
        test distribution. The probabilities from this classifier are used
        to estimate the density ratio w(z) = p_test(z) / p_calib(z).

        The weights are clipped to a predefined range [0.35, 45] to prevent
        extreme values.

        Args:
            test_instances (numpy.ndarray): The test data instances for which
                weights need to be computed.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                A tuple containing:
                * clipped_w_calib: Clipped weights for calibration samples.
                * clipped_w_tests: Clipped weights for test instances.
        """
        if self._calibration_samples.shape[0] == 0:
            raise ValueError(
                "Calibration samples are empty. Weights cannot be computed. "
                "Ensure fit() was called and strategy provided calibration_ids."
            )

        calib_labeled = np.hstack(
            (
                self._calibration_samples,
                np.zeros((self._calibration_samples.shape[0], 1)),
            )
        )
        tests_labeled = np.hstack(
            (test_instances, np.ones((test_instances.shape[0], 1)))
        )

        joint_labeled = np.vstack((calib_labeled, tests_labeled))
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(joint_labeled)

        x_joint = joint_labeled[:, :-1]
        y_joint = joint_labeled[:, -1]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1_000,
                random_state=self.seed,
                verbose=0,
                class_weight="balanced",
            ),
        )
        model.fit(x_joint, y_joint)

        calib_prob = model.predict_proba(self._calibration_samples)
        tests_prob = model.predict_proba(test_instances)

        # Density ratio w(z) = p_test(z) / p_calib(z)
        # p_calib(z) = P(label=0 | z) ; p_test(z) = P(label=1 | z)
        # For calibration samples, weight is P(label=1 | z_calib) / P(label=0 | z_calib)
        # For test samples, weight is P(label=1 | z_test) / P(label=0 | z_test)
        # These are likelihood ratios p(z | test) / p(z | calib)
        w_calib = calib_prob[:, 1] / (
            calib_prob[:, 0] + 1e-9
        )  # Add epsilon for stability
        w_tests = tests_prob[:, 1] / (
            tests_prob[:, 0] + 1e-9
        )  # Add epsilon for stability

        clipped_w_calib = np.clip(w_calib, 0.35, 45.0)
        clipped_w_tests = np.clip(w_tests, 0.35, 45.0)

        return clipped_w_calib, clipped_w_tests

    @property
    def detector_set(self) -> list[BaseDetector]:
        """Returns a copy of the list of trained detector models.

        Returns:
            list[BaseDetector]: Copy of trained detectors populated after fit().

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._detector_set.copy()

    @property
    def calibration_set(self) -> list[float]:
        """Returns a copy of the list of calibration scores.

        Returns:
            list[float]: Copy of calibration scores populated after fit().

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._calibration_set.copy()

    @property
    def calibration_samples(self) -> np.ndarray:
        """Returns a copy of the calibration samples used for weight computation.

        Returns:
            np.ndarray: Copy of data instances used for calibration.

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._calibration_samples.copy()

    @property
    def is_fitted(self) -> bool:
        """Returns whether the detector has been fitted.

        Returns:
            bool: True if fit() has been called and models are trained.
        """
        return len(self._detector_set) > 0 and len(self._calibration_set) > 0
