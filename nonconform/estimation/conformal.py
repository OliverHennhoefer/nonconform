import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.estimation.base import BaseConformalDetector
from nonconform.estimation.weight import BaseWeightEstimator, IdentityWeightEstimator
from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.decorator import _ensure_numpy_array
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import _set_params
from nonconform.utils.stat.aggregation import aggregate
from nonconform.utils.stat.statistical import calculate_p_val, calculate_weighted_p_val
from pyod.models.base import BaseDetector as PyODBaseDetector


class ConformalDetector(BaseConformalDetector):
    """Unified conformal anomaly detector with optional covariate shift handling.

    Provides distribution-free anomaly detection with valid p-values and False Discovery
    Rate (FDR) control by wrapping any PyOD detector with conformal inference.
    Optionally handles covariate shift through importance weighting when a weight
    estimator is specified.

    When no weight estimator is provided (standard conformal prediction):
    - Uses classical conformal inference for exchangeable data
    - Provides optimal performance and memory usage
    - Suitable when training and test data come from the same distribution

    When a weight estimator is provided (weighted conformal prediction):
    - Handles distribution shift between calibration and test data
    - Estimates importance weights to maintain statistical validity
    - Slightly higher computational cost but robust to covariate shift

    Examples:
        Standard conformal prediction (no distribution shift):

        ```python
        from pyod.models.iforest import IForest
        from nonconform.estimation import ConformalDetector
        from nonconform.strategy import Split

        # Create standard conformal detector
        detector = ConformalDetector(
            detector=IForest(), strategy=Split(n_calib=0.2), seed=42
        )

        # Fit on normal training data
        detector.fit(X_train)

        # Get p-values for test data
        p_values = detector.predict(X_test)
        ```

        Weighted conformal prediction (with distribution shift):

        ```python
        from nonconform.estimation.weight import LogisticWeightEstimator

        # Create weighted conformal detector
        detector = ConformalDetector(
            detector=IForest(),
            strategy=Split(n_calib=0.2),
            weight_estimator=LogisticWeightEstimator(seed=42),
            seed=42,
        )

        # Same usage as standard conformal
        detector.fit(X_train)
        p_values = detector.predict(X_test)
        ```

    Attributes:
        detector: The underlying PyOD anomaly detection model.
        strategy: The calibration strategy for computing p-values.
        weight_estimator: Optional weight estimator for handling covariate shift.
        aggregation: Method for combining scores from multiple models.
        seed: Random seed for reproducible results.
        detector_set: List of trained detector models (populated after fit).
        calibration_set: Calibration scores for p-value computation (populated by fit).
        is_fitted: Whether the detector has been fitted.
        calibration_samples: Data instances used for calibration (only for
            weighted mode).
    """

    def __init__(
        self,
        detector: PyODBaseDetector,
        strategy: BaseStrategy,
        weight_estimator: BaseWeightEstimator | None = None,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
    ):
        """Initialize the ConformalDetector.

        Args:
            detector (PyODBaseDetector): The base anomaly detection model to be
                used (e.g., an instance of a PyOD detector).
            strategy (BaseStrategy): The conformal strategy to apply for fitting
                and calibration.
            weight_estimator (BaseWeightEstimator | None, optional): Weight estimator
                for handling covariate shift. If None, uses standard conformal
                prediction (equivalent to IdentityWeightEstimator). Defaults to None.
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
                f"Example: ConformalDetector(detector=model, "
                f"strategy=strategy, aggregation=Aggregation.MEDIAN)"
            )

        self.detector: PyODBaseDetector = _set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.weight_estimator: BaseWeightEstimator | None = weight_estimator
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        # Determine if we're in weighted mode
        self._is_weighted_mode = weight_estimator is not None and not isinstance(
            weight_estimator, IdentityWeightEstimator
        )

        self._detector_set: list[PyODBaseDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_samples: np.ndarray = np.array([])
        # Only used in weighted mode

    @_ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector model(s) and computes calibration scores.

        This method uses the specified strategy to train the base detector(s)
        on parts of the provided data and then calculates non-conformity
        scores on other parts (calibration set) to establish a baseline for
        typical behavior. The resulting trained models and calibration scores
        are stored in `self._detector_set` and `self._calibration_set`.

        For weighted conformal prediction, calibration samples are also stored
        for weight computation during prediction.

        Args:
            x (pd.DataFrame | np.ndarray): The dataset used for
                fitting the model(s) and determining calibration scores.
                The strategy will dictate how this data is split or used.
            iteration_callback (callable | None): Optional callback function
                for strategies that support iteration tracking (e.g., Bootstrap).
                Called after each iteration with (iteration, scores). Defaults to None.
        """
        # Pass weighted flag only when using non-identity weight estimator
        self._detector_set, self._calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=self._is_weighted_mode,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )

        # Store calibration samples only for weighted mode
        if self._is_weighted_mode:
            if (
                self.strategy.calibration_ids is not None
                and len(self.strategy.calibration_ids) > 0
            ):
                self._calibration_samples = x[self.strategy.calibration_ids]
            else:
                # Handle case where calibration_ids might be empty or None
                self._calibration_samples = np.array([])

    @_ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates (p-values or raw scores) for new data.

        Based on the fitted models and calibration scores, this method evaluates
        new data points. For standard conformal prediction, returns p-values based
        on the calibration distribution. For weighted conformal prediction,
        incorporates importance weights to handle covariate shift.

        Args:
            x (pd.DataFrame | np.ndarray): The new data instances
                for which to generate anomaly estimates.
            raw (bool, optional): Whether to return raw anomaly scores or
                p-values. Defaults to False.
                * If True: Returns the aggregated anomaly scores (non-conformity
                  estimates) from the detector set for each data point.
                * If False: Returns the p-values for each data point based on
                  the calibration set, optionally weighted for distribution shift.

        Returns:
            np.ndarray: An array containing the anomaly estimates. The content of the
            array depends on the `raw` argument:
            - If raw=True, an array of anomaly scores (float).
            - If raw=False, an array of p-values (float).
        """
        logger = get_logger("estimation.conformal")
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

        estimates = aggregate(method=self.aggregation, scores=scores_list)

        if raw:
            return estimates

        # Choose p-value calculation method based on weight estimator
        if self._is_weighted_mode and self.weight_estimator is not None:
            # Weighted p-value calculation
            self.weight_estimator.fit(self._calibration_samples, x)
            w_cal, w_x = self.weight_estimator.get_weights()
            return calculate_weighted_p_val(
                np.array(estimates),
                self._calibration_set,
                np.array(w_x),
                np.array(w_cal),
            )
        else:
            # Standard p-value calculation (faster path)
            return calculate_p_val(
                scores=estimates, calibration_set=self._calibration_set
            )

    @property
    def detector_set(self) -> list[PyODBaseDetector]:
        """Returns a copy of the list of trained detector models.

        Returns:
            list[PyODBaseDetector]: Copy of trained detectors populated after fit().

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._detector_set.copy()

    @property
    def calibration_set(self) -> np.ndarray:
        """Returns a copy of the calibration scores.

        Returns:
            numpy.ndarray: Copy of calibration scores populated after fit().

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._calibration_set.copy()

    @property
    def calibration_samples(self) -> np.ndarray:
        """Returns a copy of the calibration samples used for weight computation.

        Only available when using weighted conformal prediction
        (non-identity weight estimator). For standard conformal prediction,
        returns an empty array.

        Returns:
            np.ndarray: Copy of data instances used for calibration, or empty array
                       if using standard conformal prediction.

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
