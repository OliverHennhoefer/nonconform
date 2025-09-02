import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.estimation.base import BaseConformalDetector
from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.decorator import _ensure_numpy_array
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import _set_params
from nonconform.utils.stat.aggregation import aggregate
from nonconform.utils.stat.statistical import calculate_p_val
from pyod.models.base import BaseDetector as PyODBaseDetector  # Alias for clarity


class StandardConformalDetector(BaseConformalDetector):
    """Standard conformal anomaly detector with statistical guarantees.

    Provides distribution-free anomaly detection with valid p-values and False Discovery
    Rate (FDR) control by wrapping any PyOD detector with conformal inference.
    The detector uses calibration data to convert anomaly scores into
    statistically valid p-values.

    Example:
        Basic usage with Isolation Forest and Split calibration:

        ```python
        from pyod.models.iforest import IForest
        from nonconform.estimation import StandardConformalDetector
        from nonconform.strategy import Split

        # Create conformal detector
        detector = StandardConformalDetector(
            detector=IForest(), strategy=Split(n_calib=0.2), seed=42
        )

        # Fit on normal training data
        detector.fit(X_train)

        # Get p-values for test data
        p_values = detector.predict(X_test)

        # Apply FDR control
        from scipy.stats import false_discovery_control

        decisions = false_discovery_control(p_values, method="bh") <= 0.1
        ```

    Attributes:
        detector: The underlying PyOD anomaly detection model.
        strategy: The calibration strategy for computing p-values.
        aggregation: Method for combining scores from multiple models.
        seed: Random seed for reproducible results.
        detector_set: List of trained detector models (populated after fit).
        calibration_set: Calibration scores for p-value computation (populated by fit).
        is_fitted: Whether the detector has been fitted.
    """

    def __init__(
        self,
        detector: PyODBaseDetector,
        strategy: BaseStrategy,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
    ):
        """Initialize the ConformalDetector.

        Args:
            detector (PyODBaseDetector): The base anomaly detection model to be
                used (e.g., an instance of a PyOD detector).
            strategy (BaseStrategy): The conformal strategy to apply for fitting
                and calibration.
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
                f"Example: StandardConformalDetector(detector=model, "
                f"strategy=strategy, aggregation=Aggregation.MEDIAN)"
            )

        self.detector: PyODBaseDetector = _set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        self._detector_set: list[PyODBaseDetector] = []
        self._calibration_set: list[float] = []

    @_ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector model(s) and computes calibration scores.

        This method uses the specified strategy to train the base detector(s)
        on parts of the provided data and then calculates non-conformity
        scores on other parts (calibration set) to establish a baseline for
        typical behavior. The resulting trained models and calibration scores
        are stored in `self._detector_set` and `self._calibration_set`.

        Args:
            x (pd.DataFrame | np.ndarray): The dataset used for
                fitting the model(s) and determining calibration scores.
                The strategy will dictate how this data is split or used.
            iteration_callback (callable | None): Optional callback function
                for strategies that support iteration tracking (e.g., Bootstrap).
                Called after each iteration with (iteration, scores). Defaults to None.
        """
        self._detector_set, self._calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=False,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )

    @_ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates (p-values or raw scores) for new data.

        Based on the fitted models and calibration scores, this method evaluates
        new data points. It can return either raw anomaly scores or p-values
        indicating how unusual each point is.

        Args:
            x (pd.DataFrame | np.ndarray): The new data instances
                for which to generate anomaly estimates.
            raw (bool, optional): Whether to return raw anomaly scores or
                p-values. Defaults to False.
                * If True: Returns the aggregated anomaly scores (non-conformity
                  estimates) from the detector set for each data point.
                * If False: Returns the p-values for each data point based on
                  the calibration set.

        Returns:
            np.ndarray: An array containing the anomaly estimates. The content of the
            array depends on the `raw` argument:
            - If raw=True, an array of anomaly scores (float).
            - If raw=False, an array of p-values (float).
        """
        logger = get_logger("estimation.standard_conformal")
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self._detector_set,
                total=len(self._detector_set),
                desc=f"Aggregating {len(self._detector_set)} models",
                disable=not logger.isEnabledFor(logging.DEBUG),
            )
        ]

        estimates = aggregate(method=self.aggregation, scores=scores_list)
        return (
            estimates
            if raw
            else calculate_p_val(
                scores=estimates, calibration_set=self._calibration_set
            )
        )

    @property
    def detector_set(self) -> list[PyODBaseDetector]:
        """Returns the list of trained detector models.

        Returns:
            list[PyODBaseDetector]: List of trained detectors populated after fit().
        """
        return self._detector_set

    @property
    def calibration_set(self) -> list[float]:
        """Returns the list of calibration scores.

        Returns:
            list[float]: List of calibration scores populated after fit().
        """
        return self._calibration_set

    @property
    def is_fitted(self) -> bool:
        """Returns whether the detector has been fitted.

        Returns:
            bool: True if fit() has been called and models are trained.
        """
        return len(self._detector_set) > 0 and len(self._calibration_set) > 0
