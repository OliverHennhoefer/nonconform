import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.estimation.base import BaseConformalDetector
from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.decorator import ensure_numpy_array
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import set_params
from nonconform.utils.stat.aggregation import aggregate
from nonconform.utils.stat.statistical import calculate_p_val
from pyod.models.base import BaseDetector as PyODBaseDetector  # Alias for clarity


class StandardConformalDetector(BaseConformalDetector):
    """Calibrates an anomaly detector using conformal prediction.

    This detector inherits from BaseConformalDetector and uses an underlying
    anomaly detection model and a specified strategy (e.g., split conformal, CV+)
    to calibrate non-conformity scores. It then uses these calibrated scores to
    generate anomaly estimates on new data, providing options for raw scores or
    p-values.

    Attributes
    ----------
        detector (PyODBaseDetector): The underlying anomaly detection model,
            initialized with a specific seed.
        strategy (BaseStrategy): The strategy used to fit and calibrate the
            detector (e.g., split conformal, cross-validation).
        aggregation (Aggregation): Method used for aggregating scores from
            multiple detector models.
        seed (int): Random seed for reproducibility in stochastic processes.
        detector_set (List[PyODBaseDetector]): A list of trained anomaly detector
            models. Populated after the `fit` method is called. Depending on
            the strategy, this might contain one or multiple models.
        calibration_set (List[float]): A list of calibration scores
            (non-conformity scores) obtained from the calibration data.
            Populated after the `fit` method is called.
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

        Raises
        ------
            ValueError: If seed is negative.
            TypeError: If aggregation is not an Aggregation enum.
        """
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(aggregation, Aggregation):
            raise TypeError(
                f"aggregation must be an Aggregation enum, got {type(aggregation)}"
            )

        self.detector: PyODBaseDetector = set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        self.detector_set: list[PyODBaseDetector] = []
        self.calibration_set: list[float] = []

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector model(s) and computes calibration scores.

        This method uses the specified strategy to train the base detector(s)
        on parts of the provided data and then calculates non-conformity
        scores on other parts (calibration set) to establish a baseline for
        typical behavior. The resulting trained models and calibration scores
        are stored in `self.detector_set` and `self.calibration_set`.

        Args:
            x (typing.Union[pd.DataFrame, np.ndarray]): The dataset used for
                fitting the model(s) and determining calibration scores.
                The strategy will dictate how this data is split or used.
            iteration_callback (callable, optional): Optional callback function
                for strategies that support iteration tracking (e.g., Bootstrap).
                Called after each iteration with (iteration, scores). Defaults to None.
        """
        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=False,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )

    @ensure_numpy_array
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
            x (typing.Union[pd.DataFrame, np.ndarray]): The new data instances
                for which to generate anomaly estimates.
            raw (bool, optional): Whether to return raw anomaly scores or
                p-values. Defaults to False.
                * If True: Returns the aggregated anomaly scores (non-conformity
                  estimates) from the detector set for each data point.
                * If False: Returns the p-values for each data point based on
                  the calibration set.

        Returns
        -------
            np.ndarray: An array containing the anomaly estimates. The content of the
            array depends on the `raw` argument:
            - If raw=True, an array of anomaly scores (float).
            - If raw=False, an array of p-values (float).
        """
        logger = get_logger("estimation.standard_conformal")
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc=f"Aggregating {len(self.detector_set)} models",
                disable=not logger.isEnabledFor(logging.DEBUG),
            )
        ]

        estimates = aggregate(method=self.aggregation, scores=scores_list)
        return (
            estimates
            if raw
            else calculate_p_val(scores=estimates, calibration_set=self.calibration_set)
        )
