import logging
from collections.abc import Callable
from typing import Literal

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
from nonconform.utils.stat.extreme import fit_gpd, select_threshold
from nonconform.utils.stat.statistical import calculate_evt_p_val, calculate_p_val
from pyod.models.base import BaseDetector as PyODBaseDetector


class ExtremeConformalDetector(BaseConformalDetector):
    """Conformal anomaly detector with Extreme Value Theory enhancement.

    This detector extends the standard conformal prediction framework by
    modeling extreme calibration scores using a Generalized Pareto Distribution.
    This provides better calibration for detecting extreme anomalies while
    maintaining the standard empirical approach for normal values.

    The detector automatically fits a GPD to calibration scores exceeding
    a configurable threshold and uses a hybrid p-value calculation that
    combines empirical and parametric approaches.

    Attributes
    ----------
        detector (PyODBaseDetector): The underlying anomaly detection model.
        strategy (BaseStrategy): The strategy used for fitting and calibration.
        aggregation (Aggregation): Method used for aggregating scores from
            multiple detector models.
        seed (int): Random seed for reproducibility in stochastic processes.
        evt_threshold_method (Literal): Method for selecting EVT threshold.
        evt_threshold_value (Union[float, Callable]): Parameter for threshold method.
        evt_min_tail_size (int): Minimum number of exceedances required for GPD fitting.
        detector_set (List[PyODBaseDetector]): Trained detector models.
        calibration_set (List[float]): Calibration scores.
        evt_threshold (float): Threshold separating bulk and tail distributions.
        gpd_params (Tuple[float, float, float]): Fitted GPD parameters
            (shape, loc, scale).
    """

    def __init__(
        self,
        detector: PyODBaseDetector,
        strategy: BaseStrategy,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
        evt_threshold_method: Literal[
            "percentile", "top_k", "mean_excess", "custom"
        ] = "percentile",
        evt_threshold_value: float | Callable[[np.ndarray], float] = 0.95,
        evt_min_tail_size: int = 10,
    ):
        """Initialize the EVTConformalDetector.

        Args:
            detector (PyODBaseDetector): The base anomaly detection model.
            strategy (BaseStrategy): The conformal strategy for calibration.
            aggregation (Aggregation, optional): Method used for aggregating
                scores from multiple detector models. Defaults to Aggregation.MEDIAN.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            evt_threshold_method (Literal, optional): Method for selecting EVT
                threshold. Defaults to "percentile".
            evt_threshold_value (Union[float, Callable], optional): Parameter for
                threshold method. Defaults to 0.95.
            evt_min_tail_size (int, optional): Minimum number of exceedances
                required for GPD fitting. Defaults to 10.
        """
        # Parameter validation (moved from StandardConformalDetector)
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(aggregation, Aggregation):
            raise TypeError(
                f"aggregation must be an Aggregation enum, got {type(aggregation)}"
            )

        # Initialize attributes (moved from StandardConformalDetector)
        self.detector: PyODBaseDetector = set_params(detector, seed)
        self.strategy: BaseStrategy = strategy
        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed

        self.detector_set: list[PyODBaseDetector] = []
        self.calibration_set: list[float] = []

        # EVT-specific parameters
        self.evt_threshold_method: Literal[
            "percentile", "top_k", "mean_excess", "custom"
        ] = evt_threshold_method
        self.evt_threshold_value: float | Callable[[np.ndarray], float] = (
            evt_threshold_value
        )
        self.evt_min_tail_size: int = evt_min_tail_size

        # EVT-specific attributes
        self.evt_threshold: float | None = None
        self.gpd_params: tuple[float, float, float] | None = None

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray, iteration_callback=None) -> None:
        """Fits the detector and prepares EVT-enhanced calibration.

        This method extends the parent's fit method by additionally:
        1. Determining the threshold for extreme values
        2. Fitting a GPD to calibration scores exceeding the threshold

        The EVT parameters are only fitted if there are sufficient
        exceedances as specified by evt_min_tail_size.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Training data.
            iteration_callback (callable, optional): Optional callback function
                for strategies that support iteration tracking. Defaults to None.
        """
        # Fit using strategy (moved from StandardConformalDetector)
        self.detector_set, self.calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=False,
            seed=self.seed,
            iteration_callback=iteration_callback,
        )

        # Fit EVT components
        if len(self.calibration_set) > 0:
            calibration_array = np.array(self.calibration_set)

            # Select threshold
            self.evt_threshold = select_threshold(
                calibration_array,
                self.evt_threshold_method,
                self.evt_threshold_value,
            )

            # Fit GPD to exceedances
            exceedances = (
                calibration_array[calibration_array > self.evt_threshold]
                - self.evt_threshold
            )

            if len(exceedances) >= self.evt_min_tail_size:
                try:
                    self.gpd_params = fit_gpd(exceedances)
                except Exception as e:
                    # If GPD fitting fails, fall back to standard approach
                    logger = get_logger("estimation.extreme_conformal")
                    logger.warning(
                        "GPD fitting failed: %s. Using standard approach.", e
                    )
                    self.gpd_params = None
            else:
                logger = get_logger("estimation.extreme_conformal")
                logger.warning(
                    "Only %d exceedances, need at least %d. Using standard approach.",
                    len(exceedances),
                    self.evt_min_tail_size,
                )
                self.gpd_params = None

    @ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Predicts using EVT-enhanced conformal prediction.

        This method calculates anomaly scores and then computes p-values
        using either the standard empirical approach or the EVT-enhanced
        hybrid approach, depending on whether EVT parameters were successfully
        fitted during training.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Test data.
            raw (bool, optional): Whether to return raw anomaly scores or
                p-values. Defaults to False.

        Returns
        -------
            np.ndarray: Predictions based on the output type.
        """
        # Calculate anomaly scores
        logger = get_logger("estimation.extreme_conformal")
        scores_list = [
            model.decision_function(x)
            for model in tqdm(
                self.detector_set,
                total=len(self.detector_set),
                desc=f"Aggregating {len(self.detector_set)} models",
                disable=not logger.isEnabledFor(logging.INFO),
            )
        ]

        estimates = aggregate(method=self.aggregation, scores=scores_list)

        if raw:
            return estimates

        # Calculate p-values using EVT if available
        if self.gpd_params is not None:
            p_val = calculate_evt_p_val(
                estimates,
                self.calibration_set,
                threshold_method=self.evt_threshold_method,
                threshold_value=self.evt_threshold_value,
                min_tail_size=self.evt_min_tail_size,
                gpd_params=self.gpd_params,
                threshold=self.evt_threshold,
            )
        else:
            # Fall back to standard empirical p-values
            p_val = calculate_p_val(estimates, self.calibration_set)

        return p_val
