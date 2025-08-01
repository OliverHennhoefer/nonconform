import abc

import numpy as np
import pandas as pd

from pyod.models.base import BaseDetector


class BaseStrategy(abc.ABC):
    """Abstract base class for anomaly detection calibration strategies.

    This class provides a common interface for various calibration strategies
    applied to anomaly detectors. Subclasses must implement the core
    calibration logic and define how calibration data is identified and used.

    Attributes
    ----------
        _plus (bool): A flag, typically set during initialization, that may
            influence calibration behavior in subclasses (e.g., by applying
            an adjustment).
    """

    def __init__(self, plus: bool = False):
        """Initialize the base calibration strategy.

        Args:
            plus (bool, optional): A flag that can be used by subclasses to
                modify their calibration behavior. Defaults to ``False``.
        """
        self._plus: bool = plus
        self._calibration_ids: list[int]

    @abc.abstractmethod
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None,
        weighted: bool = False,
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fits the detector and performs calibration.

        This abstract method must be implemented by subclasses to define the
        specific procedure for fitting the anomaly detector (if necessary)
        and then calibrating it using data derived from `x`. Calibration often
        involves determining thresholds or adjusting scores.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data, which
                may be used for both fitting the detector and deriving
                calibration data.
            detector (BaseDetector): The PyOD anomaly detection model to be
                fitted and/or calibrated.
            weighted (Optional[bool]): A flag indicating whether a weighted
                approach should be used during calibration, if applicable to
                the subclass implementation.
            seed (Optional[int]): A random seed for ensuring reproducibility
                in stochastic parts of the fitting or calibration process.

        Raises
        ------
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "The fit_calibrate() method must be implemented by subclasses."
        )

    @property
    @abc.abstractmethod
    def calibration_ids(self) -> list[int]:
        """Provides the indices of the data points used for calibration.

        This abstract property must be implemented by subclasses. It should
        return a list of integer indices identifying which samples from the
        original input data (provided to `fit_calibrate`) were selected or
        designated as the calibration set.

        Returns
        -------
            List[int]: A list of integer indices for the calibration data.

        Raises
        ------
            NotImplementedError: If the subclass does not implement this
                property.
        """
        pass
