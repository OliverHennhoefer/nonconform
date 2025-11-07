import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nonconform.strategy.calibration.base import BaseStrategy
from pyod.models.base import BaseDetector


class Split(BaseStrategy):
    """Split conformal strategy for fast anomaly detection with statistical guarantees.

    Implements the classical split conformal approach by dividing training data into
    separate fitting and calibration sets. This provides the fastest conformal inference
    at the cost of using less data for calibration compared to other strategies.

    Example:
        ```python
        from nonconform.strategy import Split

        # Use 20% of data for calibration
        strategy = Split(n_calib=0.2)

        # Use exactly 1000 samples for calibration
        strategy = Split(n_calib=1000)
        ```

    Attributes:
        _calib_size: Size or proportion of data used for calibration.
        _calibration_ids: Indices of calibration samples (for weighted conformal).
    """

    def __init__(self, n_calib: float | int = 0.1) -> None:
        """Initialize the Split strategy.

        Args:
            n_calib (float | int): The size or proportion
                of the dataset to use for the calibration set. If a float,
                it must be between 0.0 and 1.0 (exclusive of 0.0 and 1.0
                in practice for `train_test_split`). If an int, it's the
                absolute number of samples. Defaults to ``0.1`` (10%).
        """
        super().__init__()  # `plus` is not relevant for a single split
        self._calib_size: float | int = n_calib
        self._calibration_ids: list[int] | None = None

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        weighted: bool = False,
        seed: int | None = None,
        iteration_callback=None,
    ) -> tuple[list[BaseDetector], np.ndarray]:
        """Fits a detector and generates calibration scores using a data split.

        The input data `x` is split into a training set and a calibration
        set according to `_calib_size`. The provided `detector` is trained
        on the training set. Non-conformity scores are then computed using
        the trained detector on the calibration set.

        If `weighted` is ``True``, the indices of the calibration samples
        are stored in `_calibration_ids`. Otherwise, `_calibration_ids`
        remains ``None``.

        Args:
            x (pd.DataFrame | np.ndarray): The input data.
            detector (BaseDetector): The PyOD base detector instance to train.
                This instance is modified in place by fitting.
            weighted (bool, optional): If ``True``, the indices of the
                calibration samples are stored. Defaults to ``False``.
            seed (int | None, optional): Random seed for reproducibility of the
                train-test split. Defaults to None.
            iteration_callback (callable, optional): Not used in Split strategy.
                Defaults to None.

        Returns:
            tuple[list[BaseDetector], np.ndarray]: A tuple containing:
                * A list containing the single trained PyOD detector instance.
                * An array of calibration scores from the calibration set.
        """
        x_id = np.arange(len(x))
        train_id, calib_id = train_test_split(
            x_id, test_size=self._calib_size, shuffle=True, random_state=seed
        )

        detector.fit(x[train_id])
        calibration_set = detector.decision_function(x[calib_id])

        if weighted:
            self._calibration_ids = calib_id.tolist()  # Ensure it's a list
        else:
            self._calibration_ids = None
        return [detector], calibration_set  # Return numpy array directly

    @property
    def calibration_ids(self) -> list[int] | None:
        """Returns a copy of indices from `x` used for the calibration set.

        This property provides the list of indices corresponding to the samples
        that were allocated to the calibration set during the `fit_calibrate`
        method. It will be ``None`` if `fit_calibrate` was called with
        `weighted=False` or if `fit_calibrate` has not yet been called.

        Returns:
            list[int] | None: A copy of integer indices, or ``None``.

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return (
            self._calibration_ids.copy() if self._calibration_ids is not None else None
        )

    @property
    def calib_size(self) -> float | int:
        """Returns the calibration size or proportion.

        Returns:
            float | int: The calibration size as specified during initialization.
                If float (0.0-1.0), represents proportion of data.
                If int, represents absolute number of samples.
        """
        return self._calib_size
