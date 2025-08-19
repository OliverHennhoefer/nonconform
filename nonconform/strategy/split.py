import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nonconform.strategy.base import BaseStrategy
from pyod.models.base import BaseDetector


class Split(BaseStrategy):
    """Split-based conformal anomaly detection strategy.

    This strategy implements conformal prediction by splitting the input data
    into a single training set and a single calibration set. An anomaly
    detector is trained on the training set, and its non-conformity scores
    are then obtained from the calibration set.

    Attributes
    ----------
        _calib_size (Union[float, int]): Defines the size of the calibration
            set. If a float between 0.0 and 1.0, it represents the
            proportion of the dataset to allocate to the calibration set.
            If an integer, it represents the absolute number of samples for
            the calibration set.
        _calibration_ids (Optional[List[int]]): Indices of the samples from
            the input data `x` that formed the calibration set. This is
            populated by :meth:`fit_calibrate` only if `weighted` is ``True``
            during the call, otherwise it remains ``None``. Accessible via the
            :attr:`calibration_ids` property.
    """

    def __init__(self, n_calib: float | int = 0.1) -> None:
        """Initialize the Split strategy.

        Args:
            n_calib (Union[float, int], optional): The size or proportion
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
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fits a detector and generates calibration scores using a data split.

        The input data `x` is split into a training set and a calibration
        set according to `_calib_size`. The provided `detector` is trained
        on the training set. Non-conformity scores are then computed using
        the trained detector on the calibration set.

        If `weighted` is ``True``, the indices of the calibration samples
        are stored in `_calibration_ids`. Otherwise, `_calibration_ids`
        remains ``None``.

        Args:
            x (Union[pandas.DataFrame, numpy.ndarray]): The input data.
            detector (BaseDetector): The PyOD base detector instance to train.
                This instance is modified in place by fitting.
            weighted (bool, optional): If ``True``, the indices of the
                calibration samples are stored. Defaults to ``False``.
            seed (int | None, optional): Random seed for reproducibility of the
                train-test split. Defaults to None.
            iteration_callback (callable, optional): Not used in Split strategy.
                Defaults to None.

        Returns
        -------
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * A list containing the single trained PyOD detector instance.
                * A list of calibration scores from the calibration set.
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
        return [detector], calibration_set.tolist()  # Ensure list return

    @property
    def calibration_ids(self) -> list[int] | None:
        """Returns indices from `x` used for the calibration set.

        This property provides the list of indices corresponding to the samples
        that were allocated to the calibration set during the `fit_calibrate`
        method. It will be ``None`` if `fit_calibrate` was called with
        `weighted=False` or if `fit_calibrate` has not yet been called.

        Returns
        -------
            list[int] | None: A list of integer indices, or ``None``.
        """
        return self._calibration_ids
