from copy import copy, deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from nonconform.strategy.base import BaseStrategy
from nonconform.utils.func.params import set_params
from pyod.models.base import BaseDetector


class CrossValidation(BaseStrategy):
    """Implements k-fold cross-validation for conformal anomaly detection.

    This strategy splits the data into k folds and uses each fold as a calibration
    set while training on the remaining folds. This approach provides more robust
    calibration scores by utilizing all available data. The strategy can operate
    in two modes:
    1. Standard mode: Uses a single model trained on all data for prediction
    2. Plus mode: Uses an ensemble of k models, each trained on k-1 folds

    Attributes
    ----------
        _k (int): Number of folds for cross-validation
        _plus (bool): Whether to use the plus variant (ensemble of models)
        _detector_list (list[BaseDetector]): List of trained detectors
        _calibration_set (list[float]): List of calibration scores
        _calibration_ids (list[int]): Indices of samples used for calibration
    """

    def __init__(self, k: int, plus: bool = False):
        """Initialize the CrossValidation strategy.

        Args:
            k (int): The number of folds for cross-validation. Must be at
                least 2. Higher values provide more robust calibration but
                increase computational cost.
            plus (bool, optional): If ``True``, appends each fold-trained model
                to `_detector_list`, creating an ensemble. If ``False``,
                `_detector_list` will contain one model trained on all data
                after calibration scores are collected. The plus variant
                typically provides better performance but requires more memory.
                Defaults to ``False``.
        """
        super().__init__(plus)
        self._k: int = k
        self._plus: bool = plus

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: list[float] = []
        self._calibration_ids: list[int] = []

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int = 1,
        weighted: bool = False,
    ) -> tuple[list[BaseDetector], list[float]]:
        """Fit and calibrate the detector using k-fold cross-validation.

        This method implements the cross-validation strategy by:
        1. Splitting the data into k folds
        2. For each fold:
           - Train the detector on k-1 folds
           - Use the remaining fold for calibration
           - Store calibration scores and optionally the trained model
        3. If not in plus mode, train a final model on all data

        The method ensures that each sample is used exactly once for calibration,
        providing a more robust estimate of the calibration scores.

        Args:
            x (Union[pd.DataFrame, np.ndarray]): Input data matrix of shape
                (n_samples, n_features).
            detector (BaseDetector): The base anomaly detector to be used.
            weighted (bool, optional): Whether to use weighted calibration.
                Currently not implemented for cross-validation. Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.

        Returns
        -------
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List of trained detectors (either k models in plus mode or
                  a single model in standard mode)
                * List of calibration scores from all folds

        Raises
        ------
            ValueError: If k is less than 2 or if the data size is too small
                for the specified number of folds.
        """
        _detector = detector

        folds = KFold(
            n_splits=self._k,
            shuffle=True,
            random_state=seed,
        )

        last_iteration_index = 0
        for i, (train_idx, calib_idx) in enumerate(
            tqdm(folds.split(x), total=self._k, desc="Training", disable=False)
        ):
            last_iteration_index = i
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(_detector)
            model = set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx])

            if self._plus:
                self._detector_list.append(deepcopy(model))
            self._calibration_set.extend(model.decision_function(x[calib_idx]))

        if not self._plus:
            model = copy(_detector)
            model = set_params(
                model,
                seed=seed,
                random_iteration=True,
                iteration=(last_iteration_index + 1),
            )
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self) -> list[int]:
        """Returns the list of indices from `x` used for calibration.

        In k-fold cross-validation, every sample in the input data `x` is
        used exactly once as part of a calibration set (when its fold is
        the hold-out set). This property returns a list of all these indices,
        typically covering all indices from 0 to len(x)-1, but ordered by
        fold processing.

        Returns
        -------
            list[int]: A list of integer indices.
        """
        return self._calibration_ids
