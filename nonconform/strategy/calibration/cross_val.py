import logging
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from nonconform.strategy.calibration.base import BaseStrategy
from nonconform.utils.func.logger import get_logger
from nonconform.utils.func.params import _set_params
from pyod.models.base import BaseDetector


class CrossValidation(BaseStrategy):
    """Implements k-fold cross-validation for conformal anomaly detection.

    This strategy splits the data into k folds and uses each fold as a calibration
    set while training on the remaining folds. This approach provides more robust
    calibration scores by utilizing all available data. The strategy can operate
    in two modes:
    1. Standard mode: Uses a single model trained on all data for prediction
    2. Plus mode: Uses an ensemble of k models, each trained on k-1 folds

    Attributes:
        _k (int): Number of folds for cross-validation
        _plus (bool): Whether to use the plus variant (ensemble of models)
        _detector_list (list[BaseDetector]): List of trained detectors
        _calibration_set (list[float]): List of calibration scores
        _calibration_ids (list[int]): Indices of samples used for calibration
    """

    def __init__(self, k: int, plus: bool = True):
        """Initialize the CrossValidation strategy.

        Args:
            k (int): The number of folds for cross-validation. Must be at
                least 2. Higher values provide more robust calibration but
                increase computational cost.
            plus (bool, optional): If ``True``, appends each fold-trained model
                to `_detector_list`, creating an ensemble. If ``False``,
                `_detector_list` will contain one model trained on all data
                after calibration scores are collected. The plus variant
                maintains statistical validity and is strongly recommended.
                Defaults to ``True``.
        """
        super().__init__(plus)
        self._k: int = k
        self._plus: bool = plus

        # Warn if plus=False to alert about potential validity issues
        if not plus:
            from nonconform.utils.func.logger import get_logger

            logger = get_logger("strategy.cross_val")
            logger.warning(
                "Setting plus=False may compromise conformal validity. "
                "The plus variant (plus=True) is recommended "
                "for statistical guarantees."
            )

        self._detector_list: list[BaseDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_ids: list[int] = []

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: BaseDetector,
        seed: int | None = None,
        weighted: bool = False,
        iteration_callback=None,
    ) -> tuple[list[BaseDetector], np.ndarray]:
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
            x (pd.DataFrame | np.ndarray): Input data matrix of shape
                (n_samples, n_features).
            detector (BaseDetector): The base anomaly detector to be used.
            weighted (bool, optional): Whether to use weighted calibration.
                Currently not implemented for cross-validation. Defaults to False.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
            iteration_callback (callable, optional): Not used in CrossValidation
                strategy.
                Defaults to None.

        Returns:
            tuple[list[BaseDetector], list[float]]: A tuple containing:
                * List of trained detectors (either k models in plus mode or
                  a single model in standard mode)
                * Array of calibration scores from all folds

        Raises:
            ValueError: If k is less than 2 or if the data size is too small
                for the specified number of folds.
        """
        # Reset state so repeated fit_calibrate calls do not accumulate models/indices
        self._detector_list.clear()
        self._calibration_ids = []

        _detector = detector
        n_samples = len(x)

        # Validate k before creating KFold
        if self._k < 2:
            exc = ValueError(
                f"k must be at least 2 for k-fold cross-validation, got {self._k}"
            )
            exc.add_note(f"Received k={self._k}, which is invalid.")
            exc.add_note(
                "Cross-validation requires at least one split"
                " for training and one for calibration."
            )
            exc.add_note(
                f"With {n_samples} samples, consider k=min(10,"
                f" {n_samples // 10}) for balanced folds."
            )
            raise exc

        if n_samples < self._k:
            exc = ValueError(
                f"Not enough samples ({n_samples}) for "
                f"k-fold cross-validation with k={self._k}"
            )
            exc.add_note(
                f"Each fold needs at least 1 sample, but {n_samples} < {self._k}."
            )
            exc.add_note(
                f"Either increase your dataset size or reduce k to at most {n_samples}."
            )
            raise exc

        # Pre-allocate calibration array for efficiency
        self._calibration_set = np.empty(n_samples, dtype=np.float64)
        calibration_offset = 0

        folds = KFold(
            n_splits=self._k,
            shuffle=True,
            random_state=seed,
        )

        last_iteration_index = 0
        logger = get_logger("strategy.cross_val")
        fold_iterator = (
            tqdm(
                folds.split(x),
                total=self._k,
                desc="Calibration",
            )
            if logger.isEnabledFor(logging.INFO)
            else folds.split(x)
        )
        for i, (train_idx, calib_idx) in enumerate(fold_iterator):
            last_iteration_index = i
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(_detector)
            model = _set_params(model, seed=seed, random_iteration=True, iteration=i)
            model.fit(x[train_idx])

            if self._plus:
                self._detector_list.append(deepcopy(model))

            # Store calibration scores efficiently using pre-allocated array
            fold_scores = model.decision_function(x[calib_idx])
            n_fold_samples = len(fold_scores)
            end_idx = calibration_offset + n_fold_samples
            self._calibration_set[calibration_offset:end_idx] = fold_scores
            calibration_offset += n_fold_samples

        if not self._plus:
            model = copy(_detector)
            model = _set_params(
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
        """Returns a copy of the list of indices from `x` used for calibration.

        In k-fold cross-validation, every sample in the input data `x` is
        used exactly once as part of a calibration set (when its fold is
        the hold-out set). This property returns a list of all these indices,
        typically covering all indices from 0 to len(x)-1, but ordered by
        fold processing.

        Returns:
            list[int]: A copy of integer indices.

        Note:
            Returns a defensive copy to prevent external modification of internal state.
        """
        return self._calibration_ids.copy()

    @property
    def k(self) -> int:
        """Returns the number of folds for cross-validation.

        Returns:
            int: Number of folds specified during initialization.
        """
        return self._k

    @property
    def plus(self) -> bool:
        """Returns whether the plus variant is enabled.

        Returns:
            bool: True if using ensemble mode, False if using single model.
        """
        return self._plus
