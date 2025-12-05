"""Calibration strategies for conformal anomaly detection.

This module provides various calibration strategies that define how to split
data for training and calibration in conformal prediction.

Classes:
    BaseStrategy: Abstract base class for calibration strategies.
    Split: Simple train-test split strategy.
    CrossValidation: K-fold cross-validation strategy (includes Jackknife factory).
    JackknifeBootstrap: Jackknife+-after-Bootstrap (JaB+) strategy.
"""

from __future__ import annotations

import abc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from nonconform._internal import Aggregation

if TYPE_CHECKING:
    from nonconform.structures import AnomalyDetector

# Module-level loggers for performance
_crossval_logger = logging.getLogger("nonconform.resampling.crossval")
_bootstrap_logger = logging.getLogger("nonconform.resampling.bootstrap")


class BaseStrategy(abc.ABC):
    """Abstract base class for anomaly detection calibration strategies.

    This class provides a common interface for various calibration strategies
    applied to anomaly detectors. Subclasses must implement the core
    calibration logic and define how calibration data is identified and used.

    Attributes:
        _plus: A flag that may influence calibration behavior in subclasses.
    """

    def __init__(self, plus: bool = True) -> None:
        """Initialize the base calibration strategy.

        Args:
            plus: A flag that enables the "plus" variant which maintains
                statistical validity by retaining calibration models for
                inference. Strongly recommended for proper conformal guarantees.
                Defaults to True.
        """
        self._plus: bool = plus
        self._calibration_ids: list[int] = []

    @abc.abstractmethod
    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fits the detector and performs calibration.

        Args:
            x: The input data for fitting and calibration.
            detector: The anomaly detection model to be fitted and calibrated.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Whether to use weighted approach. Defaults to False.

        Returns:
            Tuple of (list of trained detectors, calibration scores array).
        """
        raise NotImplementedError(
            "The fit_calibrate() method must be implemented by subclasses."
        )

    @property
    @abc.abstractmethod
    def calibration_ids(self) -> list[int] | None:
        """Indices of data points used for calibration."""
        pass


class Split(BaseStrategy):
    """Split conformal strategy for fast anomaly detection.

    Implements the classical split conformal approach by dividing training data
    into separate fitting and calibration sets.

    Args:
        n_calib: Size or proportion of data used for calibration.
            If float, must be between 0.0 and 1.0 (proportion).
            If int, the absolute number of samples. Defaults to 0.1.

    Examples:
        ```python
        # Use 20% of data for calibration
        strategy = Split(n_calib=0.2)

        # Use exactly 1000 samples for calibration
        strategy = Split(n_calib=1000)
        ```
    """

    def __init__(self, n_calib: float | int = 0.1) -> None:
        super().__init__()
        self._calib_size: float | int = n_calib
        self._calibration_ids: list[int] | None = None

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        weighted: bool = False,
        seed: int | None = None,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fits detector and generates calibration scores using a data split.

        Args:
            x: The input data.
            detector: The detector instance to train.
            weighted: If True, stores calibration sample indices. Defaults to False.
            seed: Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple of (list with trained detector, calibration scores array).
        """
        x_id = np.arange(len(x))
        train_id, calib_id = train_test_split(
            x_id, test_size=self._calib_size, shuffle=True, random_state=seed
        )

        detector.fit(x[train_id])
        calibration_set = detector.decision_function(x[calib_id])

        if weighted:
            self._calibration_ids = calib_id.tolist()
        else:
            self._calibration_ids = None
        return [detector], calibration_set

    @property
    def calibration_ids(self) -> list[int] | None:
        """Indices of calibration samples (None if weighted=False)."""
        return (
            self._calibration_ids.copy() if self._calibration_ids is not None else None
        )

    @property
    def calib_size(self) -> float | int:
        """Returns the calibration size or proportion."""
        return self._calib_size


class CrossValidation(BaseStrategy):
    """K-fold cross-validation strategy for conformal anomaly detection.

    Splits data into k folds and uses each fold as a calibration set while
    training on the remaining folds.

    Args:
        k: Number of folds. If None, uses leave-one-out (k=n at fit time).
        plus: Whether to use ensemble mode. Strongly recommended. Defaults to True.
        shuffle: Whether to shuffle data before splitting. Defaults to True.
            Set to False for deterministic leave-one-out (Jackknife).

    Examples:
        ```python
        # 5-fold cross-validation
        strategy = CrossValidation(k=5)

        # Leave-one-out (Jackknife) via factory
        strategy = CrossValidation.jackknife()
        ```
    """

    def __init__(
        self, k: int | None = 5, plus: bool = True, shuffle: bool = True
    ) -> None:
        super().__init__(plus)
        self._k: int | None = k
        self._plus: bool = plus
        self._shuffle: bool = shuffle
        self._is_jackknife = k is None

        # Warn if plus=False
        if not plus:
            _crossval_logger.warning(
                "Setting plus=False may compromise conformal validity. "
                "The plus variant (plus=True) is recommended."
            )

        self._detector_list: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_ids: list[int] = []

    @classmethod
    def jackknife(cls, plus: bool = True) -> CrossValidation:
        """Create Leave-One-Out cross-validation (deterministic, no shuffle).

        This factory method creates a Jackknife strategy, which is a special
        case of k-fold CV where k equals n (the dataset size). Each sample is
        left out exactly once for calibration.

        Args:
            plus: Whether to use ensemble mode. Defaults to True.

        Returns:
            CrossValidation configured for leave-one-out.

        Examples:
            ```python
            strategy = CrossValidation.jackknife()
            detector_list, calib_scores = strategy.fit_calibrate(X, detector)
            ```
        """
        return cls(k=None, plus=plus, shuffle=False)

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fit and calibrate using k-fold cross-validation.

        Args:
            x: Input data matrix.
            detector: The base anomaly detector.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Whether to use weighted calibration. Defaults to False.

        Returns:
            Tuple of (list of trained detectors, calibration scores array).

        Raises:
            ValueError: If k < 2 or not enough samples for specified k.
        """
        self._detector_list.clear()
        self._calibration_ids = []

        detector_ = detector
        n_samples = len(x)

        # Determine k (for jackknife mode, k=n)
        k = n_samples if self._is_jackknife else self._k

        if k < 2:
            exc = ValueError(
                f"k must be at least 2 for k-fold cross-validation, got {k}"
            )
            exc.add_note(f"Received k={k}, which is invalid.")
            exc.add_note(
                "Cross-validation requires at least one split for training "
                "and one for calibration."
            )
            raise exc

        if n_samples < k:
            exc = ValueError(
                f"Not enough samples ({n_samples}) for "
                f"k-fold cross-validation with k={k}"
            )
            exc.add_note(f"Each fold needs at least 1 sample, but {n_samples} < {k}.")
            raise exc

        self._calibration_set = np.empty(n_samples, dtype=np.float64)
        calibration_offset = 0

        folds = KFold(
            n_splits=k,
            shuffle=self._shuffle,
            random_state=seed if self._shuffle else None,
        )

        fold_iterator = (
            tqdm(folds.split(x), total=k, desc="Calibration")
            if _crossval_logger.isEnabledFor(logging.INFO)
            else folds.split(x)
        )

        for i, (train_idx, calib_idx) in enumerate(fold_iterator):
            self._calibration_ids.extend(calib_idx.tolist())

            model = copy(detector_)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            model.fit(x[train_idx])

            if self._plus:
                self._detector_list.append(deepcopy(model))

            fold_scores = model.decision_function(x[calib_idx])
            n_fold_samples = len(fold_scores)
            end_idx = calibration_offset + n_fold_samples
            self._calibration_set[calibration_offset:end_idx] = fold_scores
            calibration_offset += n_fold_samples

        if not self._plus:
            model = copy(detector_)
            if hasattr(model, "set_params"):
                try:
                    model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            model.fit(x)
            self._detector_list.append(deepcopy(model))

        return self._detector_list, self._calibration_set

    @property
    def calibration_ids(self) -> list[int]:
        """Indices of samples used for calibration."""
        return self._calibration_ids.copy()

    @property
    def k(self) -> int | None:
        """Number of folds (None for jackknife mode)."""
        return self._k

    @property
    def plus(self) -> bool:
        """Whether ensemble mode is enabled."""
        return self._plus


def _train_bootstrap_model(
    detector: AnomalyDetector,
    x: np.ndarray,
    bootstrap_indices: np.ndarray,
    seed: int | None,
) -> AnomalyDetector:
    """Train a single bootstrap model (module-level for safe pickling).

    This function is defined at module level to ensure clean pickling
    when used with ProcessPoolExecutor, avoiding capture of unnecessary
    class state.

    Args:
        detector: Base detector to clone and train.
        x: Full training data array.
        bootstrap_indices: Indices for bootstrap sample.
        seed: Random seed for reproducibility.

    Returns:
        Trained detector model.
    """
    model = deepcopy(detector)
    if hasattr(model, "set_params"):
        try:
            model.set_params(random_state=seed)
        except (TypeError, ValueError):
            pass  # Detector may not support random_state parameter
    model.fit(x[bootstrap_indices])
    return model


class JackknifeBootstrap(BaseStrategy):
    """Jackknife+-after-Bootstrap (JaB+) conformal anomaly detection.

    Implements the JaB+ method which provides predictive inference for ensemble
    models trained on bootstrap samples. Uses out-of-bag samples for calibration.

    Args:
        n_bootstraps: Number of bootstrap iterations. Defaults to 100.
        aggregation_method: How to aggregate OOB predictions (MEAN or MEDIAN).
            Defaults to Aggregation.MEAN.
        plus: Whether to use ensemble mode. Defaults to True.

    References:
        Jin, Ying, and Emmanuel J. Candès. "Selection by Prediction with Conformal
        p-values." Journal of Machine Learning Research 24.244 (2023): 1-41.
    """

    def __init__(
        self,
        n_bootstraps: int = 100,
        aggregation_method: Aggregation = Aggregation.MEAN,
        plus: bool = True,
    ) -> None:
        super().__init__(plus=plus)

        if n_bootstraps < 2:
            exc = ValueError(
                f"Number of bootstraps must be at least 2, got {n_bootstraps}. "
                f"Typical values are 50-200 for jackknife-after-bootstrap."
            )
            exc.add_note(f"Received n_bootstraps={n_bootstraps}, which is invalid.")
            raise exc

        if aggregation_method not in [Aggregation.MEAN, Aggregation.MEDIAN]:
            exc = ValueError(
                f"aggregation_method must be Aggregation.MEAN or Aggregation.MEDIAN, "
                f"got {aggregation_method}."
            )
            raise exc

        if not plus:
            _bootstrap_logger.warning(
                "Setting plus=False may compromise conformal validity. "
                "The plus variant (plus=True) is recommended."
            )

        self._n_bootstraps: int = n_bootstraps
        self._aggregation_method: Aggregation = aggregation_method

        self._detector_list: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_ids: list[int] = []

        # Internal state
        self._bootstrap_models: list[AnomalyDetector | None] = []
        self._oob_mask: np.ndarray = np.array([])

    def fit_calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        detector: AnomalyDetector,
        seed: int | None = None,
        weighted: bool = False,
        n_jobs: int | None = None,
    ) -> tuple[list[AnomalyDetector], np.ndarray]:
        """Fit and calibrate using JaB+ method.

        Args:
            x: Input data matrix.
            detector: The base anomaly detector.
            seed: Random seed for reproducibility. Defaults to None.
            weighted: Not used in JaB+. Defaults to False.
            n_jobs: Number of parallel jobs. Defaults to None (sequential).

        Returns:
            Tuple of (list of trained detectors, calibration scores array).
        """
        n_samples = len(x)
        generator = np.random.default_rng(seed)

        _bootstrap_logger.info(
            f"Bootstrap (JaB+): {n_samples:,} samples, "
            f"{self._n_bootstraps:,} iterations"
        )

        self._bootstrap_models = [None] * self._n_bootstraps
        all_bootstrap_indices, self._oob_mask = self._generate_bootstrap_indices(
            generator, n_samples
        )

        if n_jobs is None or n_jobs == 1:
            bootstrap_iterator = (
                tqdm(range(self._n_bootstraps), desc="Calibration")
                if _bootstrap_logger.isEnabledFor(logging.INFO)
                else range(self._n_bootstraps)
            )
            for i in bootstrap_iterator:
                bootstrap_indices = all_bootstrap_indices[i]
                model = _train_bootstrap_model(detector, x, bootstrap_indices, seed)
                self._bootstrap_models[i] = model
        else:
            self._train_models_parallel(
                detector, x, all_bootstrap_indices, seed, n_jobs
            )

        oob_scores = self._compute_oob_scores(x)

        self._calibration_set = oob_scores
        self._calibration_ids = list(range(n_samples))

        if self._plus:
            self._detector_list = self._bootstrap_models.copy()
        else:
            final_model = deepcopy(detector)
            if hasattr(final_model, "set_params"):
                try:
                    final_model.set_params(random_state=seed)
                except (TypeError, ValueError):
                    pass  # Detector may not support random_state parameter
            final_model.fit(x)
            self._detector_list = [final_model]

        return self._detector_list, self._calibration_set

    def _generate_bootstrap_indices(
        self, generator: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate bootstrap indices with guaranteed OOB coverage."""
        if n_samples < 2:
            raise ValueError("JackknifeBootstrap requires at least 2 samples.")

        indices = np.empty((self._n_bootstraps, n_samples), dtype=int)
        oob_mask = np.zeros((self._n_bootstraps, n_samples), dtype=bool)
        coverage = np.zeros(n_samples, dtype=bool)
        population = np.arange(n_samples)

        for i in range(self._n_bootstraps):
            uncovered = np.where(~coverage)[0]
            if uncovered.size == 0:
                draw_pool = population
            else:
                shuffled_uncovered = generator.permutation(uncovered)
                remaining_iters = self._n_bootstraps - i
                chunk_size = int(np.ceil(shuffled_uncovered.size / remaining_iters))
                chunk_size = min(chunk_size, n_samples - 1)
                chunk_size = max(1, chunk_size)
                chunk = shuffled_uncovered[:chunk_size]
                draw_mask = np.ones(n_samples, dtype=bool)
                draw_mask[chunk] = False
                draw_pool = population[draw_mask]

            indices[i] = generator.choice(draw_pool, size=n_samples, replace=True)
            in_bag_mask = np.zeros(n_samples, dtype=bool)
            in_bag_mask[indices[i]] = True
            oob_mask[i] = ~in_bag_mask
            coverage |= oob_mask[i]

        uncovered = np.where(~coverage)[0]
        if uncovered.size > 0:
            raise ValueError(
                "Failed to generate complete OOB coverage. "
                "Consider increasing n_bootstraps."
            )
        return indices, oob_mask

    def _train_models_parallel(
        self,
        detector: AnomalyDetector,
        x: pd.DataFrame | np.ndarray,
        all_bootstrap_indices: np.ndarray,
        seed: int | None,
        n_jobs: int,
    ) -> None:
        """Train bootstrap models in parallel.

        Uses module-level _train_bootstrap_model function to ensure clean
        pickling without capturing unnecessary class state.
        """
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _train_bootstrap_model,
                    detector,
                    x,
                    all_bootstrap_indices[i],
                    seed,
                ): i
                for i in range(self._n_bootstraps)
            }

            future_iterator = (
                tqdm(
                    as_completed(futures), total=self._n_bootstraps, desc="Calibration"
                )
                if _bootstrap_logger.isEnabledFor(logging.INFO)
                else as_completed(futures)
            )
            for future in future_iterator:
                i = futures[future]
                self._bootstrap_models[i] = future.result()

    def _aggregate_predictions(self, predictions: list | np.ndarray) -> float:
        """Aggregate predictions using configured method."""
        if len(predictions) == 0:
            return np.nan

        match self._aggregation_method:
            case Aggregation.MEAN:
                return np.mean(predictions)
            case Aggregation.MEDIAN:
                return np.median(predictions)
            case _:
                raise ValueError(f"Unsupported aggregation: {self._aggregation_method}")

    def _compute_oob_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute out-of-bag calibration scores."""
        n_samples = len(x)
        all_predictions = [[] for _ in range(n_samples)]

        for model_idx, model in enumerate(self._bootstrap_models):
            oob_samples = self._oob_mask[model_idx]
            oob_indices = np.where(oob_samples)[0]

            if len(oob_indices) > 0:
                oob_predictions = model.decision_function(x[oob_indices])
                for idx, pred in zip(oob_indices, oob_predictions):
                    all_predictions[idx].append(pred)

        # Check coverage
        no_predictions = np.array([len(preds) == 0 for preds in all_predictions])
        if np.any(no_predictions):
            raise ValueError(
                f"Samples {np.where(no_predictions)[0]} have no OOB predictions. "
                "Consider increasing n_bootstraps."
            )

        oob_scores = np.array(
            [self._aggregate_predictions(preds) for preds in all_predictions]
        )
        return oob_scores

    @property
    def calibration_ids(self) -> list[int]:
        """Indices used for calibration (all samples in JaB+)."""
        return self._calibration_ids.copy()

    @property
    def n_bootstraps(self) -> int:
        """Number of bootstrap iterations."""
        return self._n_bootstraps

    @property
    def aggregation_method(self) -> Aggregation:
        """Aggregation method for OOB predictions."""
        return self._aggregation_method


__all__ = [
    "Aggregation",
    "BaseStrategy",
    "CrossValidation",
    "JackknifeBootstrap",
    "Split",
]
