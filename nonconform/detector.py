"""Core conformal anomaly detector implementation.

This module provides the main ConformalDetector class that wraps any anomaly
detector with conformal inference for valid p-values and FDR control.

Classes:
    BaseConformalDetector: Abstract base class for conformal detectors.
    ConformalDetector: Main conformal anomaly detector with optional weighting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from nonconform.adapters import adapt
from nonconform.scoring import Empirical
from nonconform.structures import AnomalyDetector, ConformalResult
from nonconform.weighting import BaseWeightEstimator, IdentityWeightEstimator

from ._internal import (
    Aggregation,
    aggregate,
    ensure_numpy_array,
    set_params,
)

if TYPE_CHECKING:
    from nonconform.resampling import BaseStrategy
    from nonconform.scoring import BaseEstimation


def _safe_copy(arr: np.ndarray | None) -> np.ndarray | None:
    """Return a copy of array or None if None."""
    return None if arr is None else arr.copy()


class BaseConformalDetector(ABC):
    """Abstract base class for all conformal anomaly detectors.

    Defines the core interface that all conformal anomaly detection implementations
    must provide. All conformal detectors follow a two-phase workflow:

    1. **Calibration Phase**: `fit()` trains detector, computes calibration scores
    2. **Inference Phase**: `predict()` converts new data scores to valid p-values

    Subclasses must implement both abstract methods.

    Note:
        This is an abstract class and cannot be instantiated directly.
        Use `ConformalDetector` for the main implementation.
    """

    @ensure_numpy_array
    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray) -> None:
        """Fit the detector model(s) and compute calibration scores.

        Args:
            x: The dataset used for fitting the model(s) and determining
                calibration scores.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @ensure_numpy_array
    @abstractmethod
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates or p-values for new data.

        Args:
            x: The new data instances for which to make anomaly estimates.
            raw: Whether to return raw anomaly scores or p-values. Defaults to False.

        Returns:
            Array containing the anomaly estimates (p-values or raw scores).
        """
        raise NotImplementedError("Subclasses must implement predict()")


class ConformalDetector(BaseConformalDetector):
    """Unified conformal anomaly detector with optional covariate shift handling.

    Provides distribution-free anomaly detection with valid p-values and False
    Discovery Rate (FDR) control by wrapping any anomaly detector with conformal
    inference. Supports PyOD detectors, sklearn-compatible detectors, and custom
    detectors implementing the AnomalyDetector protocol.

    When no weight estimator is provided (standard conformal prediction):
    - Uses classical conformal inference for exchangeable data
    - Provides optimal performance and memory usage
    - Suitable when training and test data come from the same distribution

    When a weight estimator is provided (weighted conformal prediction):
    - Handles distribution shift between calibration and test data
    - Estimates importance weights to maintain statistical validity
    - Slightly higher computational cost but robust to covariate shift

    Args:
        detector: Anomaly detector (PyOD, sklearn-compatible, or custom).
        strategy: The conformal strategy for fitting and calibration.
        estimation: P-value estimation strategy. Defaults to Empirical().
        weight_estimator: Weight estimator for covariate shift. Defaults to None.
        aggregation: Method for aggregating scores from multiple models.
            Defaults to Aggregation.MEDIAN.
        seed: Random seed for reproducibility. Defaults to None.
        verbose: If True, displays progress bars during prediction. Defaults to False.

    Attributes:
        detector: The underlying anomaly detection model.
        strategy: The calibration strategy for computing p-values.
        weight_estimator: Optional weight estimator for handling covariate shift.
        aggregation: Method for combining scores from multiple models.
        seed: Random seed for reproducible results.
        verbose: Whether to display progress bars.
        _detector_set: List of trained detector models (populated after fit).
        _calibration_set: Calibration scores (populated after fit).

    Examples:
        Standard conformal prediction:

        ```python
        from pyod.models.iforest import IForest
        from nonconform import ConformalDetector, Split

        detector = ConformalDetector(
            detector=IForest(), strategy=Split(n_calib=0.2), seed=42
        )
        detector.fit(X_train)
        p_values = detector.predict(X_test)
        ```

        Weighted conformal prediction:

        ```python
        from nonconform import logistic_weight_estimator

        detector = ConformalDetector(
            detector=IForest(),
            strategy=Split(n_calib=0.2),
            weight_estimator=logistic_weight_estimator(),
            seed=42,
        )
        detector.fit(X_train)
        p_values = detector.predict(X_test)
        ```

    Note:
        Some PyOD detectors are incompatible with conformal anomaly detection
        because they require clustering. Known incompatible detectors include:
        CBLOF, COF, RGraph, Sampling, SOS.
    """

    def __init__(
        self,
        detector: Any,
        strategy: BaseStrategy,
        estimation: BaseEstimation | None = None,
        weight_estimator: BaseWeightEstimator | None = None,
        aggregation: Aggregation = Aggregation.MEDIAN,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(aggregation, Aggregation):
            valid_methods = ", ".join([f"Aggregation.{a.name}" for a in Aggregation])
            raise TypeError(
                f"aggregation must be an Aggregation enum, "
                f"got {type(aggregation).__name__}. "
                f"Valid options: {valid_methods}."
            )

        adapted_detector = adapt(detector)
        self.detector: AnomalyDetector = set_params(deepcopy(adapted_detector), seed)
        self.strategy: BaseStrategy = strategy
        self.weight_estimator: BaseWeightEstimator | None = weight_estimator
        self.estimation = estimation if estimation is not None else Empirical()

        # Propagate seed to estimation and weight_estimator
        if seed is not None and hasattr(self.estimation, "set_seed"):
            self.estimation.set_seed(seed)
        if seed is not None and self.weight_estimator is not None:
            if hasattr(self.weight_estimator, "set_seed"):
                self.weight_estimator.set_seed(seed)

        self.aggregation: Aggregation = aggregation
        self.seed: int | None = seed
        self.verbose: bool = verbose

        self._is_weighted_mode = weight_estimator is not None and not isinstance(
            weight_estimator, IdentityWeightEstimator
        )

        self._detector_set: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_samples: np.ndarray = np.array([])
        self._last_result: ConformalResult | None = None

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray) -> None:
        """Fit detector model(s) and compute calibration scores.

        Uses the specified strategy to train the base detector(s) and calculate
        non-conformity scores on the calibration set.

        Args:
            x: The dataset used for fitting and calibration.
        """
        self._detector_set, self._calibration_set = self.strategy.fit_calibrate(
            x=x,
            detector=self.detector,
            weighted=self._is_weighted_mode,
            seed=self.seed,
        )

        if (
            self._is_weighted_mode
            and self.strategy.calibration_ids is not None
            and len(self.strategy.calibration_ids) > 0
        ):
            self._calibration_samples = x[self.strategy.calibration_ids]
        else:
            self._calibration_samples = np.array([])

        self._last_result = None

    @ensure_numpy_array
    def predict(
        self,
        x: pd.DataFrame | np.ndarray,
        raw: bool = False,
    ) -> np.ndarray:
        """Generate anomaly estimates (p-values or raw scores) for new data.

        Args:
            x: New data instances for anomaly estimation.
            raw: If True, returns raw anomaly scores. If False, returns p-values.
                Defaults to False.

        Returns:
            Array of anomaly estimates.

        Raises:
            RuntimeError: If fit() has not been called.

        Note:
            In weighted mode, the weight estimator is re-fitted on each call to
            compute density ratios relative to the test batch. This is by design
            for valid coverage guarantees, but adds computational overhead. For
            repeated predictions on small batches, consider batching test data.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        iterable = (
            tqdm(self._detector_set, total=len(self._detector_set), desc="Aggregation")
            if self.verbose
            else self._detector_set
        )

        scores = np.vstack(
            [np.asarray(model.decision_function(x)) for model in iterable]
        )

        estimates = aggregate(method=self.aggregation, scores=scores)

        weights: tuple[np.ndarray, np.ndarray] | None = None
        if self._is_weighted_mode and self.weight_estimator is not None:
            self.weight_estimator.fit(self._calibration_samples, x)
            weights = self.weight_estimator.get_weights()

        calib_weights, test_weights = weights if weights else (None, None)

        if raw:
            self._last_result = ConformalResult(
                p_values=None,
                test_scores=estimates.copy(),
                calib_scores=self._calibration_set.copy(),
                test_weights=_safe_copy(test_weights),
                calib_weights=_safe_copy(calib_weights),
                metadata={},
            )
            return estimates

        p_values = self.estimation.compute_p_values(
            estimates, self._calibration_set, weights
        )

        metadata: dict[str, Any] = {}
        if hasattr(self.estimation, "get_metadata"):
            meta = self.estimation.get_metadata()
            if meta:
                metadata = dict(meta)

        self._last_result = ConformalResult(
            p_values=p_values.copy(),
            test_scores=estimates.copy(),
            calib_scores=self._calibration_set.copy(),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata=metadata,
        )

        return p_values

    @property
    def detector_set(self) -> list[AnomalyDetector]:
        """Returns a copy of the list of trained detector models."""
        return self._detector_set.copy()

    @property
    def calibration_set(self) -> np.ndarray:
        """Returns a copy of the calibration scores."""
        return self._calibration_set.copy()

    @property
    def calibration_samples(self) -> np.ndarray:
        """Returns a copy of the calibration samples (weighted mode only)."""
        return self._calibration_samples.copy()

    @property
    def last_result(self) -> ConformalResult | None:
        """Return the most recent conformal result snapshot."""
        return None if self._last_result is None else self._last_result.copy()

    @property
    def is_fitted(self) -> bool:
        """Returns whether the detector has been fitted."""
        return len(self._detector_set) > 0 and len(self._calibration_set) > 0


__all__ = [
    "BaseConformalDetector",
    "ConformalDetector",
]
