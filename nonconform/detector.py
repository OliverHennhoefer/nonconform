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
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from nonconform.adapters import (
    adapt,
    apply_score_polarity,
    resolve_score_polarity,
)
from nonconform.scoring import Empirical
from nonconform.structures import AnomalyDetector, ConformalResult
from nonconform.weighting import BaseWeightEstimator, IdentityWeightEstimator

from ._internal import (
    AggregationMethod,
    ScorePolarity,
    aggregate,
    ensure_numpy_array,
    normalize_aggregation_method,
    set_params,
)

if TYPE_CHECKING:
    from nonconform.resampling import BaseStrategy
    from nonconform.scoring import BaseEstimation


def _safe_copy(arr: np.ndarray | None) -> np.ndarray | None:
    """Return a copy of array or None if None."""
    return None if arr is None else arr.copy()


def _as_numpy_with_index(
    x: pd.DataFrame | pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.Index | None]:
    """Return numpy view of input and optional pandas index.

    Pandas Series are interpreted as a single-feature batch with shape
    ``(n_samples, 1)``.
    """
    if isinstance(x, pd.Series):
        return x.to_numpy(copy=False).reshape(-1, 1), x.index
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(copy=False), x.index
    return x, None


class BaseConformalDetector(ABC):
    """Abstract base class for all conformal anomaly detectors.

    Defines the core interface that all conformal anomaly detection implementations
    must provide. All conformal detectors follow a two-phase workflow:

    1. **Calibration Phase**: `fit()` trains detector, computes calibration scores
    2. **Inference Phase**: `compute_p_values()` converts new data scores to valid
       p-values

    Subclasses must implement both abstract methods.

    Note:
        This is an abstract class and cannot be instantiated directly.
        Use `ConformalDetector` for the main implementation.
    """

    @ensure_numpy_array
    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Fit the detector model(s) and compute calibration scores.

        Args:
            x: The dataset used for fitting the model(s) and determining
                calibration scores.

        Returns:
            The fitted detector instance.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @abstractmethod
    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return conformal p-values for new data.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Ignored in standard mode.

        Returns:
            P-values as ndarray for numpy input, or pandas Series for pandas input.
        """
        raise NotImplementedError("Subclasses must implement compute_p_values()")

    @abstractmethod
    def score_samples(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return aggregated raw anomaly scores for new data.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Ignored in standard mode.

        Returns:
            Raw scores as ndarray for numpy input, or pandas Series for pandas input.
        """
        raise NotImplementedError("Subclasses must implement score_samples()")


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
            Defaults to "median".
        score_polarity: Score direction convention. Use `"higher_is_anomalous"`
            when higher raw scores indicate more anomalous samples, and
            `"higher_is_normal"` when higher scores indicate more normal samples.
            `"auto"` infers polarity for known detector families and raises for
            unknown detectors. Defaults to ScorePolarity.AUTO.
        seed: Random seed for reproducibility. Defaults to None.
        verbose: If True, displays progress bars during prediction. Defaults to False.

    Attributes:
        detector: The underlying anomaly detection model.
        strategy: The calibration strategy for computing p-values.
        weight_estimator: Optional weight estimator for handling covariate shift.
        aggregation: Method for combining scores from multiple models.
        score_polarity: Resolved score polarity used internally.
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
        p_values = detector.compute_p_values(X_test)
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
        p_values = detector.compute_p_values(X_test)
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
        aggregation: str = "median",
        score_polarity: ScorePolarity
        | Literal["auto", "higher_is_anomalous", "higher_is_normal"] = (
            ScorePolarity.AUTO
        ),
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        normalized_aggregation = normalize_aggregation_method(aggregation)

        adapted_detector = adapt(detector)
        resolved_polarity = resolve_score_polarity(adapted_detector, score_polarity)
        normalized_detector = apply_score_polarity(adapted_detector, resolved_polarity)

        self.detector: AnomalyDetector = set_params(deepcopy(normalized_detector), seed)
        self.strategy: BaseStrategy = strategy
        self.weight_estimator: BaseWeightEstimator | None = weight_estimator
        self.estimation = estimation if estimation is not None else Empirical()

        # Propagate seed to estimation and weight_estimator
        if seed is not None and hasattr(self.estimation, "set_seed"):
            self.estimation.set_seed(seed)
        if seed is not None and self.weight_estimator is not None:
            if hasattr(self.weight_estimator, "set_seed"):
                self.weight_estimator.set_seed(seed)

        self.aggregation: AggregationMethod = normalized_aggregation
        self._score_polarity: ScorePolarity = resolved_polarity
        self.seed: int | None = seed
        self.verbose: bool = verbose

        self._is_weighted_mode = weight_estimator is not None and not isinstance(
            weight_estimator, IdentityWeightEstimator
        )

        self._detector_set: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_samples: np.ndarray = np.array([])
        self._prepared_weight_batch_size: int | None = None
        self._last_result: ConformalResult | None = None

    def __repr__(self) -> str:
        """Return concise notebook-friendly detector summary."""
        return (
            "ConformalDetector("
            f"detector={type(self.detector).__name__}, "
            f"strategy={type(self.strategy).__name__}, "
            f"estimation={type(self.estimation).__name__}, "
            f"aggregation={self.aggregation!r}, "
            f"score_polarity={self._score_polarity.name}, "
            f"weighted_mode={self._is_weighted_mode}, "
            f"seed={self.seed}, "
            f"verbose={self.verbose}, "
            f"fitted={self.is_fitted}, "
            f"n_models={len(self._detector_set)}, "
            f"n_calibration={len(self._calibration_set)})"
        )

    @ensure_numpy_array
    def fit(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Fit detector model(s) and compute calibration scores.

        Uses the specified strategy to train the base detector(s) and calculate
        non-conformity scores on the calibration set.

        Args:
            x: The dataset used for fitting and calibration.

        Returns:
            The fitted detector instance (for method chaining).
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

        self._prepared_weight_batch_size = None
        self._last_result = None
        return self

    def _aggregate_scores(self, x: np.ndarray) -> np.ndarray:
        """Compute aggregated anomaly scores across fitted detector replicas."""
        if not self.is_fitted:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")

        iterable = (
            tqdm(self._detector_set, total=len(self._detector_set), desc="Aggregation")
            if self.verbose
            else self._detector_set
        )

        scores = np.vstack(
            [np.asarray(model.decision_function(x)) for model in iterable]
        )
        return aggregate(method=self.aggregation, scores=scores)

    def _resolve_weights(
        self,
        x: np.ndarray,
        *,
        refit_weights: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve calibration/test weights for the current batch."""
        if not self._is_weighted_mode or self.weight_estimator is None:
            return None

        if refit_weights:
            self.weight_estimator.fit(self._calibration_samples, x)
            self._prepared_weight_batch_size = len(x)
            return self.weight_estimator.get_weights()

        if self._prepared_weight_batch_size is None:
            raise RuntimeError(
                "Weights are not prepared. Call prepare_weights_for(batch) "
                "or use refit_weights=True."
            )
        if self._prepared_weight_batch_size != len(x):
            raise ValueError(
                "Prepared weights do not match current batch size. "
                "Call prepare_weights_for(batch) again or use refit_weights=True."
            )
        return self.weight_estimator.get_weights()

    @ensure_numpy_array
    def prepare_weights_for(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Prepare weighted conformal state for a specific test batch.

        In weighted mode, this fits the weight estimator for the supplied batch
        without producing predictions. Use this for explicit state transitions in
        exploratory workflows.

        Args:
            x: Test batch for which weights should be prepared.

        Returns:
            The fitted detector instance (for method chaining).

        Raises:
            NotFittedError: If fit() has not been called.
            RuntimeError: If weighted mode is disabled.
        """
        if not self.is_fitted:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")
        if not self._is_weighted_mode or self.weight_estimator is None:
            raise RuntimeError(
                "prepare_weights_for() requires weighted mode with a weight_estimator."
            )

        self.weight_estimator.fit(self._calibration_samples, x)
        self._prepared_weight_batch_size = len(x)
        return self

    def score_samples(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return aggregated raw anomaly scores for new data.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Defaults to True.

        Returns:
            Aggregated raw anomaly scores.
        """
        x_array, index = _as_numpy_with_index(x)
        estimates = self._aggregate_scores(x_array)
        weights = self._resolve_weights(x_array, refit_weights=refit_weights)
        calib_weights, test_weights = weights if weights else (None, None)

        self._last_result = ConformalResult(
            p_values=None,
            test_scores=estimates.copy(),
            calib_scores=self._calibration_set.copy(),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata={},
        )
        if index is not None:
            return pd.Series(estimates, index=index, name="score")
        return estimates

    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return conformal p-values for new data.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Defaults to True.

        Returns:
            Conformal p-values.
        """
        x_array, index = _as_numpy_with_index(x)
        estimates = self._aggregate_scores(x_array)
        weights = self._resolve_weights(x_array, refit_weights=refit_weights)
        calib_weights, test_weights = weights if weights else (None, None)

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
        if index is not None:
            return pd.Series(p_values, index=index, name="p_value")
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
    def score_polarity(self) -> ScorePolarity:
        """Returns the resolved score polarity convention."""
        return self._score_polarity

    @property
    def is_fitted(self) -> bool:
        """Returns whether the detector has been fitted."""
        return len(self._detector_set) > 0 and len(self._calibration_set) > 0


__all__ = [
    "BaseConformalDetector",
    "ConformalDetector",
]
