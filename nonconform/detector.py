"""Core conformal anomaly detector implementation.

This module provides the main ConformalDetector class that wraps any anomaly
detector with conformal inference for valid p-values and FDR control.

Classes:
    BaseConformalDetector: Abstract base class for conformal detectors.
    ConformalDetector: Main conformal anomaly detector with optional weighting.
"""

from __future__ import annotations

import hashlib
import inspect
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
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
from nonconform.scoring import Empirical
from nonconform.structures import AnomalyDetector, ConformalResult
from nonconform.weighting import BaseWeightEstimator, IdentityWeightEstimator

from ._internal import (
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


def _snapshot_param(value: Any) -> Any:
    """Return an immutable constructor-parameter snapshot."""
    return deepcopy(value)


def _batch_signature(x: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    """Return stable signature for a concrete batch.

    This helper computes a full digest over batch bytes, which is O(n) in both
    time and memory. Use only when strict batch-content verification is desired.
    """
    contiguous = np.ascontiguousarray(x)
    digest = hashlib.blake2b(
        contiguous.tobytes(),
        digest_size=16,
    ).hexdigest()
    return contiguous.shape, str(contiguous.dtype), digest


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
    must provide. Conformal detectors support either an integrated or detached
    calibration workflow:

    1. **Integrated calibration**: `fit()` trains detector(s) and computes
       calibration scores
    2. **Detached calibration**: train detector externally, then call
       `calibrate()` on a separate calibration dataset
    3. **Inference Phase**: `compute_p_values()` converts new data scores to valid
       p-values

    Subclasses must implement both abstract methods.

    Note:
        This is an abstract class and cannot be instantiated directly.
        Use `ConformalDetector` for the main implementation.
    """

    @ensure_numpy_array
    @abstractmethod
    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        *,
        n_jobs: int | None = None,
    ) -> Self:
        """Fit the detector model(s) and compute calibration scores.

        Args:
            x: The dataset used for fitting the model(s) and determining
                calibration scores.
            y: Ignored. Present for sklearn API compatibility.
            n_jobs: Optional strategy-specific parallelism hint.
                Currently used by strategies that expose an ``n_jobs`` parameter
                (for example, ``JackknifeBootstrap``).

        Returns:
            The fitted detector instance.
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @ensure_numpy_array
    def calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> Self:
        """Calibrate a pre-fitted detector on separate calibration data.

        Args:
            x: Dataset used only to compute calibration scores.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            The calibrated detector instance.
        """
        raise NotImplementedError("Subclasses must implement calibrate()")

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
            If omitted (`None`), nonconform applies an implicit default policy:
            known sklearn normality detectors resolve to `"higher_is_normal"`,
            while PyOD and unknown custom detectors resolve to
            `"higher_is_anomalous"`. Explicit `"auto"` enables strict inference:
            known detector families are inferred, and unknown detectors raise.
            Defaults to None.
        seed: Random seed for reproducibility. Defaults to None.
        verbose: If True, displays progress bars during prediction. Defaults to False.
        verify_prepared_batch_content: If True (default), weighted reuse mode
            (``refit_weights=False``) verifies exact batch content identity via
            hashing. This adds O(n) overhead per checked batch. Set to False to
            skip content hashing and validate only batch size.

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

        Detached calibration with a pre-trained model (Split strategy):

        ```python
        base_detector.fit(X_fit)
        detector = ConformalDetector(
            detector=base_detector, strategy=Split(n_calib=0.2)
        )
        detector.calibrate(X_calib)
        p_values = detector.compute_p_values(X_test)
        ```

    Note:
        Some PyOD detectors are incompatible with conformal anomaly detection
        because they require clustering. Known incompatible detectors include:
        CBLOF, COF, RGraph, Sampling, SOS.
    """

    _NESTED_COMPONENTS = ("detector", "strategy", "estimation", "weight_estimator")

    def __init__(
        self,
        detector: Any,
        strategy: BaseStrategy,
        estimation: BaseEstimation | None = None,
        weight_estimator: BaseWeightEstimator | None = None,
        aggregation: str = "median",
        score_polarity: ScorePolarity
        | Literal["auto", "higher_is_anomalous", "higher_is_normal"]
        | None = None,
        seed: int | None = None,
        verbose: bool = False,
        verify_prepared_batch_content: bool = True,
    ) -> None:
        self._configure(
            detector=detector,
            strategy=strategy,
            estimation=estimation,
            weight_estimator=weight_estimator,
            aggregation=aggregation,
            score_polarity=score_polarity,
            seed=seed,
            verbose=verbose,
            verify_prepared_batch_content=verify_prepared_batch_content,
        )

    def _reset_fit_state(self) -> None:
        """Clear all learned state derived from fit()."""
        self._detector_set: list[AnomalyDetector] = []
        self._calibration_set: np.ndarray = np.array([])
        self._calibration_samples: np.ndarray = np.array([])
        self._prepared_weight_batch_size: int | None = None
        self._prepared_weight_batch_signature: (
            tuple[tuple[int, ...], str, str] | None
        ) = None
        self._last_result: ConformalResult | None = None

    def _configure(
        self,
        *,
        detector: Any,
        strategy: BaseStrategy,
        estimation: BaseEstimation | None,
        weight_estimator: BaseWeightEstimator | None,
        aggregation: str,
        score_polarity: ScorePolarity
        | Literal["auto", "higher_is_anomalous", "higher_is_normal"]
        | None,
        seed: int | None,
        verbose: bool,
        verify_prepared_batch_content: bool,
    ) -> None:
        """Apply constructor parameters and reset learned state."""
        self._init_detector = _snapshot_param(detector)
        self._init_strategy = _snapshot_param(strategy)
        self._init_estimation = _snapshot_param(estimation)
        self._init_weight_estimator = _snapshot_param(weight_estimator)
        self._init_aggregation = aggregation
        self._init_score_polarity = score_polarity
        self._init_seed = seed
        self._init_verbose = verbose
        self._init_verify_prepared_batch_content = verify_prepared_batch_content

        if seed is not None and seed < 0:
            raise ValueError(f"seed must be a non-negative integer or None, got {seed}")
        if not isinstance(verbose, bool):
            raise TypeError(
                f"verbose must be a boolean value, got {type(verbose).__name__}."
            )
        if not isinstance(verify_prepared_batch_content, bool):
            raise TypeError("verify_prepared_batch_content must be a boolean value.")
        normalized_aggregation = normalize_aggregation_method(aggregation)

        adapted_detector = adapt(detector)
        if score_polarity is None:
            resolved_polarity = resolve_implicit_score_polarity(adapted_detector)
        else:
            resolved_polarity = resolve_score_polarity(adapted_detector, score_polarity)
        normalized_detector = apply_score_polarity(adapted_detector, resolved_polarity)

        self.detector = set_params(deepcopy(normalized_detector), seed)
        # Keep an internal strategy copy so external mutations after construction
        # do not alter detector behavior.
        self.strategy = deepcopy(strategy)
        self.weight_estimator = weight_estimator
        self.estimation = estimation if estimation is not None else Empirical()

        # Propagate seed to estimation and weight_estimator
        if seed is not None and hasattr(self.estimation, "set_seed"):
            self.estimation.set_seed(seed)
        if (
            seed is not None
            and self.weight_estimator is not None
            and hasattr(self.weight_estimator, "set_seed")
        ):
            self.weight_estimator.set_seed(seed)

        self.aggregation = normalized_aggregation
        self._score_polarity = resolved_polarity
        self.seed = seed
        self.verbose = verbose
        self.verify_prepared_batch_content = verify_prepared_batch_content
        self._is_weighted_mode = weight_estimator is not None and not isinstance(
            weight_estimator, IdentityWeightEstimator
        )
        self._reset_fit_state()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters following sklearn conventions.

        Notes:
            - ``deep=False`` returns constructor-facing parameters used for
              sklearn clone compatibility.
            - ``deep=True`` also includes nested ``component__param`` entries
              read from the current runtime components (effective/internal state),
              which may differ from originally passed constructor objects after
              adaptation/normalization.
        """
        params: dict[str, Any] = {
            "detector": self._init_detector,
            "strategy": self._init_strategy,
            "estimation": self._init_estimation,
            "weight_estimator": self._init_weight_estimator,
            "aggregation": self._init_aggregation,
            "score_polarity": self._init_score_polarity,
            "seed": self._init_seed,
            "verbose": self._init_verbose,
            "verify_prepared_batch_content": self._init_verify_prepared_batch_content,
        }
        if not deep:
            return params

        for component_name in self._NESTED_COMPONENTS:
            component = getattr(self, component_name)
            if component is None or not hasattr(component, "get_params"):
                continue
            try:
                component_params = component.get_params(deep=True)
            except TypeError:
                component_params = component.get_params()
            for key, value in component_params.items():
                params[f"{component_name}__{key}"] = value
        return params

    def set_params(self, **params: Any) -> Self:
        """Set estimator parameters following sklearn conventions."""
        if not params:
            return self

        updated_params = self.get_params(deep=False)
        nested_updates: dict[str, dict[str, Any]] = {}

        for key, value in params.items():
            if "__" in key:
                component_name, nested_key = key.split("__", 1)
                if component_name not in self._NESTED_COMPONENTS:
                    raise ValueError(f"Invalid parameter {component_name!r}.")
                nested_updates.setdefault(component_name, {})[nested_key] = value
                continue

            if key not in updated_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {type(self).__name__}."
                )
            updated_params[key] = value

        for component_name, component_params in nested_updates.items():
            component = updated_params[component_name]
            if component is None:
                raise ValueError(
                    f"Cannot set nested parameters for {component_name!r}: "
                    "component is None."
                )
            if not hasattr(component, "set_params"):
                raise ValueError(
                    f"Cannot set nested parameters for {component_name!r}: "
                    "component does not implement set_params()."
                )
            component.set_params(**component_params)

        self._configure(**updated_params)
        return self

    def __sklearn_clone__(self) -> Self:
        """Return sklearn-compatible unfitted clone from constructor snapshots."""
        params = self.get_params(deep=False)
        cloned_params = {key: _snapshot_param(value) for key, value in params.items()}
        return type(self)(**cloned_params)

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
    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        *,
        n_jobs: int | None = None,
    ) -> Self:
        """Fit detector model(s) and compute calibration scores.

        Uses the specified strategy to train the base detector(s) and calculate
        non-conformity scores on the calibration set.

        Args:
            x: The dataset used for fitting and calibration.
            y: Ignored. Present for sklearn API compatibility.
            n_jobs: Optional strategy-specific parallelism hint. Supported by
                strategies whose ``fit_calibrate`` signature includes ``n_jobs``
                (for example, ``JackknifeBootstrap``).

        Returns:
            The fitted detector instance (for method chaining).
        """
        _ = y
        fit_kwargs: dict[str, Any] = {
            "x": x,
            "detector": self.detector,
            "weighted": self._is_weighted_mode,
            "seed": self.seed,
        }
        if n_jobs is not None:
            strategy_params = inspect.signature(self.strategy.fit_calibrate).parameters
            if "n_jobs" not in strategy_params:
                raise ValueError(
                    f"Strategy {type(self.strategy).__name__} does not support n_jobs. "
                    "Pass n_jobs only when using a strategy that exposes it, "
                    "such as JackknifeBootstrap."
                )
            fit_kwargs["n_jobs"] = n_jobs

        self._detector_set, self._calibration_set = self.strategy.fit_calibrate(
            **fit_kwargs
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
        self._prepared_weight_batch_signature = None
        self._last_result = None
        return self

    @ensure_numpy_array
    def calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> Self:
        """Calibrate a pre-fitted detector on separate calibration data.

        This detached workflow is currently supported only for ``Split`` strategy,
        where a single pre-fitted model is calibrated on a dedicated dataset.

        Args:
            x: Calibration dataset used to compute calibration scores.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            The calibrated detector instance (for method chaining).

        Raises:
            ValueError: If strategy is not ``Split``.
            NotFittedError: If the base detector appears unfitted.
        """
        _ = y
        from nonconform.resampling import Split

        if not isinstance(self.strategy, Split):
            raise ValueError(
                "calibrate() is supported only with Split strategy. "
                f"Got {type(self.strategy).__name__}."
            )

        try:
            calibration_set = np.asarray(
                self.detector.decision_function(x),
                dtype=float,
            ).ravel()
        except Exception as exc:
            message = str(exc).lower()
            if (
                isinstance(exc, NotFittedError)
                or "not fitted" in message
                or (isinstance(exc, AttributeError) and "has no attribute" in message)
            ):
                raise NotFittedError(
                    "Base detector is not fitted. Fit the base detector before "
                    "calling calibrate()."
                ) from exc
            raise

        if calibration_set.shape[0] != len(x):
            raise ValueError(
                "calibration scores must have one value per calibration sample. "
                f"Got {calibration_set.shape[0]} scores for {len(x)} samples."
            )

        self._detector_set = [self.detector]
        self._calibration_set = calibration_set
        if self._is_weighted_mode:
            self._calibration_samples = x.copy()
        else:
            self._calibration_samples = np.array([])

        self._prepared_weight_batch_size = None
        self._prepared_weight_batch_signature = None
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
            if self.verify_prepared_batch_content:
                self._prepared_weight_batch_signature = _batch_signature(x)
            else:
                self._prepared_weight_batch_signature = None
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
        if self.verify_prepared_batch_content and (
            self._prepared_weight_batch_signature != _batch_signature(x)
        ):
            raise ValueError(
                "Prepared weights do not match current batch content. "
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
        if self.verify_prepared_batch_content:
            self._prepared_weight_batch_signature = _batch_signature(x)
        else:
            self._prepared_weight_batch_signature = None
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
