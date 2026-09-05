"""Core conformal anomaly detector implementation.

This module provides :class:`ConformalDetector`, which calibrates scores from a
supported anomaly detector. The default empirical estimator returns rank-based
conformal p-values; other estimators document their own interpretation. Weighted
mode estimates density-ratio weights for covariate-shift workflows. Selection
methods apply a multiple-testing procedure to a complete test batch. The
resulting validity and false discovery rate guarantees depend on the assumptions
documented for the chosen calibration and selection procedure.

Classes:
    BaseConformalDetector: Abstract base class for conformal detectors.
    ConformalDetector: Main conformal anomaly detector with optional weighting.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from nonconform.adapters import (
    adapt,
    apply_score_polarity,
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
from nonconform.resampling import Split
from nonconform.scoring import ConditionalEmpirical, Empirical
from nonconform.structures import AnomalyDetector, ConformalResult
from nonconform.weighting import BaseWeightEstimator, IdentityWeightEstimator

from ._internal import (
    Pruning,
    ScorePolarity,
    aggregate,
    ensure_numpy_array,
    normalize_aggregation_method,
    set_params,
)
from ._internal.provenance import (
    BatchSignature,
    CalibrationMode,
    EstimationFamily,
    ResultProvenance,
    StrategyFamily,
    batch_signature,
)

if TYPE_CHECKING:
    from nonconform.fdr import EValueSelectionResult
    from nonconform.resampling import BaseStrategy
    from nonconform.scoring import BaseEstimation

_WCS_PRUNING_SEED_DOMAIN = 0x574353  # ASCII "WCS"


def _safe_copy(arr: np.ndarray | None) -> np.ndarray | None:
    """Return a copy of array or None if None."""
    return None if arr is None else arr.copy()


def _snapshot_param(value: Any) -> Any:
    """Return an immutable constructor-parameter snapshot."""
    return deepcopy(value)


def _derive_wcs_pruning_seed(seed: int | None) -> int | None:
    """Derive a deterministic WCS-pruning stream distinct from ``seed``."""
    if seed is None:
        return None
    seed_sequence = np.random.SeedSequence([seed, _WCS_PRUNING_SEED_DOMAIN])
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


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
    3. **Inference phase**: `compute_p_values()` applies the configured
       estimation strategy, while `select()` combines estimation with the
       configured batch selection procedure

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
    """Wrap an anomaly detector with conformal calibration and batch selection.

    The wrapped detector may be a recognized scikit-learn estimator, a PyOD
    model, or a custom object implementing the
    :class:`~nonconform.structures.AnomalyDetector` protocol.

    In standard mode, the fitted strategy supplies one or more fixed scoring
    rules and calibration-score sets. With the default ``Empirical`` estimator,
    ``compute_p_values()`` ranks each test score against those calibration
    scores. The usual marginal p-value validity statement requires
    exchangeability of the relevant null calibration and test examples.
    ``select()`` then applies Benjamini-Hochberg to the full test family. Any FDR
    guarantee additionally depends on the dependence assumptions of that
    multiple-testing procedure. Other estimation strategies document their own
    interpretation and assumptions.

    Supplying ``weight_estimator`` enables weighted mode. The estimator learns
    density-ratio weights from calibration and test covariates, and ``select()``
    uses weighted conformalized selection. Its validity requires the applicable
    covariate-shift assumptions, support overlap, and adequate weights; the class
    cannot verify those scientific assumptions from data alone.

    With ``DerandomizedSplits``, ``fit()`` retains repeated model/calibration
    pairs and ``select()`` uniformly aggregates per-split e-values before e-BH.
    Inspect ``last_selection_result`` for evidence and selection diagnostics.
    P-value methods and detached calibration are unavailable for this strategy.

    Args:
        detector: Anomaly detector (PyOD, sklearn-compatible, or custom).
        strategy: The conformal strategy for fitting, calibration, and evidence
            construction. DerandomizedSplits selects through e-values and e-BH.
        estimation: P-value estimation strategy. Defaults to Empirical(). Unused
            by DerandomizedSplits, which accepts only None or ordinary Empirical.
        weight_estimator: Weight estimator for covariate shift. Defaults to None.
        aggregation: Method for aggregating scores from multiple fitted models:
            ``"mean"``, ``"median"``, ``"minimum"``, or ``"maximum"``.
            Defaults to ``"median"``.
            For DerandomizedSplits, affects score_samples() only; selection
            always uniformly averages per-split e-values.
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
        verbose: If True, displays aggregation progress for multi-model
            strategies. Defaults to False.
        verify_prepared_batch_content: If True (default), weighted reuse mode
            (``refit_weights=False``) verifies exact batch content identity via
            hashing. This adds O(n) overhead per checked batch. Set to False to
            skip content hashing and validate only batch size.

    Attributes:
        detector: The underlying anomaly detection model.
        strategy: The calibration and evidence-construction strategy.
        weight_estimator: Optional weight estimator for handling covariate shift.
        aggregation: Method for combining scores from multiple models.
        score_polarity: Resolved score polarity used internally.
        seed: Random seed for reproducible results.
        verbose: Whether to display progress bars.

    Examples:
        Standard conformal p-values and batch selection:

        ```python
        import numpy as np
        from sklearn.ensemble import IsolationForest

        from nonconform import ConformalDetector, Split

        rng = np.random.default_rng(42)
        x_reference = rng.normal(size=(300, 2))
        x_test = np.vstack([rng.normal(size=(38, 2)), rng.normal(loc=5.0, size=(2, 2))])
        detector = ConformalDetector(
            detector=IsolationForest(random_state=42),
            strategy=Split(n_calib=0.25),
            score_polarity="higher_is_normal",
            seed=42,
        )
        detector.fit(x_reference)
        p_values = detector.compute_p_values(x_test)
        selected = detector.select(x_test, alpha=0.10)
        print(p_values.shape, np.flatnonzero(selected))
        ```

        Weighted conformal p-values under a simulated covariate shift:

        ```python
        import numpy as np
        from sklearn.ensemble import IsolationForest

        from nonconform import (
            ConformalDetector,
            Split,
            logistic_weight_estimator,
        )

        rng = np.random.default_rng(7)
        x_reference = rng.normal(size=(400, 2))
        x_test = rng.normal(loc=0.5, size=(40, 2))
        detector = ConformalDetector(
            detector=IsolationForest(random_state=7),
            strategy=Split(n_calib=0.25),
            weight_estimator=logistic_weight_estimator(),
            score_polarity="higher_is_normal",
            seed=7,
        )
        detector.fit(x_reference)
        p_values = detector.compute_p_values(x_test)
        print(p_values.shape, p_values.min(), p_values.max())
        ```

        Detached calibration with a pre-trained model (Split strategy):

        ```python
        import numpy as np
        from sklearn.ensemble import IsolationForest

        from nonconform import ConformalDetector, Split

        rng = np.random.default_rng(11)
        x_fit = rng.normal(size=(200, 2))
        x_calibration = rng.normal(size=(100, 2))
        x_test = rng.normal(size=(10, 2))
        base_detector = IsolationForest(random_state=11).fit(x_fit)
        detector = ConformalDetector(
            detector=base_detector,
            strategy=Split(n_calib=0.2),
            score_polarity="higher_is_normal",
        )
        detector.calibrate(x_calibration)
        p_values = detector.compute_p_values(x_test)
        print(p_values)
        ```

    Note:
        Strict inductive conformal workflows require a fixed training-only
        score map at inference time. PyOD detectors known to violate this are:
        CD, COF, COPOD, ECOD, LMDD, LOCI, RGraph, SOD, SOS.
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
        self._calibration_mode: CalibrationMode | None = None
        self._n_features_in: int | None = None
        self._prepared_weight_batch_size: int | None = None
        self._prepared_weight_batch_signature: BatchSignature | None = None
        self._last_result: ConformalResult | None = None
        self._last_selection_result: EValueSelectionResult | None = None

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
        if strategy._uses_e_values or getattr(
            getattr(self, "strategy", None), "_uses_e_values", False
        ):
            self._reset_fit_state()
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
        if self.strategy._uses_e_values:
            if self._is_weighted_mode:
                raise ValueError("DerandomizedSplits does not support weighting.")
            if estimation is not None and type(estimation) is not Empirical:
                raise ValueError(
                    "DerandomizedSplits constructs e-values directly; p-value "
                    "estimation is unused. Omit estimation or use ordinary Empirical()."
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
            f"n_calibration={self._calibration_set.shape[-1]})"
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
        self._last_selection_result = None
        if self.strategy._uses_e_values:
            self._reset_fit_state()
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
        self._calibration_mode = CalibrationMode.INTEGRATED
        self._n_features_in = int(x.shape[1])

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
        self._last_selection_result = None
        if not isinstance(self.strategy, Split):
            raise ValueError(
                "calibrate() is supported only with Split strategy. "
                f"Got {type(self.strategy).__name__}. Use fit(x_reference) for "
                "integrated calibration."
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
        self._calibration_mode = CalibrationMode.DETACHED
        self._n_features_in = int(x.shape[1])
        if self._is_weighted_mode:
            self._calibration_samples = x.copy()
        else:
            self._calibration_samples = np.array([])

        self._prepared_weight_batch_size = None
        self._prepared_weight_batch_signature = None
        self._last_result = None
        return self

    def _score_models(self, x: np.ndarray) -> np.ndarray:
        """Score one batch with every replica, preserving model order."""
        if not self.is_fitted:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")
        if self.strategy._uses_e_values:
            x = np.asarray(x)
            if x.ndim != 2 or x.shape[1] != self._n_features_in:
                raise ValueError(
                    "x must be a two-dimensional batch with the fitted feature count."
                )

        iterable = (
            tqdm(self._detector_set, total=len(self._detector_set), desc="Aggregation")
            if self.verbose
            else self._detector_set
        )

        rows = []
        for model in iterable:
            row = np.asarray(model.decision_function(x))
            if self.strategy._uses_e_values and row.shape != (len(x),):
                raise ValueError("Each model must return one score per test row.")
            rows.append(row)
        return np.vstack(rows)

    def _aggregate_scores(self, x: np.ndarray) -> np.ndarray:
        """Compute aggregated anomaly scores across fitted detector replicas."""
        scores = self._score_models(x)
        return aggregate(method=self.aggregation, scores=scores)

    def _result_metadata(self) -> dict[str, Any]:
        """Return the released metadata shape for p-value snapshots."""
        return {
            "nonconform": {
                "strategy": type(self.strategy).__name__,
                "estimation": type(self.estimation).__name__,
                "weighted": self._is_weighted_mode,
            }
        }

    def _result_provenance(
        self,
        test_batch_signature: BatchSignature,
    ) -> ResultProvenance:
        """Return typed provenance for a detector-produced result snapshot."""
        if isinstance(self.estimation, ConditionalEmpirical):
            estimation_family = EstimationFamily.CONDITIONAL_EMPIRICAL
        elif isinstance(self.estimation, Empirical):
            estimation_family = EstimationFamily.EMPIRICAL
        else:
            estimation_family = EstimationFamily.OTHER
        return ResultProvenance(
            strategy_family=(
                StrategyFamily.SPLIT
                if isinstance(self.strategy, Split)
                else StrategyFamily.OTHER
            ),
            estimation_family=estimation_family,
            weighted=self._is_weighted_mode,
            calibration_mode=self._calibration_mode,
            test_batch_signature=test_batch_signature,
        )

    def _resolve_weights(
        self,
        x: np.ndarray,
        *,
        refit_weights: bool,
        test_batch_signature: BatchSignature,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve calibration/test weights for the current batch."""
        if not self._is_weighted_mode or self.weight_estimator is None:
            return None

        if refit_weights:
            self.weight_estimator.fit(self._calibration_samples, x)
            self._prepared_weight_batch_size = len(x)
            if self.verify_prepared_batch_content:
                self._prepared_weight_batch_signature = test_batch_signature
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
            self._prepared_weight_batch_signature != test_batch_signature
        ):
            raise ValueError(
                "Prepared weights do not match current batch content. "
                "Call prepare_weights_for(batch) again or use refit_weights=True."
            )
        return self.weight_estimator.get_weights()

    def select(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        alpha: float = 0.05,
        pruning: Pruning = Pruning.DETERMINISTIC,
        seed: int | None = None,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Construct evidence and select anomalies from one fixed test batch.

        This is the single-call batch workflow. It combines
        ``compute_p_values()`` with Benjamini-Hochberg in standard mode or
        weighted conformalized selection in weighted mode. Validity still
        depends on the assumptions of both the p-value construction and the
        selected multiple-testing procedure.

        With DerandomizedSplits, this instead constructs per-split e-values,
        averages them uniformly, and applies e-BH once. Configure alpha_bh and
        tie_seed on that strategy. Automatic ties use a separate fitting-derived
        seed. Selection populates last_selection_result and clears last_result.

        Args:
            x: New data instances for anomaly estimation.
            alpha: Nominal FDR target in ``(0, 1)``. Defaults to ``0.05``.
            pruning: Pruning strategy for weighted FDR control. Ignored in
                standard (unweighted) mode. Defaults to
                ``Pruning.DETERMINISTIC``.
            seed: Optional random seed for weighted randomized pruning modes.
                When ``None``, falls back to detector ``seed``. Ignored in
                standard mode and deterministic pruning mode.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Ignored in standard mode. Defaults to True.

        Returns:
            Boolean selection mask of shape ``(n_test,)``. ``True`` entries are
            the selected anomaly discoveries. Returns a pandas Series when the
            input is a DataFrame or Series.

        Examples:
            Standard workflow (no weight estimator):

            ```python
            import numpy as np
            from sklearn.ensemble import IsolationForest

            from nonconform import ConformalDetector, Split

            rng = np.random.default_rng(42)
            x_reference = rng.normal(size=(300, 2))
            x_test = np.vstack(
                [rng.normal(size=(38, 2)), rng.normal(loc=5.0, size=(2, 2))]
            )
            detector = ConformalDetector(
                detector=IsolationForest(random_state=42),
                strategy=Split(n_calib=0.25),
                score_polarity="higher_is_normal",
                seed=42,
            ).fit(x_reference)
            selected = detector.select(x_test, alpha=0.10)
            print("Selected indices:", np.flatnonzero(selected))
            ```

            Weighted workflow:

            ```python
            import numpy as np
            from sklearn.ensemble import IsolationForest

            from nonconform import (
                ConformalDetector,
                Split,
                logistic_weight_estimator,
            )

            rng = np.random.default_rng(7)
            x_reference = rng.normal(size=(400, 2))
            x_test = rng.normal(loc=0.5, size=(40, 2))
            detector = ConformalDetector(
                detector=IsolationForest(random_state=7),
                strategy=Split(n_calib=0.25),
                weight_estimator=logistic_weight_estimator(),
                score_polarity="higher_is_normal",
                seed=7,
            ).fit(x_reference)
            selected = detector.select(x_test, alpha=0.10)
            print("Number selected:", int(selected.sum()))
            ```
        """
        self._last_selection_result = None
        if self.strategy._uses_e_values:
            self._last_result = None
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        from nonconform.fdr import weighted_false_discovery_control

        x_array, index = _as_numpy_with_index(x)
        if self.strategy._uses_e_values:
            scores = self._score_models(x_array)
            selection = self.strategy._select_e_values(
                scores, self._calibration_set, alpha=alpha
            )
            self._last_selection_result = selection
            mask = selection.selected.copy()
            if index is not None:
                return pd.Series(mask, index=index, name="selected")
            return mask

        self.compute_p_values(x_array, refit_weights=refit_weights)
        result = self._last_result
        if result is None or result.p_values is None:
            raise RuntimeError(
                "Internal error: select() expected p-values after compute_p_values()."
            )

        if self._is_weighted_mode:
            selection_seed = self.seed if seed is None else seed
            mask = weighted_false_discovery_control(
                result=result,
                alpha=alpha,
                pruning=pruning,
                seed=_derive_wcs_pruning_seed(selection_seed),
            )
        else:
            p_values = np.asarray(result.p_values, dtype=float)
            mask = false_discovery_control(p_values, method="bh") <= alpha

        if index is not None:
            return pd.Series(mask, index=index, name="selected")
        return mask

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
            self._prepared_weight_batch_signature = batch_signature(x)
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

        Clears last_selection_result. With DerandomizedSplits, raw aggregation
        is diagnostic only and the last_result snapshot has calib_scores=None;
        select() instead uses each model's separate score/calibration pair.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Defaults to True.

        Returns:
            Aggregated raw anomaly scores.
        """
        self._last_selection_result = None
        if self.strategy._uses_e_values:
            self._last_result = None
        x_array, index = _as_numpy_with_index(x)
        test_batch_signature = batch_signature(x_array)
        estimates = self._aggregate_scores(x_array)
        weights = self._resolve_weights(
            x_array,
            refit_weights=refit_weights,
            test_batch_signature=test_batch_signature,
        )
        calib_weights, test_weights = weights if weights else (None, None)

        result = ConformalResult(
            p_values=None,
            test_scores=estimates.copy(),
            calib_scores=(
                None if self.strategy._uses_e_values else self._calibration_set.copy()
            ),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata={},
        )
        result._provenance = self._result_provenance(test_batch_signature)
        self._last_result = result
        if index is not None:
            return pd.Series(estimates, index=index, name="score")
        return estimates

    def compute_p_value(self, x: pd.Series | np.ndarray) -> float:
        """Return one value from the configured estimation strategy.

        Unavailable for DerandomizedSplits; use select() on a fixed test batch
        and inspect last_selection_result instead.

        This is a single-sample convenience wrapper around
        :meth:`compute_p_values`. It updates :attr:`last_result` with the
        corresponding one-row result and does not update the calibration set.
        With the default ``Empirical`` estimator, the returned value is a
        rank-based conformal p-value. Other estimators define their own
        interpretation and assumptions.

        Args:
            x: One-dimensional feature vector with the same number of features
                used during fitting or detached calibration.

        Returns:
            The observation's p-value or score-tail estimate as a Python float.

        Raises:
            NotFittedError: If the detector has not been fitted or calibrated.
            RuntimeError: If weighted conformal mode is enabled. Density-ratio
                estimation requires a representative test batch.
            ValueError: If ``x`` is not one-dimensional or its feature count does
                not match the fitted data.

        Note:
            Repeated single-sample calls are batch-equivalent only when detector
            scoring and p-value estimation are sample-wise and deterministic.
            Randomized tie-breaking can produce different values from one batch
            call.
        """
        self._require_p_value_strategy()
        if not self.is_fitted:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")
        if self._is_weighted_mode:
            raise RuntimeError(
                "compute_p_value() is unavailable in weighted mode because "
                "density-ratio estimation requires a representative test batch. "
                "Use compute_p_values() with that batch instead."
            )

        x_array = np.asarray(x)
        if x_array.ndim != 1:
            raise ValueError(
                "x must be a one-dimensional feature vector; "
                f"got shape {x_array.shape}."
            )

        expected_features = self._n_features_in
        if expected_features is None:
            raise RuntimeError(
                "Fitted feature count is unavailable. Refit or recalibrate the "
                "detector before calling compute_p_value()."
            )
        if x_array.shape[0] != expected_features:
            raise ValueError(
                f"x has {x_array.shape[0]} features, but this ConformalDetector "
                f"was fitted with {expected_features} features."
            )

        p_values = self.compute_p_values(x_array[np.newaxis, :])
        return float(np.asarray(p_values).item())

    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return values from the configured estimation strategy for new data.

        Unavailable for DerandomizedSplits; use select() and inspect
        last_selection_result instead.

        Args:
            x: New data instances for anomaly estimation.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Defaults to True.

        Returns:
            P-values or score-tail estimates. Pandas input produces a Series
            named ``"p_value"``; NumPy input produces an ndarray.
        """
        self._require_p_value_strategy()
        x_array, index = _as_numpy_with_index(x)
        test_batch_signature = batch_signature(x_array)
        estimates = self._aggregate_scores(x_array)
        weights = self._resolve_weights(
            x_array,
            refit_weights=refit_weights,
            test_batch_signature=test_batch_signature,
        )
        calib_weights, test_weights = weights if weights else (None, None)

        p_values = self.estimation.compute_p_values(
            estimates, self._calibration_set, weights
        )

        metadata = self._result_metadata()
        if hasattr(self.estimation, "get_metadata"):
            meta = self.estimation.get_metadata()
            if meta:
                metadata.update(meta)

        result = ConformalResult(
            p_values=p_values.copy(),
            test_scores=estimates.copy(),
            calib_scores=self._calibration_set.copy(),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata=metadata,
        )
        result._provenance = self._result_provenance(test_batch_signature)
        self._last_result = result
        if index is not None:
            return pd.Series(p_values, index=index, name="p_value")
        return p_values

    @property
    def detector_set(self) -> list[AnomalyDetector]:
        """Returns a copy of the list of trained detector models."""
        return self._detector_set.copy()

    @property
    def calibration_set(self) -> np.ndarray:
        """Return a copy of calibration scores.

        DerandomizedSplits returns a matrix shaped
        (n_repetitions, n_calibration), with one row per retained model.
        Other strategies return their existing one-dimensional score array.
        """
        return self._calibration_set.copy()

    @property
    def calibration_samples(self) -> np.ndarray:
        """Returns a copy of the calibration samples (weighted mode only)."""
        return self._calibration_samples.copy()

    @property
    def last_result(self) -> ConformalResult | None:
        """Return the most recent raw-score or p-value snapshot.

        DerandomizedSplits selection instead populates last_selection_result
        and clears this snapshot. Its raw-score snapshots have no pooled
        calibration scores.
        """
        return None if self._last_result is None else self._last_result.copy()

    @property
    def last_selection_result(self) -> EValueSelectionResult | None:
        """Return a defensive snapshot of the latest e-value selection.

        None before selection, after fitting or raw scoring, and for existing
        p-value selection workflows. Arrays remain read-only in the snapshot.
        """
        if self._last_selection_result is None:
            return None
        from dataclasses import replace

        return replace(self._last_selection_result)

    def _require_p_value_strategy(self) -> None:
        """Reject p-value operations for a strategy that constructs e-values."""
        self._last_selection_result = None
        if self.strategy._uses_e_values:
            self._last_result = None
            raise ValueError(
                "DerandomizedSplits constructs e-values, not p-values. "
                "Use select(x, alpha=...) and inspect last_selection_result."
            )

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
