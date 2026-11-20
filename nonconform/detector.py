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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from nonconform.resampling import CrossValidation, JackknifeBootstrap, Split
from nonconform.scoring import ConditionalEmpirical, Empirical
from nonconform.structures import AnomalyDetector, ConformalResult
from nonconform.weighting import BaseWeightEstimator

from ._internal import (
    Pruning,
    ScorePolarity,
    TieBreakMode,
    aggregate,
    ensure_numpy_array,
)
from ._internal.detector_config import (
    NESTED_COMPONENTS,
    prepare_configuration,
    updated_parameters,
)
from ._internal.detector_state import (
    FittedCalibration,
    PreparedWeights,
    as_array,
    as_feature_batch,
    check_feature_names,
    feature_names_of,
    take_rows,
)
from ._internal.groups import (
    CalibrationGroups,
    as_group_labels,
    group_conditional_p_values,
)
from ._internal.p_value_selection import (
    adjusted_p_values,
    has_classical_floor,
    normalize_method,
    validate_storey_scope,
    warn_if_floor_blocks_discovery,
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
    from nonconform.fdr import EValueSelectionResult, FDPCertificate, FPRCertificate
    from nonconform.resampling import BaseStrategy
    from nonconform.scoring import BaseEstimation

_WCS_PRUNING_SEED_DOMAIN = 0x574353  # ASCII "WCS"


def _safe_copy(arr: np.ndarray | None) -> np.ndarray | None:
    """Return a copy of array or None if None."""
    return None if arr is None else arr.copy()


def _derive_wcs_pruning_seed(seed: int | None) -> int | None:
    """Derive a deterministic WCS-pruning stream distinct from ``seed``."""
    if seed is None:
        return None
    seed_sequence = np.random.SeedSequence([seed, _WCS_PRUNING_SEED_DOMAIN])
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _as_batch_with_index(
    x: pd.DataFrame | pd.Series | np.ndarray,
) -> tuple[pd.DataFrame | np.ndarray, pd.Index | None]:
    """Return the batch for the wrapped detector and an optional pandas index.

    DataFrames pass through unchanged so the wrapped detector sees column
    names and dtypes. A pandas Series is a single-feature batch.
    """
    if isinstance(x, pd.Series):
        return x.to_frame(), x.index
    if isinstance(x, pd.DataFrame):
        return x, x.index
    if sparse.issparse(x):
        return x.tocsr(), None
    return np.asarray(x), None


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

    Subclasses must implement ``fit()``, ``compute_p_values()``, and
    ``score_samples()``. ``calibrate()`` is optional and raises
    ``NotImplementedError`` unless overridden.

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


class ConformalDetector(OutlierMixin, BaseConformalDetector, BaseEstimator):
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

    Passing ``groups`` to ``fit()`` or ``calibrate()`` makes p-values
    group-conditional (Mondrian): each test point is ranked only among
    calibration scores from its own group, and p-value methods then require the
    test labels. ``select()`` still applies BH once to the whole batch.

    The class is a scikit-learn ``BaseEstimator`` and ``OutlierMixin``.
    ``predict()`` labels a whole test batch with ``select(x, alpha=alpha)``,
    returning ``-1`` for selected anomalies and ``1`` otherwise, so the detector
    can be the last step of a ``Pipeline``. DataFrames reach the wrapped
    detector unchanged, and column names are checked at inference.

    With ``DerandomizedSplits``, ``fit()`` retains repeated model/calibration
    pairs and ``select()`` uniformly aggregates per-split e-values before e-BH.
    Inspect ``last_selection_result`` for evidence and selection diagnostics.
    P-value methods and detached calibration are unavailable for this strategy.

    Args:
        detector: Anomaly detector (PyOD, sklearn-compatible, or custom), or
            a scoring function ``f(X) -> scores`` treated as already fitted.
        strategy: The conformal strategy for fitting, calibration, and evidence
            construction. DerandomizedSplits selects through e-values and e-BH.
            Defaults to None, which uses ``Split()``.
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
        alpha: FDR target in ``(0, 1)`` used by ``predict()`` and
            ``decision_function()``. ``select()`` takes its own ``alpha``.
            Defaults to 0.05.

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

    # scikit-learn's common checks cannot build this class without a detector.
    _required_parameters: ClassVar[list[str]] = ["detector"]

    def __init__(
        self,
        detector: Any,
        strategy: BaseStrategy | None = None,
        estimation: BaseEstimation | None = None,
        weight_estimator: BaseWeightEstimator | None = None,
        aggregation: str = "median",
        score_polarity: ScorePolarity
        | Literal["auto", "higher_is_anomalous", "higher_is_normal"]
        | None = None,
        seed: int | None = None,
        verbose: bool = False,
        verify_prepared_batch_content: bool = True,
        alpha: float = 0.05,
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
            alpha=alpha,
        )

    def __getstate__(self) -> Any:
        """Pickle like a plain object, so subclasses may define __slots__."""
        return object.__getstate__(self)

    def __setstate__(
        self, state: dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any]]
    ) -> None:
        """Restore pickled state, rejecting the pre-2.0 fitted-state layout."""
        # Subclasses defining __slots__ pickle state as (dict_state, slot_state).
        slot_state: dict[str, Any] = {}
        if isinstance(state, tuple):
            state, slot_state = state
            state = state or {}
            slot_state = slot_state or {}
        if "_calibration" not in state or "_constructor_params" not in state:
            raise TypeError(
                "This ConformalDetector was pickled with nonconform 1.x, whose "
                "fitted-state layout is not supported in 2.0. Recreate the "
                "detector and fit or calibrate it again."
            )
        state = dict(state)
        state.pop("_sklearn_version", None)
        # Detectors pickled before alpha existed use its default.
        state.setdefault("alpha", 0.05)
        state["_constructor_params"] = {
            "alpha": 0.05,
            **state["_constructor_params"],
        }
        self.__dict__.update(state)
        for name, value in slot_state.items():
            setattr(self, name, value)

    def _reset_fit_state(self) -> None:
        """Clear all learned state derived from fit()."""
        self._calibration: FittedCalibration | None = None
        self._prepared_weights: PreparedWeights | None = None
        self._last_result: ConformalResult | None = None
        self._last_selection_result: EValueSelectionResult | None = None

    def _require_calibration(self) -> FittedCalibration:
        """Return complete learned state or reject an unfitted operation."""
        if self._calibration is None:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")
        return self._calibration

    def _publish_calibration(
        self,
        models: list[AnomalyDetector],
        scores: np.ndarray,
        samples: np.ndarray,
        *,
        n_features: int,
        mode: CalibrationMode,
        groups: CalibrationGroups | None = None,
        feature_names: np.ndarray | None = None,
    ) -> None:
        """Own and commit all fitted facts only after calibration succeeds."""
        scores = np.asarray(scores)
        if not models or scores.size == 0:
            raise ValueError("Calibration requires fitted models and nonempty scores.")
        self._calibration = FittedCalibration(
            models=tuple(models),
            scores=scores.copy(),
            samples=samples.copy(),
            n_features=n_features,
            mode=mode,
            groups=groups,
            feature_names=feature_names,
        )

    def _configure(
        self,
        *,
        detector: Any,
        strategy: BaseStrategy | None,
        estimation: BaseEstimation | None,
        weight_estimator: BaseWeightEstimator | None,
        aggregation: str,
        score_polarity: ScorePolarity
        | Literal["auto", "higher_is_anomalous", "higher_is_normal"]
        | None,
        seed: int | None,
        verbose: bool,
        verify_prepared_batch_content: bool,
        alpha: float,
    ) -> None:
        """Apply constructor parameters and reset learned state."""
        configuration = prepare_configuration(
            detector=detector,
            strategy=strategy,
            estimation=estimation,
            weight_estimator=weight_estimator,
            aggregation=aggregation,
            score_polarity=score_polarity,
            seed=seed,
            verbose=verbose,
            verify_prepared_batch_content=verify_prepared_batch_content,
            alpha=alpha,
        )
        self.detector = configuration.detector
        self.strategy = configuration.strategy
        self.estimation = configuration.estimation
        self.weight_estimator = configuration.weight_estimator
        self.aggregation = configuration.aggregation
        self._score_polarity = configuration.score_polarity
        self.seed = configuration.seed
        self.verbose = configuration.verbose
        self.verify_prepared_batch_content = configuration.verify_prepared_batch_content
        self.alpha = configuration.alpha
        self._is_weighted_mode = configuration.weighted
        self._constructor_params = configuration.constructor_params
        self._reset_fit_state()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters following sklearn conventions.

        Notes:
            - ``deep=False`` returns the constructor parameters as passed to
              ``__init__`` or ``set_params``, following scikit-learn. The
              fitted runtime components are separate copies, so changing a
              returned object affects later clones, not this fitted detector.
            - ``deep=True`` also includes nested ``component__param`` entries
              read from the current runtime components (effective/internal state),
              which may differ from originally passed constructor objects after
              adaptation/normalization.
        """
        params = dict(self._constructor_params)
        if deep:
            for name in NESTED_COMPONENTS:
                component = getattr(self, name)
                if component is None or not hasattr(component, "get_params"):
                    continue
                try:
                    nested = component.get_params(deep=True)
                except TypeError:
                    nested = component.get_params()
                params.update(
                    {f"{name}__{key}": value for key, value in nested.items()}
                )
        return params

    def set_params(self, **params: Any) -> Self:
        """Commit valid parameter updates and clear learned state.

        Rejected updates leave configuration, calibration, and prepared weights
        intact, including when several nested components are updated together.
        """
        if params:
            self._configure(**updated_parameters(self._constructor_params, params))
        return self

    def __sklearn_clone__(self) -> Self:
        """Return an unfitted clone with independent copies of the parameters.

        Like ``sklearn.base.clone``, each parameter is cloned, so the clone
        shares no estimation, weighting, or detector state with this detector.
        """
        from sklearn.base import clone

        params = self.get_params(deep=False)
        return type(self)(
            **{name: clone(value, safe=False) for name, value in params.items()}
        )

    def __repr__(self) -> str:
        """Return concise notebook-friendly detector summary."""
        calibration = self._calibration
        n_models = 0 if calibration is None else len(calibration.models)
        n_calibration = 0 if calibration is None else calibration.scores.shape[-1]
        return (
            "ConformalDetector("
            f"detector={type(self.detector).__name__}, "
            f"strategy={type(self.strategy).__name__}, "
            f"estimation={type(self.estimation).__name__}, "
            f"aggregation={self.aggregation!r}, "
            f"score_polarity={self._score_polarity.name}, "
            f"weighted_mode={self._is_weighted_mode}, "
            f"seed={self.seed}, verbose={self.verbose}, fitted={self.is_fitted}, "
            f"n_models={n_models}, n_calibration={n_calibration})"
        )

    def fit(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        *,
        groups: Any = None,
        n_jobs: int | None = None,
    ) -> Self:
        """Fit detector model(s) and compute calibration scores.

        Uses the specified strategy to train the base detector(s) and calculate
        non-conformity scores on the calibration set.

        A new fit clears previous calibration, results, and prepared weights.
        If training fails, this detector remains unfitted; fit again before
        requesting evidence.

        Args:
            x: The dataset used for fitting and calibration.
            y: Ignored. Present for sklearn API compatibility.
            groups: Optional group label for each row of ``x``, matched by
                position. With labels, p-values are group-conditional
                (Mondrian): each test point is ranked only among calibration
                scores from its own group, and p-value methods then require
                the test labels. Supported for unweighted ``Split``,
                ``CrossValidation``, and ``JackknifeBootstrap`` with
                ``Empirical(tie_break="classical")``. The base detector is
                still fitted on all groups together.
            n_jobs: Optional strategy-specific parallelism hint. Supported by
                strategies whose ``fit_calibrate`` signature includes ``n_jobs``
                (for example, ``JackknifeBootstrap``).

        Returns:
            The fitted detector instance (for method chaining).
        """
        _ = y
        self._reset_fit_state()
        x = as_feature_batch(x)
        labels = None
        if groups is not None:
            self._require_group_scope()
            labels = as_group_labels(groups, n_rows=x.shape[0])
        # Split records calibration rows only on request; groups need them.
        record_ids = self._is_weighted_mode or labels is not None
        fit_kwargs: dict[str, Any] = {
            "x": x,
            "detector": self.detector,
            "weighted": record_ids,
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
        models, scores = self.strategy.fit_calibrate(**fit_kwargs)
        ids = self.strategy.calibration_ids if record_ids else None
        samples = (
            as_array(take_rows(x, ids))
            if self._is_weighted_mode and ids
            else np.array([])
        )
        calibration_groups = None
        if labels is not None:
            rows = np.asarray(ids if ids is not None else [], dtype=np.intp)
            if rows.shape != np.shape(scores):
                raise RuntimeError(
                    "Internal error: calibration rows do not match the "
                    "calibration scores."
                )
            calibration_groups = CalibrationGroups.from_rows(labels, rows)
        self._publish_calibration(
            models,
            scores,
            samples,
            n_features=x.shape[1],
            mode=CalibrationMode.INTEGRATED,
            groups=calibration_groups,
            feature_names=feature_names_of(x),
        )
        return self

    def calibrate(
        self,
        x: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        *,
        groups: Any = None,
    ) -> Self:
        """Calibrate a pre-fitted detector on separate calibration data.

        This detached workflow is currently supported only for ``Split`` strategy,
        where a single pre-fitted model is calibrated on a dedicated dataset.

        A supported calibration attempt clears previous learned state. If it
        fails, supply valid calibration data and call ``calibrate`` again.

        Args:
            x: Calibration dataset used to compute calibration scores.
            y: Ignored. Present for sklearn API compatibility.
            groups: Optional group label for each row of ``x``, matched by
                position, for group-conditional (Mondrian) p-values. See
                :meth:`fit`.

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
                f"calibrate() is supported only with Split strategy. "
                f"Got {type(self.strategy).__name__}. "
                "Use fit(x_reference) for integrated calibration."
            )
        if groups is not None:
            self._require_group_scope()
        self._reset_fit_state()
        x = as_feature_batch(x)
        labels = None if groups is None else as_group_labels(groups, n_rows=x.shape[0])
        try:
            scores = np.asarray(self.detector.decision_function(x), dtype=float).ravel()
        except Exception as exc:
            message = str(exc).lower()
            if (
                isinstance(exc, NotFittedError)
                or "not fitted" in message
                or (isinstance(exc, AttributeError) and "has no attribute" in message)
            ):
                raise NotFittedError(
                    "Base detector is not fitted. Fit the base detector "
                    "before calling calibrate()."
                ) from exc
            raise
        if scores.shape[0] != x.shape[0]:
            raise ValueError(
                "calibration scores must have one value per calibration sample. "
                f"Got {scores.shape[0]} scores for {x.shape[0]} samples."
            )
        self._publish_calibration(
            [self.detector],
            scores,
            as_array(x) if self._is_weighted_mode else np.array([]),
            n_features=x.shape[1],
            mode=CalibrationMode.DETACHED,
            groups=(
                None
                if labels is None
                else CalibrationGroups.from_rows(labels, np.arange(x.shape[0]))
            ),
            feature_names=feature_names_of(x),
        )
        return self

    def _check_batch(self, x: Any) -> pd.DataFrame | np.ndarray:
        """Validate a test batch against the fitted feature count and names."""
        calibration = self._require_calibration()
        x = as_feature_batch(x, n_features=calibration.n_features)
        check_feature_names(
            calibration.feature_names, x, estimator_name=type(self).__name__
        )
        return x

    def _score_models(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Score one batch with every replica, preserving model order."""
        calibration = self._require_calibration()
        x = self._check_batch(x)
        models = calibration.models
        iterable = (
            tqdm(models, total=len(models), desc="Aggregation")
            if self.verbose
            else models
        )
        rows = []
        for model in iterable:
            row = np.asarray(model.decision_function(x))
            if row.shape != (x.shape[0],):
                raise ValueError("Each model must return one score per test row.")
            rows.append(row)
        return np.vstack(rows)

    def _aggregate_scores(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
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
        test_batch_signature: BatchSignature | None,
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
            calibration_mode=(
                None if self._calibration is None else self._calibration.mode
            ),
            test_batch_signature=test_batch_signature,
            tie_break=(
                getattr(self.estimation, "_tie_break", None)
                if isinstance(self.estimation, Empirical)
                else None
            ),
            grouped=(
                self._calibration is not None and self._calibration.groups is not None
            ),
        )

    def _fit_weights_for_batch(
        self,
        x: np.ndarray,
        *,
        test_batch_signature: BatchSignature | None = None,
    ) -> None:
        """Fit batch weights and record the size and optional reuse signature."""
        self._prepared_weights = None
        calibration = self._require_calibration()
        batch = as_feature_batch(x, n_features=calibration.n_features)
        x = as_array(batch)
        self.weight_estimator.fit(calibration.samples, x)
        calib_weights, test_weights = self.weight_estimator.get_weights()
        signature = (
            (
                test_batch_signature
                if test_batch_signature is not None
                else batch_signature(batch)
            )
            if self.verify_prepared_batch_content
            else None
        )
        self._prepared_weights = PreparedWeights(
            calibration=calib_weights.copy(),
            test=test_weights.copy(),
            batch_size=len(x),
            signature=signature,
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
            self._fit_weights_for_batch(x, test_batch_signature=test_batch_signature)
        prepared = self._prepared_weights
        if prepared is None:
            raise RuntimeError(
                "Weights are not prepared. Call prepare_weights_for(batch) "
                "or use refit_weights=True."
            )
        if prepared.batch_size != x.shape[0]:
            raise ValueError(
                "Prepared weights do not match current batch size. "
                "Call prepare_weights_for(batch) again or use refit_weights=True."
            )
        if (
            self.verify_prepared_batch_content
            and prepared.signature != test_batch_signature
        ):
            raise ValueError(
                "Prepared weights do not match current batch content. "
                "Call prepare_weights_for(batch) again or use refit_weights=True."
            )
        return prepared.copy_arrays()

    def select(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        alpha: float = 0.05,
        method: Literal["bh", "storey"] = "bh",
        groups: Any = None,
        pruning: Pruning = Pruning.DETERMINISTIC,
        seed: int | None = None,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Construct evidence and select anomalies from one fixed test batch.

        This is the single-call batch workflow. It combines
        ``compute_p_values()`` with Benjamini-Hochberg, or Storey's adaptive
        BH with ``method="storey"``, in standard mode, and with weighted
        conformalized selection in weighted mode. Validity still depends on the
        assumptions of both the p-value construction and the selected
        multiple-testing procedure.

        With DerandomizedSplits, this instead constructs per-split e-values,
        averages them uniformly, and applies e-BH once. Configure alpha_bh and
        tie_seed on that strategy. Automatic ties use a separate fitting-derived
        seed. Selection populates last_selection_result and clears last_result.

        Classical empirical p-values cannot fall below ``1 / (n + 1)`` for
        ``n`` calibration scores. When a standard selection with such p-values
        is empty and that floor exceeds the first step-up threshold, a warning
        on the ``nonconform.fdr`` logger reports the calibration size needed to
        discover a single isolated anomaly. With groups, ``n`` is the size of
        the largest calibration group among the test points.

        Args:
            x: New data instances for anomaly estimation.
            alpha: Nominal FDR target in ``(0, 1)``. Defaults to ``0.05``.
            method: Multiple-testing rule for standard p-value selection.
                ``"bh"`` applies Benjamini-Hochberg. ``"storey"`` applies
                Storey's adaptive BH: it estimates the null proportion from the
                p-values at or above ``lambda``, the smallest calibration-grid
                point ``k / (n + 1)`` not below 1/2, then runs BH at ``alpha``
                divided by that estimate. This can select more when anomalies
                are common and slightly less when they are rare. ``"storey"``
                requires unweighted classical conformal p-values (``Empirical``
                or ``ConditionalEmpirical`` with ``tie_break="classical"``) and
                a p-value strategy. Weighted mode uses WCS and
                DerandomizedSplits uses e-BH regardless. Defaults to ``"bh"``.
            groups: Group label for each row of ``x`` when the detector was
                fitted or calibrated with groups; required then and rejected
                otherwise. BH runs once over the whole batch of
                group-conditional p-values. ``"storey"`` does not support
                groups.
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
        method = normalize_method(method)
        if method == "storey":
            self._require_storey_scope()

        from nonconform.fdr import weighted_false_discovery_control

        x_batch, index = _as_batch_with_index(x)
        if self.strategy._uses_e_values:
            if groups is not None:
                raise ValueError(
                    "groups= is unavailable with DerandomizedSplits, which "
                    "selects with e-values."
                )
            scores = self._score_models(x_batch)
            selection = self.strategy._select_e_values(
                scores, self._require_calibration().scores, alpha=alpha
            )
            self._last_selection_result = selection
            mask = selection.selected.copy()
            if index is not None:
                return pd.Series(mask, index=index, name="selected")
            return mask

        # Pass groups only when given, so overrides without the keyword still work.
        group_kwargs = {} if groups is None else {"groups": groups}
        self.compute_p_values(x_batch, refit_weights=refit_weights, **group_kwargs)
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
            calibration = self._require_calibration()
            n_calibration = calibration.scores.shape[-1]
            mask = (
                adjusted_p_values(p_values, method=method, n_calibration=n_calibration)
                <= alpha
            )
            if (
                p_values.size
                and not mask.any()
                and has_classical_floor(result._provenance)
            ):
                if calibration.groups is not None:
                    # The smallest attainable p-value comes from the largest
                    # calibration group among the test points.
                    test_codes = calibration.groups.encode(groups, n_rows=p_values.size)
                    n_calibration = int(calibration.groups.counts[test_codes].max())
                warn_if_floor_blocks_discovery(
                    p_values,
                    alpha=alpha,
                    method=method,
                    n_calibration=n_calibration,
                    grouped=calibration.groups is not None,
                )

        if index is not None:
            return pd.Series(mask, index=index, name="selected")
        return mask

    def prepare_weights_for(self, x: pd.DataFrame | np.ndarray) -> Self:
        """Prepare weighted conformal state for a specific test batch.

        In weighted mode, this fits the weight estimator for the supplied batch
        without producing predictions. Use this for explicit state transitions in
        exploratory workflows.

        Prepared weights own their arrays independently of the estimator. A
        failed preparation clears the previous prepared batch; prepare it again
        before using ``refit_weights=False``.

        Args:
            x: Test batch for which weights should be prepared.

        Returns:
            The fitted detector instance (for method chaining).

        Raises:
            NotFittedError: If fit() has not been called.
            RuntimeError: If weighted mode is disabled.
        """
        self._require_calibration()
        if not self._is_weighted_mode or self.weight_estimator is None:
            raise RuntimeError(
                "prepare_weights_for() requires weighted mode with a weight_estimator."
            )

        self._fit_weights_for_batch(x)
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
        x_batch, index = _as_batch_with_index(x)
        test_batch_signature = batch_signature(x_batch)
        estimates = self._aggregate_scores(x_batch)
        weights = self._resolve_weights(
            x_batch,
            refit_weights=refit_weights,
            test_batch_signature=test_batch_signature,
        )
        calib_weights, test_weights = weights if weights else (None, None)

        result = ConformalResult(
            p_values=None,
            test_scores=estimates.copy(),
            calib_scores=(
                None
                if self.strategy._uses_e_values
                else self._require_calibration().scores.copy()
            ),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata={},
        )
        result._provenance = self._result_provenance(test_batch_signature)
        result._index = index
        self._last_result = result
        if index is not None:
            return pd.Series(estimates, index=index, name="score")
        return estimates

    def compute_p_value(
        self,
        x: pd.Series | np.ndarray,
        *,
        group: Any = None,
    ) -> float:
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
            group: The observation's group label. Required when the detector
                was fitted or calibrated with groups, and rejected otherwise.

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
        calibration = self._require_calibration()
        if self._is_weighted_mode:
            raise RuntimeError(
                "compute_p_value() is unavailable in weighted mode because "
                "density-ratio estimation requires a representative test batch. "
                "Use compute_p_values() with that batch instead."
            )

        if isinstance(x, pd.Series):
            # A row of a DataFrame: keep its column names for the name check.
            row = x.to_frame().T.infer_objects()
        else:
            x_batch = np.asarray(x)
            if x_batch.ndim != 1:
                raise ValueError(
                    "x must be a one-dimensional feature vector; "
                    f"got shape {x_batch.shape}."
                )
            row = x_batch[np.newaxis, :]

        expected_features = calibration.n_features
        if row.shape[1] != expected_features:
            raise ValueError(
                f"x has {row.shape[1]} features, but this ConformalDetector "
                f"was fitted with {expected_features} features."
            )

        group_kwargs: dict[str, Any] = {}
        if group is not None:
            labels = np.empty(1, dtype=object)
            labels[0] = group
            group_kwargs["groups"] = labels
        p_values = self.compute_p_values(row, **group_kwargs)
        return float(np.asarray(p_values).item())

    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        groups: Any = None,
        refit_weights: bool = True,
    ) -> np.ndarray | pd.Series:
        """Return values from the configured estimation strategy for new data.

        Unavailable for DerandomizedSplits; use select() and inspect
        last_selection_result instead.

        Args:
            x: New data instances for anomaly estimation.
            groups: Group label for each row of ``x``, matched by position.
                Required when the detector was fitted or calibrated with
                groups, and rejected otherwise. Each p-value then ranks the
                test score only among calibration scores from its group, so
                every label needs calibration scores. The snapshot's
                ``calib_scores`` still holds the scores of all groups.
            refit_weights: Whether to refit the weight estimator for this batch
                in weighted mode. Defaults to True.

        Returns:
            P-values or score-tail estimates. Pandas input produces a Series
            named ``"p_value"``; NumPy input produces an ndarray.
        """
        self._require_p_value_strategy()
        x_batch, index = _as_batch_with_index(x)
        test_codes = self._test_group_codes(x_batch, groups)
        test_batch_signature = batch_signature(x_batch)
        estimates = self._aggregate_scores(x_batch)
        weights = self._resolve_weights(
            x_batch,
            refit_weights=refit_weights,
            test_batch_signature=test_batch_signature,
        )
        calib_weights, test_weights = weights if weights else (None, None)

        calibration = self._require_calibration()
        calibration_scores = calibration.scores
        if calibration.groups is None:
            p_values = self.estimation.compute_p_values(
                estimates, calibration_scores, weights
            )
        else:
            p_values = group_conditional_p_values(
                self.estimation,
                estimates,
                test_codes,
                calibration_scores,
                calibration.groups.codes,
            )

        metadata = self._result_metadata()
        if hasattr(self.estimation, "get_metadata"):
            meta = self.estimation.get_metadata()
            if meta:
                metadata.update(meta)

        result = ConformalResult(
            p_values=p_values.copy(),
            test_scores=estimates.copy(),
            calib_scores=calibration_scores.copy(),
            test_weights=_safe_copy(test_weights),
            calib_weights=_safe_copy(calib_weights),
            metadata=metadata,
        )
        result._provenance = self._result_provenance(test_batch_signature)
        result._index = index
        self._last_result = result
        if index is not None:
            return pd.Series(p_values, index=index, name="p_value")
        return p_values

    def fdp_bounds(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        confidence: float = 0.95,
    ) -> FDPCertificate:
        """Compute p-values once and return a simultaneous FDP certificate.

        Supports unweighted empirical Split inference, including detached
        calibration. Keep the testing family fixed. Scientific exchangeability
        remains the caller's responsibility. Confidence is simultaneous coverage
        jointly over data and independent Monte Carlo sampling, not an FDR target
        or a guarantee conditional on every realized envelope.

        Uses 1000 fresh Monte Carlo draws with the default truncated
        higher-criticism envelope and threshold-specific sharpening. The fitting
        seed does not control envelope sampling. Choose advanced settings via
        :meth:`nonconform.fdr.FDPCertificate.from_p_values` before inspecting
        bounds. Do not cherry-pick repeated certificate constructions. The
        returned certificate is independent of subsequent detector operations;
        queries never resample and select() returns a NumPy mask.
        """
        from nonconform._internal.certificates import validate_scope

        if not self.is_fitted:
            raise NotFittedError("This ConformalDetector instance is not fitted yet.")
        validate_scope(self._result_provenance(None), procedure="fdp_bounds")
        self.compute_p_values(x)
        return self._last_result.fdp_bounds(confidence=confidence)

    def fpr_bounds(
        self,
        *,
        confidence: float = 0.95,
    ) -> FPRCertificate:
        """Return a simultaneous raw-score false-positive-rate certificate.

        The certificate is built from the fitted detector's calibration scores
        and does not score a test batch or mutate ``last_result``. It supports
        unweighted ``Split`` calibration, including detached calibration,
        independently of the configured p-value estimator.
        The returned certificate uses anomalous-higher scores and can be
        inverted with ``threshold_for()`` or applied with ``select()``.

        The finite-sample one-sided KS band is deterministic. Confidence is
        simultaneous coverage over the calibration draw, conditional on the
        independently fitted scoring map, not an FPR target.

        Coverage requires clean calibration and future inlier scores that are
        i.i.d. conditional on a scoring map fixed independently of calibration.
        Native scope checks do not establish these distributional assumptions;
        exchangeability alone is insufficient.

        Args:
            confidence: Simultaneous coverage probability in ``(0, 1)``.

        Returns:
            An immutable simultaneous raw-score FPR certificate.

        Raises:
            NotFittedError: If the detector has not been fitted or calibrated.
            ValueError: If the fitted construction is outside the supported
                unweighted Split scope.
            RuntimeError: If the numerical KS cutoff is invalid.
        """
        from nonconform._internal.certificates import validate_scope
        from nonconform.fdr import FPRCertificate

        calibration = self._require_calibration()
        validate_scope(self._result_provenance(None), procedure="fpr_bounds")
        return FPRCertificate.from_scores(
            calibration.scores,
            confidence=confidence,
        )

    @property
    def detector_set(self) -> list[AnomalyDetector]:
        """Returns a copy of the list of trained detector models."""
        return [] if self._calibration is None else list(self._calibration.models)

    @property
    def calibration_set(self) -> np.ndarray:
        """Return a copy of calibration scores.

        DerandomizedSplits returns a matrix shaped
        (n_repetitions, n_calibration), with one row per retained model.
        Other strategies return their existing one-dimensional score array.
        """
        return (
            np.array([])
            if self._calibration is None
            else self._calibration.scores.copy()
        )

    @property
    def calibration_samples(self) -> np.ndarray:
        """Returns a copy of the calibration samples (weighted mode only)."""
        return (
            np.array([])
            if self._calibration is None
            else self._calibration.samples.copy()
        )

    @property
    def calibration_groups(self) -> np.ndarray | None:
        """Return the group label of each calibration score, or None.

        Labels align with ``calibration_set`` for detectors fitted or
        calibrated with ``groups``; otherwise this is None.
        """
        calibration = self._calibration
        if calibration is None or calibration.groups is None:
            return None
        return calibration.groups.calibration_labels()

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

    def _require_storey_scope(self) -> None:
        """Reject Storey selection outside unweighted classical p-value scope."""
        if self.strategy._uses_e_values:
            raise ValueError(
                "method='storey' is unavailable with DerandomizedSplits, "
                "which selects with e-BH."
            )
        if self._is_weighted_mode:
            raise ValueError(
                "method='storey' supports only unweighted p-values. Weighted "
                "select() uses weighted conformalized selection instead."
            )
        validate_storey_scope(self._result_provenance(None))

    def _require_group_scope(self) -> None:
        """Reject group-conditional calibration outside its supported scope."""
        if self._is_weighted_mode:
            raise ValueError("groups= is unavailable in weighted mode.")
        if self.strategy._uses_e_values:
            raise ValueError(
                "groups= is unavailable with DerandomizedSplits, which selects "
                "with e-values."
            )
        if not isinstance(self.strategy, Split | CrossValidation | JackknifeBootstrap):
            raise ValueError(
                "groups= requires Split, CrossValidation, or JackknifeBootstrap; "
                f"got {type(self.strategy).__name__}."
            )
        if (
            not isinstance(self.estimation, Empirical)
            or isinstance(self.estimation, ConditionalEmpirical)
            or self.estimation._tie_break is not TieBreakMode.CLASSICAL
        ):
            raise ValueError(
                "groups= requires Empirical() p-values with "
                "tie_break='classical'. Randomized, conditional, and KDE-based "
                "p-values do not support groups."
            )

    def _test_group_codes(self, x: np.ndarray, groups: Any) -> np.ndarray | None:
        """Validate test labels against the calibration groups, if any."""
        calibration = self._require_calibration()
        if calibration.groups is None:
            if groups is not None:
                raise ValueError(
                    "groups= requires a detector fitted or calibrated with groups."
                )
            return None
        if groups is None:
            raise ValueError(
                "This detector was calibrated by group. Pass the test group "
                "labels with groups= (group= for compute_p_value())."
            )
        x = as_feature_batch(x, n_features=calibration.n_features)
        return calibration.groups.encode(groups, n_rows=x.shape[0])

    def _require_p_value_strategy(self) -> None:
        """Reject p-value operations for a strategy that constructs e-values."""
        self._last_selection_result = None
        if self.strategy._uses_e_values:
            self._last_result = None
            raise ValueError(
                "DerandomizedSplits constructs e-values, not p-values. "
                "Use select(x, alpha=...) and inspect last_selection_result."
            )

    @contextmanager
    def _preserve_results(self) -> Iterator[None]:
        """Keep last_result and last_selection_result across predict calls.

        scikit-learn expects predict-style methods to leave estimator state
        unchanged; use select() to inspect a batch's p-values.
        """
        saved = (self._last_result, self._last_selection_result)
        try:
            yield
        finally:
            self._last_result, self._last_selection_result = saved

    def predict(self, x: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Label a test batch with scikit-learn's outlier convention.

        Runs ``select(x, alpha=self.alpha)`` on the whole batch and returns
        ``-1`` for selected anomalies and ``1`` for the rest, so the detector
        works as the last step of a scikit-learn ``Pipeline``. The labels are
        a batch-level decision with FDR control at ``alpha``, not a per-row
        threshold: the same row can be labelled differently in another batch.
        Pass the whole batch at once, and use ``select()`` for its options
        (``method``, ``groups``, weighted pruning) or a pandas Series result.

        ``fit_predict(x)`` fits and then predicts on the same rows. Because
        rows used for calibration are then also tested, prefer separate
        reference and test data.

        Args:
            x: The test batch.

        Unlike ``select()``, it leaves ``last_result`` unchanged.

        Returns:
            Integer array of shape ``(n_test,)`` with ``-1`` for anomalies and
            ``1`` for inliers.
        """
        with self._preserve_results():
            mask = np.asarray(self.select(x, alpha=self.alpha), dtype=bool)
        return np.where(mask, -1, 1)

    def decision_function(self, x: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
        """Return a batch-level margin whose negative entries are anomalies.

        The margin is the Benjamini-Hochberg adjusted p-value minus
        ``alpha``, so ``decision_function(x) < 0`` matches
        ``predict(x) == -1``, following scikit-learn's outlier convention.
        Like ``predict``, it depends on the whole batch. It requires
        unweighted p-values; weighted mode and ``DerandomizedSplits`` select
        without adjusted p-values, so use ``predict`` or ``select`` there.

        Args:
            x: The test batch.

        Returns:
            Float array of shape ``(n_test,)``.

        Raises:
            ValueError: In weighted mode or with ``DerandomizedSplits``.
        """
        if self._is_weighted_mode or self.strategy._uses_e_values:
            raise ValueError(
                "decision_function() needs adjusted p-values, which weighted "
                "selection and DerandomizedSplits do not produce. Use predict() "
                "or select() instead."
            )
        with self._preserve_results():
            mask = np.asarray(self.select(x, alpha=self.alpha), dtype=bool)
            adjusted = self._last_result.adjusted_p_values(method="bh")
        margin = adjusted - self.alpha
        # A p-value exactly at alpha is selected; keep its margin negative.
        margin[mask & (margin >= 0)] = -np.finfo(float).eps
        return margin

    def __sklearn_tags__(self) -> Any:
        """Inherit input tags (NaN, sparse) from the wrapped detector.

        Used by scikit-learn 1.6 and later.
        """
        tags = super().__sklearn_tags__()
        try:
            from sklearn.utils import get_tags

            inner = get_tags(self._constructor_params["detector"])
        except Exception:
            return tags
        tags.input_tags.allow_nan = inner.input_tags.allow_nan
        tags.input_tags.sparse = inner.input_tags.sparse
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether the detector is fitted or calibrated."""
        return self.is_fitted

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit or calibrate."""
        if self._calibration is None:
            raise AttributeError(
                "This ConformalDetector is not fitted yet; n_features_in_ is "
                "set by fit() or calibrate()."
            )
        return self._calibration.n_features

    @property
    def feature_names_in_(self) -> np.ndarray:
        """Column names seen during fit or calibrate (DataFrame input only)."""
        names = None if self._calibration is None else self._calibration.feature_names
        if names is None:
            raise AttributeError(
                "feature_names_in_ is defined only after fitting on a DataFrame "
                "whose column names are all strings."
            )
        return names.copy()

    @property
    def score_polarity(self) -> ScorePolarity:
        """Returns the resolved score polarity convention."""
        return self._score_polarity

    @property
    def is_fitted(self) -> bool:
        """Returns whether the detector has been fitted."""
        return self._calibration is not None


__all__ = [
    "BaseConformalDetector",
    "ConformalDetector",
]
