"""Prepare detector configuration without mutating an existing estimator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

from nonconform.adapters import _adapt_with_score_polarity
from nonconform.scoring import Empirical
from nonconform.weighting import IdentityWeightEstimator

from .config import set_params
from .constants import ScorePolarity
from .math_utils import AggregationMethod, normalize_aggregation_method
from .validation import validate_optional_seed

if TYPE_CHECKING:
    from nonconform.resampling import BaseStrategy
    from nonconform.scoring import BaseEstimation
    from nonconform.structures import AnomalyDetector
    from nonconform.weighting import BaseWeightEstimator

NESTED_COMPONENTS = ("detector", "strategy", "estimation", "weight_estimator")


@dataclass(frozen=True, slots=True)
class DetectorConfiguration:
    """Validated runtime components and their constructor-facing snapshots."""

    detector: AnomalyDetector
    strategy: BaseStrategy
    estimation: BaseEstimation
    weight_estimator: BaseWeightEstimator | None
    aggregation: AggregationMethod
    score_polarity: ScorePolarity
    seed: int | None
    verbose: bool
    verify_prepared_batch_content: bool
    alpha: float
    weighted: bool
    constructor_params: dict[str, Any]


def prepare_configuration(
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
) -> DetectorConfiguration:
    """Resolve a complete candidate before the detector commits any changes."""
    seed = validate_optional_seed("seed", seed)
    alpha = _validate_alpha(alpha)
    runtime_strategy = _resolve_strategy(strategy)
    if not isinstance(verbose, bool):
        raise TypeError(
            f"verbose must be a boolean value, got {type(verbose).__name__}."
        )
    if not isinstance(verify_prepared_batch_content, bool):
        raise TypeError("verify_prepared_batch_content must be a boolean value.")
    normalized_aggregation = normalize_aggregation_method(aggregation)
    weighted = weight_estimator is not None and not isinstance(
        weight_estimator, IdentityWeightEstimator
    )
    if runtime_strategy._uses_e_values:
        if weighted:
            raise ValueError("DerandomizedSplits does not support weighting.")
        if estimation is not None and type(estimation) is not Empirical:
            raise ValueError(
                "DerandomizedSplits constructs e-values directly; p-value "
                "estimation is unused. Omit estimation or use ordinary Empirical()."
            )

    # Constructor values are kept as passed, as scikit-learn's get_params and
    # clone expect; the runtime components below are independent copies.
    constructor_params = dict(
        {
            "detector": detector,
            "strategy": strategy,
            "estimation": estimation,
            "weight_estimator": weight_estimator,
            "aggregation": aggregation,
            "score_polarity": score_polarity,
            "seed": seed,
            "verbose": verbose,
            "verify_prepared_batch_content": verify_prepared_batch_content,
            "alpha": alpha,
        }
    )
    normalized_detector, resolved_polarity = _adapt_with_score_polarity(
        detector, score_polarity
    )
    runtime_detector = set_params(deepcopy(normalized_detector), seed)
    runtime_strategy = deepcopy(runtime_strategy)
    runtime_estimation = estimation if estimation is not None else Empirical()
    if seed is not None and hasattr(runtime_estimation, "set_seed"):
        runtime_estimation.set_seed(seed)
    runtime_weight_estimator = weight_estimator
    if (
        seed is not None
        and runtime_weight_estimator is not None
        and hasattr(runtime_weight_estimator, "set_seed")
    ):
        runtime_weight_estimator.set_seed(seed)
    return DetectorConfiguration(
        detector=runtime_detector,
        strategy=runtime_strategy,
        estimation=runtime_estimation,
        weight_estimator=runtime_weight_estimator,
        aggregation=normalized_aggregation,
        score_polarity=resolved_polarity,
        seed=seed,
        verbose=verbose,
        verify_prepared_batch_content=verify_prepared_batch_content,
        alpha=alpha,
        weighted=weighted,
        constructor_params=constructor_params,
    )


def _validate_alpha(alpha: Any) -> float:
    """Require the predict() FDR level to be a probability in (0, 1)."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a float in (0, 1), got {alpha!r}.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    return alpha


def _resolve_strategy(strategy: Any) -> BaseStrategy:
    """Default to ``Split()`` and explain common strategy mistakes."""
    from nonconform.resampling import BaseStrategy, Split

    if strategy is None:
        return Split()
    if isinstance(strategy, str):
        raise TypeError(
            f"strategy must be a strategy object, not the string {strategy!r}. "
            "Pass for example strategy=Split(), CrossValidation(k=5), or "
            "JackknifeBootstrap(), or omit it to use Split()."
        )
    if isinstance(strategy, type):
        if issubclass(strategy, BaseStrategy):
            raise TypeError(
                f"strategy must be an instance: pass {strategy.__name__}() "
                f"instead of {strategy.__name__}."
            )
        raise TypeError(f"strategy must be a strategy instance, got {strategy!r}.")
    if not hasattr(strategy, "fit_calibrate"):
        raise TypeError(
            "strategy must be a nonconform strategy such as Split(), "
            f"got {type(strategy).__name__}."
        )
    return strategy


def updated_parameters(
    current: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    """Apply updates to a candidate, leaving the current parameters intact.

    Top-level values are stored as passed. A component with nested updates is
    copied first, so a rejected update cannot change the current parameters.
    """
    updated = dict(current)
    nested: dict[str, dict[str, Any]] = {}
    for key, value in updates.items():
        if "__" in key:
            component, nested_key = key.split("__", 1)
            if component not in NESTED_COMPONENTS:
                raise ValueError(f"Invalid parameter {component!r}.")
            nested.setdefault(component, {})[nested_key] = value
        elif key in updated:
            updated[key] = value
        else:
            raise ValueError(f"Invalid parameter {key!r} for ConformalDetector.")
    for name, params in nested.items():
        component = updated[name]
        if component is None or not hasattr(component, "set_params"):
            raise ValueError(f"Cannot set nested parameters for {name!r}.")
        component = deepcopy(component)
        component.set_params(**params)
        updated[name] = component
    return updated
