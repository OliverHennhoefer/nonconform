"""Prepare detector configuration without mutating an existing estimator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from nonconform.adapters import (
    adapt,
    apply_score_polarity,
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
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
    weighted: bool
    constructor_params: dict[str, Any]


def prepare_configuration(
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
) -> DetectorConfiguration:
    """Resolve a complete candidate before the detector commits any changes."""
    validate_optional_seed("seed", seed)
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
    if strategy._uses_e_values:
        if weighted:
            raise ValueError("DerandomizedSplits does not support weighting.")
        if estimation is not None and type(estimation) is not Empirical:
            raise ValueError(
                "DerandomizedSplits constructs e-values directly; p-value "
                "estimation is unused. Omit estimation or use ordinary Empirical()."
            )

    constructor_params = deepcopy(
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
        }
    )
    adapted_detector = adapt(detector)
    resolved_polarity = (
        resolve_implicit_score_polarity(adapted_detector)
        if score_polarity is None
        else resolve_score_polarity(adapted_detector, score_polarity)
    )
    normalized_detector = apply_score_polarity(adapted_detector, resolved_polarity)
    runtime_detector = set_params(deepcopy(normalized_detector), seed)
    runtime_strategy = deepcopy(strategy)
    runtime_estimation = estimation if estimation is not None else Empirical()
    if seed is not None and hasattr(runtime_estimation, "set_seed"):
        runtime_estimation.set_seed(seed)
    if (
        seed is not None
        and weight_estimator is not None
        and hasattr(weight_estimator, "set_seed")
    ):
        weight_estimator.set_seed(seed)
    return DetectorConfiguration(
        detector=runtime_detector,
        strategy=runtime_strategy,
        estimation=runtime_estimation,
        weight_estimator=weight_estimator,
        aggregation=normalized_aggregation,
        score_polarity=resolved_polarity,
        seed=seed,
        verbose=verbose,
        verify_prepared_batch_content=verify_prepared_batch_content,
        weighted=weighted,
        constructor_params=constructor_params,
    )


def updated_parameters(
    current: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    """Apply nested updates to owned candidates, leaving stored snapshots intact."""
    updated = deepcopy(current)
    nested: dict[str, dict[str, Any]] = {}
    for key, value in updates.items():
        if "__" in key:
            component, nested_key = key.split("__", 1)
            if component not in NESTED_COMPONENTS:
                raise ValueError(f"Invalid parameter {component!r}.")
            nested.setdefault(component, {})[nested_key] = value
        elif key in updated:
            updated[key] = deepcopy(value)
        else:
            raise ValueError(f"Invalid parameter {key!r} for ConformalDetector.")
    for name, params in nested.items():
        component = updated[name]
        if component is None or not hasattr(component, "set_params"):
            raise ValueError(f"Cannot set nested parameters for {name!r}.")
        component.set_params(**params)
    return updated
