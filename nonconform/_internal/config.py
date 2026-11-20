"""Detector parameter configuration utilities.

This module provides utilities for setting up detector models with
standard parameters for conformal anomaly detection.
"""

from __future__ import annotations

import inspect
import logging
import sys
from typing import TYPE_CHECKING, Any

from .random_utils import derive_seed

if TYPE_CHECKING:
    from nonconform.structures import AnomalyDetector

logger = logging.getLogger(__name__)

_RANDOM_STATE_ALIASES = ["random_state", "seed", "random_seed"]
_N_JOBS_ALIASES = ["n_jobs", "n_threads", "num_workers"]
_CONTAMINATION_ALIASES = ["contamination"]


def set_params(
    detector: AnomalyDetector,
    seed: int | None,
    random_iteration: bool = False,
    iteration: int | None = None,
) -> AnomalyDetector:
    """Configure a detector with parameters for conformal prediction.

    Sets common parameters following sklearn conventions with fallback aliases:
    - contamination: Set to minimum for one-class training (optional)
    - n_jobs/n_threads: Set to -1 to use all cores when the value is still
      the detector's default (optional)
    - random_state/seed: Set for reproducibility (only if seed provided)

    Args:
        detector: Detector instance to configure.
        seed: Base random seed for reproducibility.
        random_iteration: If True and iteration provided, creates varying
            random_state per iteration.
        iteration: Current iteration number for dynamic random_state.

    Returns:
        Configured detector instance.

    Note:
        For a scikit-learn ``Pipeline``, ``contamination`` and parallelism
        are set on the final step only (``isolationforest__n_jobs``), while a
        seed reaches every step that exposes ``random_state``. Missing
        contamination, parallelism, or random-state parameters are logged at
        debug level. Detectors without a seed parameter are usually
        deterministic, so a missing one is not a warning.
    """
    params = detector.get_params()
    final_prefix = _final_step_prefix(detector)

    _try_set_contamination(detector, params, final_prefix)
    _try_set_parallelism(detector, params, final_prefix)

    if seed is not None:
        if random_iteration and iteration is not None:
            seed = derive_seed(iteration, seed)
        _set_random_state(detector, params, seed)

    return detector


def matching_param_keys(params: dict[str, Any], alias: str) -> list[str]:
    """Return the parameter keys that configure ``alias``.

    A top-level key wins. Otherwise nested keys whose last ``__`` component is
    ``alias`` are returned, so every step of a scikit-learn ``Pipeline`` that
    exposes the parameter is configured.
    """
    if alias in params:
        return [alias]
    suffix = f"__{alias}"
    return [key for key in params if key.endswith(suffix)]


def _final_step_prefix(detector: Any) -> str:
    """Return the nested-parameter prefix of a Pipeline's final step.

    Adapters that wrap a detector in ``_detector`` are looked through. The
    result is ``""`` for a detector that is not a scikit-learn ``Pipeline``.
    """
    from sklearn.pipeline import Pipeline

    while "_detector" in getattr(detector, "__dict__", {}):
        detector = detector.__dict__["_detector"]
    prefix = ""
    while isinstance(detector, Pipeline) and detector.steps:
        name, detector = detector.steps[-1]
        prefix += f"{name}__"
    return prefix


def _set_alias(
    detector: AnomalyDetector,
    params: dict[str, Any],
    alias: str,
    value: Any,
    final_prefix: str | None = None,
) -> bool:
    """Set the keys matching ``alias``; return whether any key matched.

    With ``final_prefix``, nested keys are limited to that Pipeline step, so
    settings such as ``n_jobs`` reach only the scoring estimator and not
    preprocessing steps like ``ColumnTransformer``.
    """
    if final_prefix is None:
        keys = matching_param_keys(params, alias)
    elif alias in params:
        keys = [alias]
    else:
        nested = f"{final_prefix}{alias}"
        keys = [nested] if final_prefix and nested in params else []
    if not keys:
        return False
    detector.set_params(**dict.fromkeys(keys, value))
    return True


def _try_set_contamination(
    detector: AnomalyDetector, params: dict[str, Any], final_prefix: str
) -> None:
    """Try to set contamination to minimum for one-class training."""
    for alias in _CONTAMINATION_ALIASES:
        if _set_alias(detector, params, alias, sys.float_info.min, final_prefix):
            logger.debug(f"Set {alias}={sys.float_info.min} for one-class training")
            return

    logger.debug("Detector has no contamination parameter (acceptable)")


def _parallelism_keys(
    params: dict[str, Any], alias: str, final_prefix: str
) -> list[str]:
    """Return the keys ``_set_alias`` would set for a parallelism alias."""
    if alias in params:
        return [alias]
    nested = f"{final_prefix}{alias}"
    return [nested] if final_prefix and nested in params else []


def _constructor_default(owner: Any, name: str) -> Any:
    """Return the default of ``name`` in ``type(owner).__init__``, if declared."""
    try:
        parameter = inspect.signature(type(owner).__init__).parameters.get(name)
    except (TypeError, ValueError):
        return inspect.Parameter.empty
    if parameter is None:
        return inspect.Parameter.empty
    return parameter.default


def _left_at_default(detector: Any, params: dict[str, Any], key: str) -> bool:
    """Whether the parallelism parameter ``key`` still holds its default.

    ``None`` always counts as unset. Otherwise the value counts as unset only
    when it equals the default declared by the constructor of the object that
    owns the parameter (the Pipeline step for nested keys). A value that
    differs, or one whose default cannot be found, was chosen by the user.
    """
    value = params[key]
    if value is None:
        return True
    if "__" in key:
        owner_key, name = key.rsplit("__", 1)
        owner = params.get(owner_key)
    else:
        name = key
        owner = detector
        while "_detector" in getattr(owner, "__dict__", {}):
            owner = owner.__dict__["_detector"]
    if owner is None:
        return False
    default = _constructor_default(owner, name)
    return default is not inspect.Parameter.empty and value == default


def _try_set_parallelism(
    detector: AnomalyDetector, params: dict[str, Any], final_prefix: str
) -> None:
    """Use all cores unless the user chose the detector's parallelism.

    A parallelism value the user set is kept, so a detector configured with
    ``n_jobs=2`` (for example inside ``GridSearchCV`` or on a shared machine)
    is not switched to every core. Only a value left at its default is set
    to ``-1``.
    """
    for alias in _N_JOBS_ALIASES:
        keys = _parallelism_keys(params, alias, final_prefix)
        if not keys:
            continue
        if not all(_left_at_default(detector, params, key) for key in keys):
            logger.debug(f"Kept user-set {alias}={params[keys[0]]!r}")
            return
        _set_alias(detector, params, alias, -1, final_prefix)
        logger.debug(f"Set {alias}=-1 for parallelism")
        return

    logger.debug("Detector has no parallelism parameter (acceptable)")


def _set_random_state(
    detector: AnomalyDetector, params: dict[str, Any], seed: int
) -> None:
    """Set random state for reproducibility."""
    for alias in _RANDOM_STATE_ALIASES:
        if _set_alias(detector, params, alias, seed):
            logger.debug(f"Set {alias}={seed} for reproducibility")
            return

    tried_names = ", ".join(_RANDOM_STATE_ALIASES)
    logger.debug(
        f"Detector has no random_state parameter. Tried: {tried_names}. "
        f"Reproducibility cannot be guaranteed if detector uses randomness. "
        f"This is acceptable for deterministic detectors."
    )


__all__ = [
    "matching_param_keys",
    "set_params",
]
