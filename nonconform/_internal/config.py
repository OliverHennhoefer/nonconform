"""Detector parameter configuration utilities.

This module provides utilities for setting up detector models with
standard parameters for conformal anomaly detection.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

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
    - n_jobs/n_threads: Set to -1 to use all cores (optional)
    - random_state/seed: Set for reproducibility (required if seed provided)

    Args:
        detector: Detector instance to configure.
        seed: Base random seed for reproducibility.
        random_iteration: If True and iteration provided, creates varying
            random_state per iteration.
        iteration: Current iteration number for dynamic random_state.

    Returns:
        Configured detector instance.

    Note:
        Logs warnings if optional parameters (contamination, n_jobs) unavailable.
    """
    params = detector.get_params()

    _try_set_contamination(detector, params)
    _try_set_parallelism(detector, params)

    if seed is not None:
        if random_iteration and iteration is not None:
            dynamic_seed = hash((iteration, seed)) % (2**32)
            _set_random_state(detector, params, dynamic_seed)
        else:
            _set_random_state(detector, params, seed)

    return detector


def _try_set_contamination(detector: AnomalyDetector, params: dict[str, Any]) -> None:
    """Try to set contamination to minimum for one-class training."""
    for alias in _CONTAMINATION_ALIASES:
        if alias in params:
            detector.set_params(**{alias: sys.float_info.min})
            logger.debug(f"Set {alias}={sys.float_info.min} for one-class training")
            return

    logger.debug("Detector has no contamination parameter (acceptable)")


def _try_set_parallelism(detector: AnomalyDetector, params: dict[str, Any]) -> None:
    """Try to set parallelism to use all cores."""
    for alias in _N_JOBS_ALIASES:
        if alias in params:
            detector.set_params(**{alias: -1})
            logger.debug(f"Set {alias}=-1 for parallelism")
            return

    logger.debug("Detector has no parallelism parameter (acceptable)")


def _set_random_state(
    detector: AnomalyDetector, params: dict[str, Any], seed: int
) -> None:
    """Set random state for reproducibility."""
    for alias in _RANDOM_STATE_ALIASES:
        if alias in params:
            detector.set_params(**{alias: seed})
            logger.debug(f"Set {alias}={seed} for reproducibility")
            return

    tried_names = ", ".join(_RANDOM_STATE_ALIASES)
    logger.warning(
        f"Detector has no random_state parameter. Tried: {tried_names}. "
        f"Reproducibility cannot be guaranteed if detector uses randomness. "
        f"This is acceptable for deterministic detectors."
    )


__all__ = [
    "set_params",
]
