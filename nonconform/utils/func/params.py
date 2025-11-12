"""Manages and configures anomaly detection models from the PyOD library.

This module provides utilities for setting up PyOD detector models,
including handling a list of models that are restricted or unsupported
for use with conformal anomaly detection.

Attributes:
    forbidden_model_list (list[type[BaseDetector]]): A list of PyOD detector
        classes that are considered unsupported or restricted for use by
        the `set_params` function. These models are not suitable for
        conformal anomaly detection due to their specific design requirements.
"""

import sys

from pyod.models.base import BaseDetector
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.sos import SOS

forbidden_model_list: list[type[BaseDetector]] = [
    CBLOF,
    COF,
    RGraph,
    Sampling,
    SOS,
]


def _set_params(
    detector: BaseDetector,
    seed: int | None,
    random_iteration: bool = False,
    iteration: int | None = None,
) -> BaseDetector:
    """Configure a PyOD detector with specific default and dynamic parameters.

    **Internal use only.** This function modifies PyOD detector instances by setting
    common parameters for one-class conformal prediction: contamination to zero,
    n_jobs to use all cores, and random_state for reproducibility.

    Args:
        detector: The PyOD detector instance to configure.
        seed: The base random seed for reproducibility.
        random_iteration: If True and iteration is provided, creates varying
            random_state per iteration.
        iteration: Current iteration number for dynamic random_state generation.

    Returns:
        The configured detector instance with updated parameters.

    Raises:
        ValueError: If the detector class is in the forbidden_model_list.

    Note:
        Forbidden models (CBLOF, COF, RGraph, Sampling, SOS) require clustering
        or grouping which is incompatible with one-class training.
    """
    if detector.__class__ in forbidden_model_list:
        forbidden_names = ", ".join([cls.__name__ for cls in forbidden_model_list])
        raise ValueError(
            f"{detector.__class__.__name__} is not supported for conformal prediction. "
            f"Forbidden detectors: {forbidden_names}. "
            f"These models require clustering or grouping which is incompatible with "
            f"one-class training. Use detectors like IForest, HBOS, or ECOD instead."
        )

    # Set contamination to the smallest possible float for one-class classification
    if "contamination" in detector.get_params():
        detector.set_params(contamination=sys.float_info.min)

    # Utilize all available cores if n_jobs parameter exists
    if "n_jobs" in detector.get_params():
        detector.set_params(n_jobs=-1)

    # Set random_state for reproducibility when a seed is provided
    if seed is not None and "random_state" in detector.get_params():
        if random_iteration and iteration is not None:
            # Create a reproducible but varying seed per iteration
            # Ensure the result is within the typical 32-bit unsigned int range
            # for random seeds.
            dynamic_seed = hash((iteration, seed)) % (2**32)
            detector.set_params(random_state=dynamic_seed)
        else:
            detector.set_params(random_state=seed)

    return detector
