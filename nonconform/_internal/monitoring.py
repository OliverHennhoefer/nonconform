"""Private bridge state for sequential exchangeability monitoring."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import NotFittedError

from nonconform.structures import AnomalyDetector


@dataclass(slots=True, frozen=True)
class _SplitMonitoringSnapshot:
    """Owned fitted scoring state extracted from a Split detector."""

    detector: AnomalyDetector
    calibration_scores: np.ndarray
    n_features_in: int


def _snapshot_split_detector(detector: Any) -> _SplitMonitoringSnapshot:
    """Validate and copy the fitted state needed by a sequential monitor."""
    from nonconform.detector import ConformalDetector
    from nonconform.resampling import Split

    if not isinstance(detector, ConformalDetector):
        raise TypeError("detector must be a ConformalDetector instance.")
    if not detector.is_fitted:
        raise NotFittedError("ConformalDetector must be fitted first.")
    if not isinstance(detector.strategy, Split):
        raise ValueError("from_split_detector requires a Split strategy.")
    if detector._is_weighted_mode:
        raise ValueError("from_split_detector does not support weighted mode.")

    fitted_models = detector.detector_set
    if len(fitted_models) != 1:
        raise ValueError("from_split_detector requires exactly one fitted model.")
    if detector._n_features_in is None:
        raise RuntimeError(
            "Fitted feature count is unavailable. Refit or recalibrate the detector."
        )

    try:
        owned_detector = deepcopy(fitted_models[0])
    except Exception as exc:
        raise TypeError("detector must support deep copying.") from exc

    return _SplitMonitoringSnapshot(
        detector=owned_detector,
        calibration_scores=detector.calibration_set,
        n_features_in=int(detector._n_features_in),
    )
