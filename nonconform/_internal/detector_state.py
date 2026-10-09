"""Owned, complete state for a fitted detector and one prepared weight batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .provenance import BatchSignature, CalibrationMode

if TYPE_CHECKING:
    from nonconform.structures import AnomalyDetector


def as_feature_batch(
    values: np.ndarray, *, n_features: int | None = None
) -> np.ndarray:
    """Require a two-dimensional batch with a consistent feature count."""
    batch = np.asarray(values)
    if batch.ndim != 2 or batch.shape[1] == 0:
        raise ValueError("x must be a two-dimensional batch with features.")
    if n_features is not None and batch.shape[1] != n_features:
        raise ValueError(
            "x must be a two-dimensional batch with the fitted feature count."
        )
    return batch


@dataclass(frozen=True, slots=True)
class FittedCalibration:
    """Publish the fitted models, scores, and their provenance together."""

    models: tuple[AnomalyDetector, ...]
    scores: np.ndarray
    samples: np.ndarray
    n_features: int
    mode: CalibrationMode


@dataclass(frozen=True, slots=True)
class PreparedWeights:
    """Own weights bound to the batch that produced them."""

    calibration: np.ndarray
    test: np.ndarray
    batch_size: int
    signature: BatchSignature | None

    def copy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return weights without exposing the prepared batch's arrays."""
        return self.calibration.copy(), self.test.copy()
