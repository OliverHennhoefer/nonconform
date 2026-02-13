"""Core data structures and protocols for nonconform.

This module provides the fundamental types used throughout the package:

Classes:
    AnomalyDetector: Protocol defining the detector interface.
    ConformalResult: Container for conformal prediction outputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np


def _array_summary(arr: np.ndarray | None) -> str:
    """Return compact summary for ndarray fields."""
    if arr is None:
        return "None"
    return f"array(shape={arr.shape}, dtype={arr.dtype})"


@runtime_checkable
class AnomalyDetector(Protocol):
    """Protocol defining the interface for anomaly detectors.

    Any detector (PyOD, sklearn-compatible, or custom) can be used with
    nonconform by implementing this protocol.

    Required methods:
        fit: Train the detector on data
        decision_function: Compute anomaly scores
        get_params: Retrieve detector parameters
        set_params: Configure detector parameters

    The detector must be copyable (support copy.copy and copy.deepcopy).

    Examples:
        ```python
        # Any PyOD detector works automatically
        from pyod.models.iforest import IForest

        detector: AnomalyDetector = IForest()


        # Custom detector implementing the protocol
        class MyDetector:
            def fit(self, X, y=None): ...
            def decision_function(self, X): ...
            def get_params(self, deep=True): ...
            def set_params(self, **params): ...


        detector: AnomalyDetector = MyDetector()
        ```
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the anomaly detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            The fitted detector instance.
        """
        ...

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this detector.

        Args:
            deep: If True, return parameters for sub-objects.

        Returns:
            Parameter names mapped to their values.
        """
        ...

    def set_params(self, **params: Any) -> Self:
        """Set parameters for this detector.

        Args:
            **params: Detector parameters.

        Returns:
            The detector instance.
        """
        ...


@dataclass(slots=True)
class ConformalResult:
    """Snapshot of detector outputs for downstream procedures.

    This dataclass holds all outputs from a conformal prediction, including
    p-values, raw scores, and optional weights for weighted conformal.

    Attributes:
        p_values: Conformal p-values for test instances (None when unavailable).
        test_scores: Non-conformity scores for the test instances (raw predictions).
        calib_scores: Non-conformity scores for the calibration set.
        test_weights: Importance weights for test instances (weighted mode only).
        calib_weights: Importance weights for calibration instances.
        metadata: Optional dictionary with extra data (debug info, timings, etc.).

    Examples:
        ```python
        p_values = detector.compute_p_values(X_test)
        result = detector.last_result
        print(result.p_values)  # Access p-values
        print(result.metadata)  # Access optional metadata
        ```
    """

    p_values: np.ndarray | None = None
    test_scores: np.ndarray | None = None
    calib_scores: np.ndarray | None = None
    test_weights: np.ndarray | None = None
    calib_weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return concise notebook-friendly result summary."""
        metadata_keys = sorted(self.metadata.keys())
        if len(metadata_keys) > 5:
            metadata_repr = f"{metadata_keys[:5]}... (total={len(metadata_keys)})"
        else:
            metadata_repr = str(metadata_keys)
        return (
            "ConformalResult("
            f"p_values={_array_summary(self.p_values)}, "
            f"test_scores={_array_summary(self.test_scores)}, "
            f"calib_scores={_array_summary(self.calib_scores)}, "
            f"test_weights={_array_summary(self.test_weights)}, "
            f"calib_weights={_array_summary(self.calib_weights)}, "
            f"metadata_keys={metadata_repr})"
        )

    def copy(self) -> ConformalResult:
        """Return a copy with arrays and metadata fully duplicated.

        Returns:
            A new ConformalResult with copied arrays and deep-copied metadata.
        """

        def _copy_arr(arr: np.ndarray | None) -> np.ndarray | None:
            return arr.copy() if arr is not None else None

        return ConformalResult(
            p_values=_copy_arr(self.p_values),
            test_scores=_copy_arr(self.test_scores),
            calib_scores=_copy_arr(self.calib_scores),
            test_weights=_copy_arr(self.test_weights),
            calib_weights=_copy_arr(self.calib_weights),
            metadata=deepcopy(self.metadata),
        )


__all__ = [
    "AnomalyDetector",
    "ConformalResult",
]
