"""Core data structures and protocols for nonconform.

This module provides the fundamental types used throughout the package:

Classes:
    AnomalyDetector: Protocol defining the detector interface.
    ConformalResult: Container for conformal inference outputs.
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

    A supported PyOD model, a recognized scikit-learn estimator, or a custom
    object can be used with nonconform when it provides this interface. The
    object must also support shallow and deep copying because resampling
    strategies create independent detector replicas.

    Required methods:
        fit: Train the detector on data
        decision_function: Compute anomaly scores
        get_params: Retrieve detector parameters
        set_params: Configure detector parameters

    Examples:
        ```python
        from sklearn.ensemble import IsolationForest

        from nonconform.structures import AnomalyDetector

        detector: AnomalyDetector = IsolationForest(random_state=42)
        print(isinstance(detector, AnomalyDetector))
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

        Score direction is detector-specific. Pass the corresponding
        ``score_polarity`` to :class:`~nonconform.detector.ConformalDetector`
        when it cannot be inferred safely.

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

    This dataclass holds the latest p-values or score-tail estimates, raw scores,
    and optional weights produced by a detector call.

    Attributes:
        p_values: Values produced by the configured estimation strategy, or None
            when only scores were requested. With ``Empirical``, these are
            rank-based conformal p-values.
        test_scores: Aggregated, anomalous-higher scores for test instances.
        calib_scores: Anomalous-higher scores for the calibration set.
        test_weights: Importance weights for test instances (weighted mode only).
        calib_weights: Importance weights for calibration instances.
        metadata: Method metadata, including the strategy, estimator, and
            weighted-mode marker for p-value computations.

    Examples:
        ```python
        import numpy as np

        from nonconform.structures import ConformalResult

        result = ConformalResult(
            p_values=np.array([0.50, 0.02]),
            test_scores=np.array([0.1, 2.4]),
            calib_scores=np.array([-0.2, 0.0, 0.3, 0.8]),
            metadata={"nonconform": {"weighted": False}},
        )
        print(result.p_values)
        print(result.metadata)
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
