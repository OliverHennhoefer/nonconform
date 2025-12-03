from typing import Any, Protocol, Self, runtime_checkable

import numpy as np


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
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the anomaly detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            Self: The fitted detector instance.
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
            Self: The detector instance.
        """
        ...
