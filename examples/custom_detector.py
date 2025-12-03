"""Example: Using a custom detector with nonconform.

Demonstrates how to implement the AnomalyDetector protocol for use with
ConformalDetector without depending on PyOD.
"""

from typing import Any, Self

import numpy as np
from scipy.stats import false_discovery_control
from sklearn.datasets import make_blobs

from nonconform.detection import ConformalDetector
from nonconform.strategy import Split


class MahalanobisDetector:
    """Simple Mahalanobis distance-based anomaly detector.

    Computes anomaly scores based on Mahalanobis distance from the
    training data mean, accounting for feature correlations.

    This detector implements the AnomalyDetector protocol required
    by nonconform:
    - fit(X, y=None) -> Self
    - decision_function(X) -> np.ndarray
    - get_params(deep=True) -> dict
    - set_params(**params) -> Self
    """

    def __init__(self, random_state: int | None = None):
        """Initialize the detector.

        Args:
            random_state: Random seed (unused, but included for API consistency).
        """
        self.random_state = random_state
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit the detector on training data.

        Computes the mean and inverse covariance matrix of the training data.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            Self: The fitted detector instance.
        """
        self._mean = np.mean(X, axis=0)
        # Add small regularization for numerical stability
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
        self._cov_inv = np.linalg.inv(cov)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples (further from
        the training distribution in Mahalanobis distance).

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        if self._mean is None or self._cov_inv is None:
            msg = "Detector must be fitted before calling decision_function"
            raise RuntimeError(msg)
        diff = X - self._mean
        return np.sqrt(np.sum(diff @ self._cov_inv * diff, axis=1))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this detector.

        Args:
            deep: If True, return parameters for sub-objects.

        Returns:
            Parameter names mapped to their values.
        """
        return {"random_state": self.random_state}

    def set_params(self, **params: Any) -> Self:
        """Set parameters for this detector.

        Args:
            **params: Detector parameters.

        Returns:
            Self: The detector instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    # Generate synthetic data
    # Normal training data: single cluster
    X_train, _ = make_blobs(n_samples=500, centers=1, random_state=42)

    # Test data: normal samples + anomalies
    X_test_normal, _ = make_blobs(n_samples=80, centers=1, random_state=123)
    X_anomalies = np.random.default_rng(42).uniform(-6, 6, (20, 2))
    X_test = np.vstack([X_test_normal, X_anomalies])
    y_test = np.hstack([np.zeros(80), np.ones(20)])

    print("Custom Detector Example")
    print("=" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)} ({int(y_test.sum())} anomalies)")
    print()

    # Create conformal detector with custom Mahalanobis detector
    detector = ConformalDetector(
        detector=MahalanobisDetector(random_state=42),
        strategy=Split(n_calib=0.3),
        seed=42,
    )

    # Fit on normal training data
    detector.fit(X_train)

    # Get p-values for test data
    p_values = detector.predict(X_test)

    # Apply FDR control at 10% level
    decisions = false_discovery_control(p_values, method="bh") <= 0.1

    # Evaluate results
    true_positives = np.sum(decisions & (y_test == 1))
    false_positives = np.sum(decisions & (y_test == 0))
    total_discoveries = decisions.sum()

    print("Results with FDR control at 10%:")
    print(f"  Discoveries: {total_discoveries}")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    if total_discoveries > 0:
        print(f"  Empirical FDR: {false_positives / total_discoveries:.2f}")
    print(f"  Recall: {true_positives / y_test.sum():.2f}")
