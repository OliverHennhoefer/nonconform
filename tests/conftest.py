"""Shared test fixtures and utilities for nonconform tests.

This module provides reusable test fixtures including mock detectors that
implement the AnomalyDetector protocol for unit testing.
"""

from copy import deepcopy
from typing import Any, Self

import numpy as np
import pytest


class MockDetector:
    """Mock anomaly detector implementing the AnomalyDetector protocol.

    This mock detector is used throughout the test suite to test conformal
    prediction components without relying on external detector implementations.

    Args:
        scores: Optional fixed scores to return from decision_function.
            If None, returns random standard normal values.

    Attributes:
        _fitted: Whether fit() has been called.
        _training_size: Number of samples seen during fit() (set after fitting).
        _params: Dictionary of detector parameters.
    """

    def __init__(self, scores: np.ndarray | None = None) -> None:
        self._fitted = False
        self._training_size: int | None = None
        self._scores = scores if scores is not None else None
        self._params = {"random_state": None, "n_jobs": 1, "contamination": 0.1}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the detector (records training size).

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            The fitted detector instance.
        """
        self._fitted = True
        self._training_size = len(X)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,). Returns fixed scores if
            provided during initialization, otherwise random standard normal.
        """
        n_samples = len(X)
        if self._scores is not None:
            # Tile or truncate fixed scores to match input size
            if len(self._scores) >= n_samples:
                return self._scores[:n_samples].copy()
            return np.tile(self._scores, (n_samples // len(self._scores)) + 1)[
                :n_samples
            ]
        # Return random scores if none provided
        rng = np.random.default_rng()
        return rng.standard_normal(n_samples)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get detector parameters.

        Args:
            deep: If True, return parameters for sub-objects.

        Returns:
            Parameter names mapped to their values.
        """
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        """Set detector parameters.

        Args:
            **params: Detector parameters.

        Returns:
            The detector instance.
        """
        self._params.update(params)
        return self

    def __copy__(self) -> "MockDetector":
        """Create a shallow copy."""
        new = MockDetector(self._scores.copy() if self._scores is not None else None)
        new._params = self._params.copy()
        new._fitted = self._fitted
        new._training_size = self._training_size
        return new

    def __deepcopy__(self, memo: dict) -> "MockDetector":
        """Create a deep copy."""
        new = MockDetector(
            deepcopy(self._scores, memo) if self._scores is not None else None
        )
        new._params = deepcopy(self._params, memo)
        new._fitted = self._fitted
        new._training_size = self._training_size
        return new


@pytest.fixture
def mock_detector() -> MockDetector:
    """Provide a fresh MockDetector instance for testing.

    Returns:
        A new MockDetector with default configuration.
    """
    return MockDetector()


@pytest.fixture
def mock_detector_with_scores() -> MockDetector:
    """Provide a MockDetector with fixed scores for deterministic testing.

    Returns:
        A MockDetector configured to return fixed scores [0.1, 0.5, 0.9].
    """
    return MockDetector(scores=np.array([0.1, 0.5, 0.9]))
