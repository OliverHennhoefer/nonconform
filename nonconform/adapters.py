"""External detector adapters for nonconform.

This module provides adapters for integrating external anomaly detection
libraries, particularly PyOD, with nonconform.

Functions:
    adapt: Adapt any detector to the AnomalyDetector protocol.

Classes:
    PyODAdapter: Wrapper for PyOD detectors to ensure protocol compliance.
"""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import Any, Self

import numpy as np

from nonconform.structures import AnomalyDetector

logger = logging.getLogger(__name__)

# Soft dependency handling for PyOD
try:
    from pyod.models.base import BaseDetector as PyODBaseDetector

    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    PyODBaseDetector = None


def adapt(detector: Any) -> AnomalyDetector:
    """Adapt a detector to the AnomalyDetector protocol.

    PyOD detectors are automatically wrapped if PyOD is installed.
    Protocol-compliant detectors are passed through unchanged.
    Non-compliant detectors raise a clear error.

    Args:
        detector: Detector instance to adapt.

    Returns:
        Protocol-compliant detector.

    Raises:
        TypeError: If detector doesn't conform to protocol.
        ImportError: If PyOD detector provided but PyOD not installed.

    Examples:
        ```python
        from pyod.models.iforest import IForest
        from nonconform.adapters import adapt

        # Automatically wraps PyOD detector
        detector = adapt(IForest())

        # Custom detectors pass through if protocol-compliant
        detector = adapt(my_custom_detector)
        ```
    """
    # Already protocol-compliant
    if isinstance(detector, AnomalyDetector):
        return detector

    # PyOD detector - wrap if available
    if PYOD_AVAILABLE and isinstance(detector, PyODBaseDetector):
        return PyODAdapter(detector)

    # Looks like PyOD but not installed
    if not PYOD_AVAILABLE and _looks_like_pyod(detector):
        raise ImportError(
            "Detector appears to be a PyOD detector, but PyOD is not installed. "
            "Install with: pip install pyod"
        )

    # Check for required methods
    required_methods = ["fit", "decision_function", "get_params", "set_params"]
    missing_methods = [m for m in required_methods if not hasattr(detector, m)]

    if missing_methods:
        raise TypeError(
            f"Detector must implement AnomalyDetector protocol. "
            f"Missing methods: {', '.join(missing_methods)}"
        )

    return detector


def _looks_like_pyod(detector: Any) -> bool:
    """Check if detector looks like a PyOD detector based on module path."""
    module = type(detector).__module__
    return module is not None and module.startswith("pyod.")


class PyODAdapter:
    """Adapter wrapping PyOD detectors to ensure protocol compliance.

    This is a thin wrapper that delegates all calls to the underlying
    PyOD detector. It exists to guarantee protocol conformance.

    The adapter is copyable and supports all standard PyOD operations
    through attribute delegation.

    Examples:
        ```python
        from pyod.models.iforest import IForest
        from nonconform.adapters import PyODAdapter

        # Direct usage (usually not needed - use adapt() instead)
        adapter = PyODAdapter(IForest())
        adapter.fit(X_train)
        scores = adapter.decision_function(X_test)
        ```
    """

    def __init__(self, detector: Any) -> None:
        """Initialize adapter.

        Args:
            detector: PyOD detector instance.

        Raises:
            ImportError: If PyOD is not installed.
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is not installed. Install with: pip install pyod")

        self._detector = detector

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the detector.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.

        Returns:
            The fitted adapter instance.
        """
        self._detector.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).
        """
        return self._detector.decision_function(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get detector parameters.

        Args:
            deep: If True, return parameters for sub-objects.

        Returns:
            Parameter names mapped to their values.
        """
        return self._detector.get_params(deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Set detector parameters.

        Args:
            **params: Detector parameters.

        Returns:
            The adapter instance.
        """
        self._detector.set_params(**params)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped detector.

        Note:
            Guards against recursion during pickle unpickling by checking
            if _detector exists in __dict__ before delegating.
        """
        if "_detector" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._detector, name)

    def __repr__(self) -> str:
        """String representation."""
        return f"PyODAdapter({self._detector!r})"

    def __copy__(self) -> PyODAdapter:
        """Create a shallow copy of the adapter."""
        return PyODAdapter(copy(self._detector))

    def __deepcopy__(self, memo: dict) -> PyODAdapter:
        """Create a deep copy of the adapter."""
        return PyODAdapter(deepcopy(self._detector, memo))


__all__ = [
    "PYOD_AVAILABLE",
    "PyODAdapter",
    "adapt",
]
