import logging
from typing import Any, Self

import numpy as np

from nonconform.detection.protocol import AnomalyDetector

logger = logging.getLogger(__name__)

try:
    from pyod.models.base import BaseDetector as PyODBaseDetector

    _PYOD_AVAILABLE = True
except ImportError:
    _PYOD_AVAILABLE = False
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
    """
    if isinstance(detector, AnomalyDetector):
        return detector

    if _PYOD_AVAILABLE and PyODBaseDetector and isinstance(detector, PyODBaseDetector):
        return PyODAdapter(detector)

    if not _PYOD_AVAILABLE and _looks_like_pyod(detector):
        msg = (
            "Detector appears to be a PyOD detector, but PyOD is not installed. "
            "Install with: pip install nonconform[pyod]"
        )
        raise ImportError(msg)

    required_methods = ["fit", "decision_function", "get_params", "set_params"]
    missing_methods = [m for m in required_methods if not hasattr(detector, m)]

    if missing_methods:
        msg = (
            f"Detector must implement AnomalyDetector protocol. "
            f"Missing methods: {', '.join(missing_methods)}"
        )
        raise TypeError(msg)

    return detector


def _looks_like_pyod(detector: Any) -> bool:
    """Check if detector looks like a PyOD detector."""
    module = type(detector).__module__
    return module is not None and module.startswith("pyod.")


class PyODAdapter:
    """Adapter wrapping PyOD detectors to ensure protocol compliance.

    This is a thin wrapper that delegates all calls to the underlying
    PyOD detector. It exists to guarantee protocol conformance.
    """

    def __init__(self, detector: Any) -> None:
        """Initialize adapter.

        Args:
            detector: PyOD detector instance.
        """
        if not _PYOD_AVAILABLE:
            msg = "PyOD is not installed. Install with: pip install nonconform[pyod]"
            raise ImportError(msg)

        self._detector = detector

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Train the detector."""
        self._detector.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self._detector.decision_function(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get detector parameters."""
        return self._detector.get_params(deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Set detector parameters."""
        self._detector.set_params(**params)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped detector."""
        return getattr(self._detector, name)

    def __repr__(self) -> str:
        """String representation."""
        return f"PyODAdapter({self._detector!r})"

    def __copy__(self) -> "PyODAdapter":
        """Create a shallow copy of the adapter."""
        from copy import copy

        return PyODAdapter(copy(self._detector))

    def __deepcopy__(self, memo: dict) -> "PyODAdapter":
        """Create a deep copy of the adapter."""
        from copy import deepcopy

        return PyODAdapter(deepcopy(self._detector, memo))
