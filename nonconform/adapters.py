"""External detector adapters for nonconform."""

from __future__ import annotations

import logging
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from nonconform._internal import ScorePolarity
from nonconform.structures import AnomalyDetector

if TYPE_CHECKING:
    ScorePolarityInput = (
        ScorePolarity
        | Literal[
            "auto",
            "higher_is_anomalous",
            "higher_is_normal",
        ]
    )
else:
    ScorePolarityInput = ScorePolarity | str

logger = logging.getLogger(__name__)

# Soft dependency handling for PyOD
try:
    from pyod.models.base import BaseDetector as PyODBaseDetector

    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    PyODBaseDetector = None


def adapt(detector: Any) -> AnomalyDetector:
    """Adapt a detector to the AnomalyDetector protocol."""
    if isinstance(detector, AnomalyDetector):
        return detector

    if PYOD_AVAILABLE and isinstance(detector, PyODBaseDetector):
        return PyODAdapter(detector)

    if not PYOD_AVAILABLE and _looks_like_pyod(detector):
        raise ImportError(
            "Detector appears to be a PyOD detector, but PyOD is not installed. "
            'Install with: pip install "nonconform[pyod]" or pip install pyod.'
        )

    required_methods = ["fit", "decision_function", "get_params", "set_params"]
    missing_methods = [m for m in required_methods if not hasattr(detector, m)]
    if missing_methods:
        raise TypeError(
            "Detector must implement AnomalyDetector protocol. "
            f"Missing methods: {', '.join(missing_methods)}"
        )

    return detector


def parse_score_polarity(score_polarity: ScorePolarityInput) -> ScorePolarity:
    """Parse score polarity input to canonical enum representation."""
    if isinstance(score_polarity, ScorePolarity):
        return score_polarity

    if isinstance(score_polarity, str):
        normalized = score_polarity.strip().lower()
        mapping = {
            "auto": ScorePolarity.AUTO,
            "higher_is_anomalous": ScorePolarity.HIGHER_IS_ANOMALOUS,
            "higher_is_normal": ScorePolarity.HIGHER_IS_NORMAL,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise ValueError(
            "Invalid score_polarity value. "
            "Use one of: 'auto', 'higher_is_anomalous', 'higher_is_normal'."
        )

    raise TypeError(
        "score_polarity must be a ScorePolarity enum or string literal "
        "('auto', 'higher_is_anomalous', 'higher_is_normal')."
    )


def _sklearn_normality_types() -> tuple[type[Any], ...]:
    """Return sklearn estimators whose scores increase with normality."""
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    return (
        IsolationForest,
        OneClassSVM,
        SGDOneClassSVM,
        LocalOutlierFactor,
        EllipticEnvelope,
    )


def _is_known_sklearn_normality_detector(detector: Any) -> bool:
    """Check whether detector is a known sklearn normality-scoring estimator."""
    return isinstance(detector, _sklearn_normality_types())


def resolve_implicit_score_polarity(detector: Any) -> ScorePolarity:
    """Resolve score polarity when users omit score_polarity.

    This pre-release default favors low-friction custom detector onboarding while
    preserving safe behavior for known detector families:
    - Known sklearn normality detectors -> HIGHER_IS_NORMAL
    - PyOD detectors -> HIGHER_IS_ANOMALOUS
    - Unknown custom detectors -> HIGHER_IS_ANOMALOUS
    """
    if _is_known_sklearn_normality_detector(detector):
        return ScorePolarity.HIGHER_IS_NORMAL
    if isinstance(detector, PyODAdapter) or _looks_like_pyod(detector):
        return ScorePolarity.HIGHER_IS_ANOMALOUS
    return ScorePolarity.HIGHER_IS_ANOMALOUS


def resolve_score_polarity(
    detector: Any,
    score_polarity: ScorePolarityInput,
) -> ScorePolarity:
    """Resolve requested score polarity in strict AUTO mode.

    Unlike ``resolve_implicit_score_polarity``, this function is intentionally
    strict for explicit ``score_polarity="auto"`` and raises for unknown
    detectors.
    """
    parsed = parse_score_polarity(score_polarity)
    if parsed is not ScorePolarity.AUTO:
        return parsed

    if isinstance(detector, PyODAdapter) or _looks_like_pyod(detector):
        return ScorePolarity.HIGHER_IS_ANOMALOUS
    if _is_known_sklearn_normality_detector(detector):
        return ScorePolarity.HIGHER_IS_NORMAL

    detector_cls = type(detector)
    detector_name = f"{detector_cls.__module__}.{detector_cls.__qualname__}"
    raise ValueError(
        "Unable to infer score polarity automatically in strict auto mode for "
        f"detector '{detector_name}'. Auto inference currently supports PyOD "
        "detectors and known sklearn normality estimators. For custom detectors, "
        "pass score_polarity='higher_is_anomalous' (recommended when larger "
        "scores mean more anomalous) or score_polarity='higher_is_normal'."
    )


def apply_score_polarity(
    detector: AnomalyDetector,
    score_polarity: ScorePolarityInput,
) -> AnomalyDetector:
    """Return detector that follows requested score polarity convention."""
    parsed = parse_score_polarity(score_polarity)
    if parsed is ScorePolarity.AUTO:
        raise ValueError(
            "score_polarity='auto' must be resolved first with resolve_score_polarity."
        )
    if parsed is ScorePolarity.HIGHER_IS_ANOMALOUS:
        return detector
    return ScorePolarityAdapter(detector=detector, score_polarity=parsed)


def _looks_like_pyod(detector: Any) -> bool:
    """Check if detector looks like a PyOD detector based on module path."""
    module = type(detector).__module__
    return module is not None and module.startswith("pyod.")


class ScorePolarityAdapter:
    """Adapter that normalizes detector score direction conventions."""

    def __init__(
        self,
        detector: AnomalyDetector,
        score_polarity: ScorePolarity,
    ) -> None:
        if score_polarity not in {
            ScorePolarity.HIGHER_IS_ANOMALOUS,
            ScorePolarity.HIGHER_IS_NORMAL,
        }:
            raise ValueError(
                "ScorePolarityAdapter requires explicit non-auto score polarity."
            )
        self._detector = detector
        self._score_polarity = score_polarity
        self._multiplier = (
            1.0 if score_polarity is ScorePolarity.HIGHER_IS_ANOMALOUS else -1.0
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit wrapped detector."""
        self._detector.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return scores transformed to anomalous-higher convention."""
        scores = np.asarray(self._detector.decision_function(X), dtype=float)
        return self._multiplier * scores

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Delegate parameter retrieval to wrapped detector."""
        return self._detector.get_params(deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Delegate parameter updates to wrapped detector."""
        self._detector.set_params(**params)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped detector."""
        if "_detector" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._detector, name)

    def __repr__(self) -> str:
        """Return debug representation."""
        return (
            "ScorePolarityAdapter("
            f"score_polarity={self._score_polarity.name}, "
            f"detector={self._detector!r})"
        )

    def __copy__(self) -> ScorePolarityAdapter:
        """Create a shallow copy."""
        return ScorePolarityAdapter(
            detector=copy(self._detector),
            score_polarity=self._score_polarity,
        )

    def __deepcopy__(self, memo: dict) -> ScorePolarityAdapter:
        """Create a deep copy."""
        return ScorePolarityAdapter(
            detector=deepcopy(self._detector, memo),
            score_polarity=self._score_polarity,
        )


class PyODAdapter:
    """Adapter wrapping PyOD detectors to ensure protocol compliance."""

    def __init__(self, detector: Any) -> None:
        """Initialize adapter for a PyOD detector."""
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is not installed. Install with: pip install pyod")
        self._detector = detector

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        """Fit wrapped detector."""
        self._detector.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores from wrapped detector."""
        return self._detector.decision_function(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Delegate parameter retrieval to wrapped detector."""
        return self._detector.get_params(deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Delegate parameter updates to wrapped detector."""
        self._detector.set_params(**params)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to wrapped detector."""
        if "_detector" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._detector, name)

    def __repr__(self) -> str:
        """Return debug representation."""
        return f"PyODAdapter({self._detector!r})"

    def __copy__(self) -> PyODAdapter:
        """Create a shallow copy."""
        return PyODAdapter(copy(self._detector))

    def __deepcopy__(self, memo: dict) -> PyODAdapter:
        """Create a deep copy."""
        return PyODAdapter(deepcopy(self._detector, memo))


__all__ = [
    "PYOD_AVAILABLE",
    "PyODAdapter",
    "ScorePolarityAdapter",
    "adapt",
    "apply_score_polarity",
    "parse_score_polarity",
    "resolve_implicit_score_polarity",
    "resolve_score_polarity",
]
