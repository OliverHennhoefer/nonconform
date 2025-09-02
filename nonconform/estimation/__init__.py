"""Conformal anomaly detection estimators.

This module provides the core conformal anomaly detection classes that wrap
PyOD detectors with uncertainty quantification capabilities.
"""

from .base import BaseConformalDetector
from .standard_conformal import StandardConformalDetector
from .weighted_conformal import WeightedConformalDetector

__all__ = [
    "BaseConformalDetector",
    "StandardConformalDetector",
    "WeightedConformalDetector",
]
