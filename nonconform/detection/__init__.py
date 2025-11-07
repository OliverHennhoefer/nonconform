"""Conformal anomaly detection estimators.

This module provides the core conformal anomaly detection classes that wrap
PyOD detectors with uncertainty quantification capabilities.
"""

from .base import BaseConformalDetector
from .conformal import ConformalDetector

__all__ = [
    "BaseConformalDetector",
    "ConformalDetector",
]
