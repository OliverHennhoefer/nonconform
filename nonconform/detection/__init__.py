"""Conformal anomaly detection estimators.

This module provides the core conformal anomaly detection classes that wrap
anomaly detectors with uncertainty quantification capabilities.

The AnomalyDetector protocol defines the interface that custom detectors
must implement to be compatible with nonconform.
"""

from .base import BaseConformalDetector
from .conformal import ConformalDetector
from .protocol import AnomalyDetector

__all__ = [
    "AnomalyDetector",
    "BaseConformalDetector",
    "ConformalDetector",
]
