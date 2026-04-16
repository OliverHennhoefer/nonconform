"""Integrative conformal detection with labeled outliers.

This subpackage provides a parallel API surface for conformal out-of-distribution
testing when both labeled inliers and labeled outliers are available.
"""

from nonconform.integrative.detector import IntegrativeConformalDetector
from nonconform.integrative.models import IntegrativeModel
from nonconform.integrative.strategies import IntegrativeSplit, TransductiveCVPlus

__all__ = [
    "IntegrativeConformalDetector",
    "IntegrativeModel",
    "IntegrativeSplit",
    "TransductiveCVPlus",
]
