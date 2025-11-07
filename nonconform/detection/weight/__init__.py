"""Weight estimators for covariate shift in conformal prediction.

This module provides various weight estimation strategies for handling
distribution shift between calibration and test data in weighted
conformal prediction.
"""

from .base import BaseWeightEstimator
from .forest import ForestWeightEstimator
from .identity import IdentityWeightEstimator
from .logistic import LogisticWeightEstimator
from .wrapper.bagging import BootstrapBaggedWeightEstimator

__all__ = [
    "BaseWeightEstimator",
    "BootstrapBaggedWeightEstimator",
    "ForestWeightEstimator",
    "IdentityWeightEstimator",
    "LogisticWeightEstimator",
]
