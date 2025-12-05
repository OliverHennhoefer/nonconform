"""nonconform: Conformal Anomaly Detection with Uncertainty Quantification.

This package provides statistically rigorous anomaly detection with p-values
and error control metrics like False Discovery Rate (FDR). Supports PyOD
detectors, sklearn-compatible detectors, and custom detectors.

Main Components:
    - Conformal detectors with uncertainty quantification
    - Calibration strategies for different data scenarios
    - Weighted conformal detection for covariate shift
    - Statistical utilities and FDR control

Logging Control:
    By default, INFO level messages and above are shown.
    Control verbosity with standard Python logging:

        import logging
        logging.getLogger("nonconform").setLevel(logging.ERROR)  # Silence warnings
        logging.getLogger("nonconform").setLevel(logging.DEBUG)  # Enable debug

Examples:
    Basic usage with PyOD detector:

    >>> from pyod.models.iforest import IForest
    >>> from nonconform import ConformalDetector, Split
    >>> detector = ConformalDetector(detector=IForest(), strategy=Split(n_calib=0.2))
    >>> detector.fit(X_train)
    >>> p_values = detector.predict(X_test)

    Weighted conformal prediction:

    >>> from nonconform import logistic_weight_estimator
    >>> detector = ConformalDetector(
    ...     detector=IForest(),
    ...     strategy=Split(n_calib=0.2),
    ...     weight_estimator=logistic_weight_estimator(),
    ... )
"""

__version__ = "0.98.0"
__author__ = "Oliver Hennhoefer"
__email__ = "oliver.hennhoefer@mail.de"

# Core detector
# Enums and Utilities
# External adapters
from nonconform.adapters import PYOD_AVAILABLE, PyODAdapter, adapt
from nonconform.detector import BaseConformalDetector, ConformalDetector

# FDR control
from nonconform.fdr import (
    weighted_bh,
    weighted_false_discovery_control,
)

# Calibration strategies
from nonconform.resampling import (
    BaseStrategy,
    CrossValidation,
    JackknifeBootstrap,
    Split,
)

# P-value estimation
from nonconform.scoring import (
    BaseEstimation,
    Empirical,
    Probabilistic,
)

# Data structures
from nonconform.structures import AnomalyDetector, ConformalResult

# Weight estimation
from nonconform.weighting import (
    BaseWeightEstimator,
    BootstrapBaggedWeightEstimator,
    IdentityWeightEstimator,
    SklearnWeightEstimator,
    forest_weight_estimator,
    logistic_weight_estimator,
)

from ._internal import (
    Aggregation,
    Distribution,
    Kernel,
    Pruning,
    aggregate,
    false_discovery_rate,
    statistical_power,
)

__all__ = [
    "PYOD_AVAILABLE",
    "Aggregation",
    "AnomalyDetector",
    "BaseConformalDetector",
    "BaseEstimation",
    "BaseStrategy",
    "BaseWeightEstimator",
    "BootstrapBaggedWeightEstimator",
    "ConformalDetector",
    "ConformalResult",
    "CrossValidation",
    "Distribution",
    "Empirical",
    "IdentityWeightEstimator",
    "JackknifeBootstrap",
    "Kernel",
    "Probabilistic",
    "Pruning",
    "PyODAdapter",
    "SklearnWeightEstimator",
    "Split",
    "adapt",
    "aggregate",
    "false_discovery_rate",
    "forest_weight_estimator",
    "logistic_weight_estimator",
    "statistical_power",
    "weighted_bh",
    "weighted_false_discovery_control",
]
