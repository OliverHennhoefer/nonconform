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
    >>> p_values = detector.compute_p_values(X_test)

    Weighted conformal prediction:

    >>> from nonconform import logistic_weight_estimator
    >>> detector = ConformalDetector(
    ...     detector=IForest(),
    ...     strategy=Split(n_calib=0.2),
    ...     weight_estimator=logistic_weight_estimator(),
    ... )
"""

__version__ = "0.98.4"
__author__ = "Oliver Hennhoefer"
__email__ = "oliver.hennhoefer@mail.de"

from nonconform.detector import ConformalDetector

# Calibration strategies
from nonconform.resampling import (
    CrossValidation,
    JackknifeBootstrap,
    Split,
)

# P-value estimation
from nonconform.scoring import (
    Empirical,
    Probabilistic,
)

# Weight estimation
from nonconform.weighting import (
    forest_weight_estimator,
    logistic_weight_estimator,
)

__all__ = [  # noqa: RUF022
    "ConformalDetector",
    "Split",
    "CrossValidation",
    "JackknifeBootstrap",
    "Empirical",
    "Probabilistic",
    "logistic_weight_estimator",
    "forest_weight_estimator",
]
