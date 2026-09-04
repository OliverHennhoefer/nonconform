"""Conformal evidence for anomaly detection and change monitoring.

``nonconform`` wraps PyOD, recognized scikit-learn, and custom anomaly scorers
for two primary workflows: batch conformal p-values with false discovery rate
control, and sequential randomized ranks with exchangeability martingales.
Guarantees depend on the assumptions of the selected workflow.

Main Components:
    - Batch conformal p-values and FDR-controlled selection
    - Split and resampling calibration strategies
    - Weighted conformal inference and WCS under covariate shift
    - Post-hoc simultaneous FDP bounds
    - Sequential conformal monitoring via ``nonconform.monitoring`` and
      ``nonconform.martingales``

Logging Control:
    Configure package progress and warnings with standard Python logging:

        import logging
        logging.getLogger("nonconform").setLevel(logging.WARNING)

Examples:
    FDR-controlled batch selection:

    >>> import numpy as np
    >>> from sklearn.ensemble import IsolationForest
    >>> from nonconform import ConformalDetector, Split
    >>> rng = np.random.default_rng(42)
    >>> x_reference = rng.normal(size=(300, 3))
    >>> x_test = rng.normal(size=(10, 3))
    >>> detector = ConformalDetector(
    ...     detector=IsolationForest(random_state=42),
    ...     strategy=Split(n_calib=0.3),
    ...     seed=42,
    ... )
    >>> _ = detector.fit(x_reference)
    >>> mask = detector.select(x_test, alpha=0.05)
    >>> mask.shape
    (10,)
"""

__version__ = "1.1.1"
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
