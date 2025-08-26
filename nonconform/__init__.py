"""nonconform: Conformal Anomaly Detection with Uncertainty Quantification.

This package provides statistically rigorous anomaly detection with p-values
and error control metrics like False Discovery Rate (FDR) for PyOD detectors.

Main Components:
- Conformal detectors with uncertainty quantification
- Calibration strategies for different data scenarios
- Extreme Value Theory enhanced detection
- Statistical utilities and data handling tools
"""

__version__ = "0.9.165"
__author__ = "Oliver Hennhoefer"
__email__ = "oliver.hennhoefer@mail.de"

from . import estimation, strategy, utils

__all__ = ["estimation", "strategy", "utils"]
