"""nonconform: Conformal Anomaly Detection with Uncertainty Quantification.

This package provides statistically rigorous anomaly detection with p-values
and error control metrics like False Discovery Rate (FDR) for PyOD detectors.

Main Components:
- Conformal detectors with uncertainty quantification
- Calibration strategies for different data scenarios
- Extreme Value Theory enhanced detection
- Statistical utilities and data handling tools

Logging Control:
By default, INFO level messages and above are shown (INFO, WARNING, ERROR).
Progress bars (tqdm) are always visible.

To control logging verbosity, use standard Python logging:

    import logging

    # To silence warnings and show only errors:
    logging.getLogger("nonconform").setLevel(logging.ERROR)

    # To enable debug messages:
    logging.getLogger("nonconform").setLevel(logging.DEBUG)

    # To turn off all logging:
    logging.getLogger("nonconform").setLevel(logging.CRITICAL)
"""

__version__ = "0.9.166"
__author__ = "Oliver Hennhoefer"
__email__ = "oliver.hennhoefer@mail.de"

from . import estimation, strategy, utils

__all__ = ["estimation", "strategy", "utils"]
