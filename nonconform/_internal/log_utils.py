"""Logging utilities for nonconform.

This module provides logging and decorator utilities used throughout the package.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the nonconform package.

    Args:
        name: The name of the logger, typically the module name.

    Returns:
        A logger instance for the nonconform package.

    Notes:
        Creates loggers with the naming convention "nonconform.{name}".
        By default, shows INFO level and above.
        Control verbosity with: logging.getLogger("nonconform").setLevel(level).

    Examples:
        >>> logger = get_logger("detector")
        >>> logger.info("Calibration completed")

        >>> # To silence warnings:
        >>> import logging
        >>> logging.getLogger("nonconform").setLevel(logging.ERROR)
    """
    logger = logging.getLogger(f"nonconform.{name}")

    # Configure root nonconform logger if not already done
    root_logger = logging.getLogger("nonconform")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    return logger


def ensure_numpy_array(func: Callable) -> Callable:
    """Ensure a specific input argument is a numpy array.

    **Internal use only.** This decorator is designed for methods where the first
    argument after `self` (conventionally named `x`) is expected to be a numpy array.
    Automatically converts pandas DataFrame to numpy array.

    Args:
        func: The method to be decorated. Must have `self` as first parameter,
            followed by the data argument `x`.

    Returns:
        The wrapped method that will receive `x` as a numpy array.

    Note:
        This is an internal utility decorator used throughout the package.
    """

    @wraps(func)
    def wrapper(
        self, x: pd.DataFrame | pd.Series | np.ndarray, *args: Any, **kwargs: Any
    ) -> Any:
        # Convert pandas objects without forcing a copy
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x_converted = x.to_numpy(copy=False)
        else:
            x_converted = x
        return func(self, x_converted, *args, **kwargs)

    return wrapper


__all__ = [
    "ensure_numpy_array",
    "get_logger",
]
