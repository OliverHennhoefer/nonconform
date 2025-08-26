"""Logging utilities for the nonconform package."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the nonconform package.

    Parameters
    ----------
    name : str
        The name of the logger, typically the module name.

    Returns
    -------
    logging.Logger
        A logger instance for the nonconform package.

    Notes
    -----
    This function creates loggers with the naming convention "nonconform.{name}".
    By default, shows INFO level and above (INFO, WARNING, ERROR, CRITICAL).
    Users can control verbosity with standard logging:
    logging.getLogger("nonconform").setLevel(level).

    Examples
    --------
    >>> logger = get_logger("estimation.extreme_conformal")
    >>> logger.warning("GPD fitting failed, using standard approach")

    >>> # To silence warnings:
    >>> logging.getLogger("nonconform").setLevel(logging.ERROR)

    >>> # To enable debug:
    >>> logging.getLogger("nonconform").setLevel(logging.DEBUG)
    """
    logger = logging.getLogger(f"nonconform.{name}")

    # Configure root nonconform logger if not already done
    root_logger = logging.getLogger("nonconform")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)  # Show INFO and above by default
        root_logger.propagate = False

    return logger
