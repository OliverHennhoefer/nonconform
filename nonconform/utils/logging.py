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
    By default, a NullHandler is added to prevent unwanted output unless
    explicitly configured by the user.

    Examples
    --------
    >>> logger = get_logger("estimation.extreme_conformal")
    >>> logger.warning("GPD fitting failed, using standard approach")
    """
    logger = logging.getLogger(f"nonconform.{name}")
    if not logger.handlers:
        # Add NullHandler by default (library best practice)
        # This prevents unwanted output unless user explicitly configures logging
        logger.addHandler(logging.NullHandler())
    return logger
