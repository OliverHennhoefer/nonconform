"""Shared private validation helpers."""

from __future__ import annotations

import numpy as np


def as_1d_numeric(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize array-like input into a strict one-dimensional float array."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape!r}.")
    return arr


def validate_finite(name: str, values: np.ndarray) -> None:
    """Validate that an array has only finite entries."""
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")


def validate_non_negative_finite(name: str, values: np.ndarray) -> None:
    """Validate that an array is finite and non-negative."""
    validate_finite(name, values)
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative.")


def validate_p_values(p_values: np.ndarray) -> None:
    """Validate that p-values are finite and within the closed unit interval."""
    if p_values.size == 0:
        return
    if not np.all(np.isfinite(p_values)):
        raise ValueError("p_values must be finite.")
    eps = 1e-10
    if np.any((p_values < -eps) | (p_values > 1 + eps)):
        raise ValueError("p_values must be within [0, 1].")


def validate_probability(name: str, value: float) -> float:
    """Validate a scalar probability in the open unit interval."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value in (0, 1).") from exc
    if not np.isfinite(scalar) or not (0.0 < scalar < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")
    return scalar


def validate_positive_integer(name: str, value: int) -> int:
    """Validate a positive built-in integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def validate_optional_seed(name: str, value: int | None) -> int | None:
    """Validate an optional non-negative built-in integer seed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer or None.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None.")
    return value


def validate_positive_finite(name: str, value: float) -> float:
    """Validate a positive finite scalar."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite value.") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return scalar
