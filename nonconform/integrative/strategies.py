"""Strategies for integrative conformal detection."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_split_size(n_calib: float | int, n_samples: int, *, name: str) -> int:
    """Validate and resolve a split calibration size."""
    if isinstance(n_calib, float):
        if not (0.0 < n_calib < 1.0):
            raise ValueError(f"{name} proportional n_calib must be in (0, 1).")
        n_calib_abs = math.ceil(n_samples * n_calib)
    elif isinstance(n_calib, int):
        if n_calib < 1:
            raise ValueError(f"{name} absolute n_calib must be at least 1.")
        n_calib_abs = n_calib
    else:
        raise TypeError(f"{name} n_calib must be int or float.")

    if n_calib_abs >= n_samples:
        raise ValueError(
            f"{name} calibration size ({n_calib_abs}) must leave at least one "
            "training sample."
        )
    return n_calib_abs


@dataclass(slots=True)
class IntegrativeSplit:
    """Split strategy for integrative conformal detection."""

    n_calib: float | int = 0.1

    def resolve_sizes(self, n_inliers: int, n_outliers: int) -> tuple[int, int]:
        """Resolve calibration sizes for inlier and outlier sets."""
        n_calib_in = _validate_split_size(self.n_calib, n_inliers, name="Inlier")
        n_calib_out = _validate_split_size(self.n_calib, n_outliers, name="Outlier")
        return n_calib_in, n_calib_out


@dataclass(slots=True)
class TransductiveCVPlus:
    """Transductive cross-validation+ strategy for integrative detection."""

    k_in: int = 5
    k_out: int = 5
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Validate fold configuration."""
        if not isinstance(self.k_in, int) or self.k_in < 2:
            raise ValueError("k_in must be an integer >= 2.")
        if not isinstance(self.k_out, int) or self.k_out < 2:
            raise ValueError("k_out must be an integer >= 2.")
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be a boolean value.")

    def validate_dataset_sizes(self, n_inliers: int, n_outliers: int) -> None:
        """Validate that the configured folds fit the dataset sizes."""
        if n_inliers < self.k_in:
            raise ValueError(
                f"Not enough inliers ({n_inliers}) for k_in={self.k_in} folds."
            )
        if n_outliers < self.k_out:
            raise ValueError(
                f"Not enough outliers ({n_outliers}) for k_out={self.k_out} folds."
            )


__all__ = [
    "IntegrativeSplit",
    "TransductiveCVPlus",
]
