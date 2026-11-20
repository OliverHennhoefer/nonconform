"""Owned, complete state for a fitted detector and one prepared weight batch."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

from .groups import CalibrationGroups
from .provenance import BatchSignature, CalibrationMode, restore_fields

if TYPE_CHECKING:
    from nonconform.structures import AnomalyDetector


FeatureBatch = np.ndarray | pd.DataFrame


def as_feature_batch(values: Any, *, n_features: int | None = None) -> FeatureBatch:
    """Require a two-dimensional batch with a consistent feature count.

    A pandas DataFrame is returned unchanged so the wrapped detector sees its
    column names and dtypes; a Series becomes a one-column DataFrame. Other
    inputs are converted to a NumPy array.
    """
    if isinstance(values, pd.Series):
        values = values.to_frame()
    batch = values if isinstance(values, pd.DataFrame) else as_feature_input(values)
    if batch.ndim != 2:
        raise ValueError(
            "x must be a two-dimensional batch with features; "
            f"got an array of shape {batch.shape}. Reshape your data with "
            "x.reshape(-1, 1) for a single feature or x.reshape(1, -1) for a "
            "single sample."
        )
    if batch.shape[1] == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape={batch.shape}) while a "
            "minimum of 1 is required."
        )
    if n_features is not None and batch.shape[1] != n_features:
        raise ValueError(
            f"X has {batch.shape[1]} features, but ConformalDetector is "
            f"expecting {n_features} features as input."
        )
    return batch


def take_rows(x: FeatureBatch, rows: Any) -> FeatureBatch:
    """Select rows by position from a NumPy array or a pandas DataFrame."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[rows]
    return x[rows]


def as_array(x: FeatureBatch) -> np.ndarray:
    """Return the NumPy values of a feature batch."""
    return x.to_numpy() if isinstance(x, pd.DataFrame) else np.asarray(x)


def as_feature_input(values: Any) -> np.ndarray:
    """Convert non-pandas input to NumPy, rejecting sparse matrices."""
    if sparse.issparse(values):
        raise TypeError(
            "Sparse input is not supported. Convert it with X.toarray() first."
        )
    return np.asarray(values)


def feature_names_of(x: Any) -> np.ndarray | None:
    """Return string column names of a DataFrame, following scikit-learn.

    Only DataFrames whose column names are all strings have feature names.
    """
    if not isinstance(x, pd.DataFrame):
        return None
    columns = np.asarray(x.columns, dtype=object)
    if len(columns) and all(isinstance(name, str) for name in columns):
        return columns
    return None


def _name_list(names: list[str], limit: int = 5) -> str:
    """Format names one per line, as scikit-learn does."""
    lines = [f"- {name}\n" for name in names[:limit]]
    if len(names) > limit:
        lines.append("- ...\n")
    return "".join(lines)


def check_feature_names(
    fitted: np.ndarray | None, x: Any, *, estimator_name: str
) -> None:
    """Reject renamed or reordered columns and warn about missing names.

    Mirrors the messages scikit-learn estimators use for the same checks.
    """
    names = feature_names_of(x)
    if fitted is None and names is None:
        return
    if fitted is None:
        warnings.warn(
            f"X has feature names, but {estimator_name} was fitted without "
            "feature names",
            UserWarning,
            stacklevel=4,
        )
        return
    if names is None:
        warnings.warn(
            f"X does not have valid feature names, but {estimator_name} was "
            "fitted with feature names",
            UserWarning,
            stacklevel=4,
        )
        return
    if len(fitted) == len(names) and np.array_equal(fitted, names):
        return
    unexpected = sorted(set(names) - set(fitted))
    missing = sorted(set(fitted) - set(names))
    message = "The feature names should match those that were passed during fit.\n"
    if unexpected:
        message += "Feature names unseen at fit time:\n" + _name_list(unexpected)
    if missing:
        message += "Feature names seen at fit time, yet now missing:\n" + _name_list(
            missing
        )
    if not unexpected and not missing:
        message += "Feature names must be in the same order as they were in fit.\n"
    raise ValueError(message)


@dataclass(frozen=True, slots=True)
class FittedCalibration:
    """Publish the fitted models, scores, and their provenance together."""

    models: tuple[AnomalyDetector, ...]
    scores: np.ndarray
    samples: np.ndarray
    n_features: int
    mode: CalibrationMode
    # Group of each calibration score; None for pooled calibration.
    groups: CalibrationGroups | None = None
    # String column names of the fitted DataFrame; None for other inputs.
    feature_names: np.ndarray | None = None

    def __setstate__(self, state: Sequence[Any]) -> None:
        """Restore pickles written before ``groups`` or ``feature_names``."""
        restore_fields(self, state, groups=None, feature_names=None)


@dataclass(frozen=True, slots=True)
class PreparedWeights:
    """Own weights bound to the batch that produced them."""

    calibration: np.ndarray
    test: np.ndarray
    batch_size: int
    signature: BatchSignature | None

    def copy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return weights without exposing the prepared batch's arrays."""
        return self.calibration.copy(), self.test.copy()
