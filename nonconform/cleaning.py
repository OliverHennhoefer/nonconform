"""Calibration-cleaning helpers for contaminated reference data.

This module provides score-level utilities for the Label-Trim workflow from
Bashari, Sesia, and Romano (2025). The helpers are intentionally independent of
``ConformalDetector`` so users can opt into calibration cleaning without
changing existing detector, p-value, weighting, or FDR behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_1d_finite_scores(values: Any) -> np.ndarray:
    """Normalize calibration scores into a strict finite 1D float array."""
    try:
        scores = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration_scores must be a numeric 1D array.") from exc

    if scores.ndim != 1:
        raise ValueError(
            f"calibration_scores must be a 1D array, got shape {scores.shape!r}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration_scores must be finite.")
    return scores


def _validate_label_budget(label_budget: int, n_scores: int) -> int:
    """Validate and normalize a Label-Trim annotation budget."""
    if isinstance(label_budget, bool) or not isinstance(label_budget, int):
        raise TypeError("label_budget must be a non-negative integer.")
    if label_budget < 0:
        raise ValueError("label_budget must be non-negative.")
    return min(label_budget, n_scores)


def _as_1d_candidate_indices(values: Any, n_scores: int) -> np.ndarray:
    """Normalize and validate candidate indices."""
    indices = np.asarray(values)
    if indices.ndim != 1:
        raise ValueError(
            f"candidate_indices must be a 1D array, got shape {indices.shape!r}."
        )
    if indices.size == 0:
        return indices.astype(int, copy=True)
    if indices.dtype == bool or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("candidate_indices must contain integers.")

    indices = indices.astype(int, copy=True)
    if np.any(indices < 0) or np.any(indices >= n_scores):
        raise IndexError("candidate_indices must be within calibration_scores bounds.")
    if np.unique(indices).size != indices.size:
        raise ValueError("candidate_indices must not contain duplicates.")
    return indices


def _as_1d_candidate_labels(values: Any, n_candidates: int) -> np.ndarray:
    """Normalize and validate candidate labels."""
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(
            f"candidate_labels must be a 1D array, got shape {labels.shape!r}."
        )
    if labels.size != n_candidates:
        raise ValueError(
            "candidate_labels must have the same length as candidate_indices. "
            f"Got {labels.size} and {n_candidates}."
        )
    return labels.copy()


@dataclass(frozen=True, slots=True)
class LabelTrimResult:
    """Result of applying Label-Trim to calibration scores.

    Attributes:
        trimmed_scores: Calibration scores after removing annotated outliers.
        keep_mask: Boolean mask over the original calibration scores.
        removed_indices: Original calibration indices removed by Label-Trim.
        candidate_indices: Original calibration indices selected for annotation.
        n_original: Number of input calibration scores.
        n_candidates: Number of annotated candidate points.
        n_removed: Number of annotated outliers removed.
        n_kept: Number of calibration scores retained.
    """

    trimmed_scores: np.ndarray
    keep_mask: np.ndarray
    removed_indices: np.ndarray
    candidate_indices: np.ndarray
    n_original: int
    n_candidates: int
    n_removed: int
    n_kept: int


def select_label_trim_candidates(
    calibration_scores: np.ndarray,
    *,
    label_budget: int,
) -> np.ndarray:
    """Select high-score calibration points for expert annotation.

    Label-Trim inspects the most suspicious calibration points, where suspicion
    is measured by anomaly-oriented scores (higher means more anomalous). Ties
    are broken deterministically by original calibration order.

    Args:
        calibration_scores: 1D anomaly-oriented calibration scores.
        label_budget: Maximum number of calibration points to annotate. Values
            above the number of scores select all scores.

    Returns:
        Original calibration indices in descending anomaly-score priority.
    """
    scores = _as_1d_finite_scores(calibration_scores)
    budget = _validate_label_budget(label_budget, len(scores))
    if budget == 0:
        return np.array([], dtype=int)

    ranked_indices = np.argsort(-scores, kind="mergesort")
    return ranked_indices[:budget].astype(int, copy=True)


def apply_label_trim(
    calibration_scores: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    outlier_label: object = 1,
) -> LabelTrimResult:
    """Remove only annotated outliers from a calibration score set.

    This function implements the Label-Trim cleaning step after the selected
    candidate points have been annotated. It keeps unannotated points and
    annotated inliers, removing only annotated candidate points whose label
    equals ``outlier_label``.

    Args:
        calibration_scores: 1D anomaly-oriented calibration scores.
        candidate_indices: Original calibration indices selected for annotation.
        candidate_labels: Labels for ``candidate_indices`` in the same order.
        outlier_label: Label value indicating an annotated outlier. Defaults to
            ``1``.

    Returns:
        LabelTrimResult containing the trimmed scores, original-index mask, and
        summary counts.
    """
    scores = _as_1d_finite_scores(calibration_scores)
    candidates = _as_1d_candidate_indices(candidate_indices, len(scores))
    labels = _as_1d_candidate_labels(candidate_labels, len(candidates))

    annotated_outliers = labels == outlier_label
    removed_indices = candidates[annotated_outliers].astype(int, copy=True)

    keep_mask = np.ones(len(scores), dtype=bool)
    keep_mask[removed_indices] = False
    trimmed_scores = scores[keep_mask].copy()

    return LabelTrimResult(
        trimmed_scores=trimmed_scores,
        keep_mask=keep_mask,
        removed_indices=removed_indices,
        candidate_indices=candidates,
        n_original=len(scores),
        n_candidates=len(candidates),
        n_removed=len(removed_indices),
        n_kept=len(trimmed_scores),
    )


__all__ = [
    "LabelTrimResult",
    "apply_label_trim",
    "select_label_trim_candidates",
]
