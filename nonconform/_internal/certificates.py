"""Shared private primitives for FDP and FPR certificates."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .provenance import (
    CalibrationMode,
    EstimationFamily,
    ResultProvenance,
    StrategyFamily,
)


def immutable_array(values: np.ndarray) -> np.ndarray:
    """Own an array in immutable bytes, preventing write-flag escalation."""
    arr = np.ascontiguousarray(values)
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def validate_scope(
    provenance: ResultProvenance | None,
    *,
    procedure: Literal["fdp_bounds", "fpr_bounds"],
) -> None:
    """Require unweighted Split provenance and procedure-specific evidence."""
    if procedure == "fdp_bounds":
        external_evidence = "p-values"
        factory = "FDPCertificate.from_p_values()"
        weighted_evidence = "conformal p-values"
    else:
        external_evidence = "scores"
        factory = "FPRCertificate.from_scores()"
        weighted_evidence = "calibration scores"

    if provenance is None:
        raise ValueError(
            f"{procedure}() requires native provenance. For external "
            f"{external_evidence}, use {factory} and verify its assumptions."
        )
    if provenance.weighted:
        raise ValueError(f"{procedure}() supports only unweighted {weighted_evidence}.")
    if (
        procedure == "fdp_bounds"
        and provenance.estimation_family is not EstimationFamily.EMPIRICAL
    ):
        raise ValueError(f"{procedure}() supports empirical conformal p-values only.")
    if provenance.strategy_family is not StrategyFamily.SPLIT:
        raise ValueError(f"{procedure}() supports split or detached calibration only.")
    if provenance.calibration_mode not in {
        CalibrationMode.INTEGRATED,
        CalibrationMode.DETACHED,
    }:
        raise ValueError(
            f"{procedure}() requires a fitted or calibrated native result."
        )
