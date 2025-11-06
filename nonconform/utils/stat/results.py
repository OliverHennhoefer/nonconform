from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ConformalResult:
    """Snapshot of detector outputs required for downstream procedures.

    Attributes:
        p_values: Conformal p-values for test instances (None when unavailable).
        test_scores: Non-conformity scores for the test instances (raw predictions).
        calib_scores: Non-conformity scores for the calibration set.
        test_weights: Importance weights for test instances (weighted mode only).
        calib_weights: Importance weights for calibration instances.
        metadata: Optional dictionary with extra data (debug info, timings, etc.).
    """

    p_values: np.ndarray | None = None
    test_scores: np.ndarray | None = None
    calib_scores: np.ndarray | None = None
    test_weights: np.ndarray | None = None
    calib_weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> ConformalResult:
        """Return a shallow copy with numpy arrays duplicated."""
        return ConformalResult(
            p_values=None if self.p_values is None else self.p_values.copy(),
            test_scores=None if self.test_scores is None else self.test_scores.copy(),
            calib_scores=None
            if self.calib_scores is None
            else self.calib_scores.copy(),
            test_weights=None
            if self.test_weights is None
            else self.test_weights.copy(),
            calib_weights=None
            if self.calib_weights is None
            else self.calib_weights.copy(),
            metadata=self.metadata.copy(),
        )
