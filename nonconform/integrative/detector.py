"""Public detector API for integrative conformal detection."""

from __future__ import annotations

from copy import deepcopy
from typing import Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from nonconform.integrative._core import (
    SplitState,
    TCVPlusState,
    _as_numpy_with_index,
    _ensure_2d,
    build_split_state,
    build_tcv_plus_state,
    compute_split_batch,
    compute_tcv_plus_batch,
)
from nonconform.integrative.models import IntegrativeModel
from nonconform.integrative.strategies import IntegrativeSplit, TransductiveCVPlus
from nonconform.structures import ConformalResult


class IntegrativeConformalDetector:
    """Integrative conformal detector with labeled inliers and outliers.

    This detector implements the split and TCV+ integrative conformal workflows
    described in the Liang-Sesia-Sun reference paper.

    Args:
        models: Collection of typed model specifications.
        strategy: Integrative strategy instance.
        seed: Optional random seed.
        verbose: Reserved for parity with the main detector API.
    """

    def __init__(
        self,
        *,
        models: IntegrativeModel | list[IntegrativeModel],
        strategy: IntegrativeSplit | TransductiveCVPlus,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if isinstance(models, IntegrativeModel):
            models = [models]
        if not models:
            raise ValueError("models must contain at least one IntegrativeModel.")
        self.models = [deepcopy(model) for model in models]
        self.strategy = deepcopy(strategy)
        self.seed = seed
        self.verbose = verbose
        self._state: SplitState | TCVPlusState | None = None
        self._last_result: ConformalResult | None = None

    def fit(
        self,
        x_inliers: pd.DataFrame | np.ndarray,
        x_outliers: pd.DataFrame | np.ndarray,
    ) -> Self:
        """Fit the detector on labeled inliers and labeled outliers."""
        x_in = _ensure_2d("x_inliers", np.asarray(x_inliers))
        x_out = _ensure_2d("x_outliers", np.asarray(x_outliers))
        if x_in.shape[1] != x_out.shape[1]:
            raise ValueError(
                "x_inliers and x_outliers must have the same feature count."
            )

        if isinstance(self.strategy, IntegrativeSplit):
            n_calib_in, n_calib_out = self.strategy.resolve_sizes(len(x_in), len(x_out))
            self._state = build_split_state(
                models=self.models,
                x_inliers=x_in,
                x_outliers=x_out,
                n_calib_in=n_calib_in,
                n_calib_out=n_calib_out,
                seed=self.seed,
            )
        else:
            self.strategy.validate_dataset_sizes(len(x_in), len(x_out))
            self._state = build_tcv_plus_state(
                models=self.models,
                x_inliers=x_in,
                x_outliers=x_out,
                k_in=self.strategy.k_in,
                k_out=self.strategy.k_out,
                seed=self.seed,
                shuffle=self.strategy.shuffle,
            )

        self._last_result = None
        return self

    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
    ) -> np.ndarray | pd.Series:
        """Return integrative conformal p-values for a test batch."""
        if self._state is None:
            raise NotFittedError(
                "This IntegrativeConformalDetector instance is not fitted yet."
            )
        x_array, index = _as_numpy_with_index(x)
        x_array = _ensure_2d("x", x_array)

        if isinstance(self._state, SplitState):
            (
                p_values,
                ratios,
                ratio_cal,
                metadata,
            ) = compute_split_batch(self._state, x_array)
        else:
            (
                p_values,
                ratios,
                ratio_cal,
                metadata,
            ) = compute_tcv_plus_batch(self._state, x_array)

        self._last_result = ConformalResult(
            p_values=p_values.copy(),
            test_scores=ratios.copy(),
            calib_scores=ratio_cal.copy(),
            metadata={"integrative": metadata},
        )
        if index is not None:
            return pd.Series(p_values, index=index, name="p_value")
        return p_values

    def score_samples(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
    ) -> np.ndarray | pd.Series:
        """Return the integrative ranking statistic ``r = u0 / u1``."""
        p_values = self.compute_p_values(x)
        result = self.last_result
        if result is None or result.test_scores is None:
            raise RuntimeError(
                "Internal error: expected test_scores after compute_p_values()."
            )
        if isinstance(p_values, pd.Series):
            return pd.Series(result.test_scores, index=p_values.index, name="score")
        return result.test_scores.copy()

    def select(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> np.ndarray | pd.Series:
        """Return the split conditional-FDR selection mask."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        x_array, index = _as_numpy_with_index(x)
        x_array = _ensure_2d("x", x_array)
        _ = self.compute_p_values(x_array)

        if isinstance(self.strategy, TransductiveCVPlus):
            raise NotImplementedError(
                "select() is currently implemented only for IntegrativeSplit. "
                "TCV+ conditional-FDR selection is intentionally deferred."
            )

        from nonconform.fdr import integrative_false_discovery_control

        selection_seed = self.seed if seed is None else seed
        selected = integrative_false_discovery_control(
            result=self._last_result,
            alpha=alpha,
            seed=selection_seed,
        )
        if index is not None:
            return pd.Series(selected, index=index, name="selected")
        return selected

    @property
    def last_result(self) -> ConformalResult | None:
        """Return the most recent result bundle."""
        return None if self._last_result is None else self._last_result.copy()

    @property
    def is_fitted(self) -> bool:
        """Return whether fit() has been called successfully."""
        return self._state is not None


__all__ = [
    "IntegrativeConformalDetector",
]
