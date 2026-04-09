from __future__ import annotations

from copy import deepcopy
from typing import Any, Self

import numpy as np
import pytest


class MeanDistanceDetector:
    """Simple one-class detector with distance-based anomaly scores."""

    def __init__(self, invert: bool = False) -> None:
        self.invert = invert
        self._params = {"random_state": None, "invert": invert}
        self._center: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self._center = np.mean(X, axis=0)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._center is None:
            raise RuntimeError("Detector is not fitted.")
        dist = np.linalg.norm(X - self._center, axis=1)
        return -dist if self.invert else dist

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        self.invert = bool(self._params["invert"])
        return self

    def __copy__(self) -> MeanDistanceDetector:
        new = type(self)(invert=self.invert)
        new._params = self._params.copy()
        new._center = None if self._center is None else self._center.copy()
        return new

    def __deepcopy__(self, memo: dict) -> MeanDistanceDetector:
        new = type(self)(invert=self.invert)
        memo[id(self)] = new
        new._params = deepcopy(self._params, memo)
        new._center = None if self._center is None else deepcopy(self._center, memo)
        return new


@pytest.fixture
def labeled_ood_dataset():
    """Create labeled inlier/outlier/test sets for integrative workflows."""

    def _build(
        *,
        n_inliers: int = 80,
        n_outliers: int = 50,
        n_test: int = 24,
        n_features: int = 3,
        seed: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_inliers = rng.normal(loc=0.0, scale=0.7, size=(n_inliers, n_features))
        x_outliers = rng.normal(loc=3.0, scale=0.7, size=(n_outliers, n_features))

        n_test_in = n_test // 2
        n_test_out = n_test - n_test_in
        x_test_in = rng.normal(loc=0.1, scale=0.8, size=(n_test_in, n_features))
        x_test_out = rng.normal(loc=3.1, scale=0.8, size=(n_test_out, n_features))
        x_test = np.vstack([x_test_in, x_test_out])
        y_test = np.array([0] * n_test_in + [1] * n_test_out)

        shuffle_idx = rng.permutation(n_test)
        return x_inliers, x_outliers, x_test[shuffle_idx], y_test[shuffle_idx]

    return _build
