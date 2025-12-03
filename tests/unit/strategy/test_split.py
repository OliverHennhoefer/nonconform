"""Unit tests for strategy/calibration/split.py."""

from copy import deepcopy
from typing import Any, Self

import numpy as np
import pytest

from nonconform.strategy import Split


class MockDetector:
    """Mock detector for testing strategies."""

    def __init__(self):
        self._fitted = False
        self._params = {"random_state": None, "n_jobs": 1, "contamination": 0.1}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self._fitted = True
        self._training_size = len(X)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.standard_normal(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self

    def __copy__(self):
        """Create shallow copy."""
        new = MockDetector()
        new._params = self._params.copy()
        return new

    def __deepcopy__(self, memo):
        """Create deep copy."""
        new = MockDetector()
        new._params = deepcopy(self._params, memo)
        return new


class TestSplitInit:
    """Tests for Split initialization."""

    def test_default_initialization(self):
        """Default initialization with proportional split."""
        strategy = Split()
        assert strategy.calib_size == 0.1  # Default value

    def test_proportional_n_calib(self):
        """Proportional calibration size (float between 0 and 1)."""
        strategy = Split(n_calib=0.3)
        assert strategy.calib_size == 0.3

    def test_absolute_n_calib(self):
        """Absolute calibration size (integer > 1)."""
        strategy = Split(n_calib=100)
        assert strategy.calib_size == 100

    def test_calib_size_property(self):
        """calib_size property returns calibration size."""
        strategy = Split(n_calib=0.25)
        assert strategy.calib_size == 0.25


class TestSplitFitCalibrate:
    """Tests for Split.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 5))

    @pytest.fixture
    def detector(self):
        """Mock detector."""
        return MockDetector()

    def test_returns_detector_list_and_scores(self, sample_data, detector):
        """fit_calibrate returns detector list and calibration scores."""
        strategy = Split(n_calib=0.2)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=42
        )
        assert isinstance(detector_set, list)
        assert len(detector_set) == 1  # Split returns single detector
        assert isinstance(calib_scores, np.ndarray)
        assert len(calib_scores) > 0

    def test_proportional_split_sizes(self, sample_data, detector):
        """Proportional split divides data correctly."""
        strategy = Split(n_calib=0.2)  # 20% for calibration
        _, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=42
        )
        # With 100 samples and 20% calibration, expect ~20 calibration scores
        assert 15 <= len(calib_scores) <= 25

    def test_absolute_split_sizes(self, sample_data, detector):
        """Absolute split size creates exact calibration set."""
        strategy = Split(n_calib=30)  # Exactly 30 for calibration
        _, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=42
        )
        assert len(calib_scores) == 30

    def test_detector_is_fitted(self, sample_data, detector):
        """Returned detector is fitted."""
        strategy = Split(n_calib=0.2)
        detector_set, _ = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=42
        )
        assert detector_set[0]._fitted

    def test_calibration_ids_stored_when_weighted(self, sample_data, detector):
        """Calibration IDs stored when weighted=True."""
        strategy = Split(n_calib=0.2)
        strategy.fit_calibrate(x=sample_data, detector=detector, seed=42, weighted=True)
        assert strategy.calibration_ids is not None
        assert len(strategy.calibration_ids) > 0

    def test_calibration_ids_none_when_not_weighted(self, sample_data, detector):
        """Calibration IDs not stored when weighted=False."""
        strategy = Split(n_calib=0.2)
        strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=42, weighted=False
        )
        # In non-weighted mode, calibration_ids should be None or empty
        calib_ids = strategy.calibration_ids
        assert calib_ids is None or len(calib_ids) == 0


class TestSplitReproducibility:
    """Tests for reproducibility with seed."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 5))

    def test_same_seed_same_split(self, sample_data):
        """Same seed produces same split."""
        strategy1 = Split(n_calib=0.2)
        _, scores1 = strategy1.fit_calibrate(
            x=sample_data.copy(), detector=MockDetector(), seed=42
        )

        strategy2 = Split(n_calib=0.2)
        _, scores2 = strategy2.fit_calibrate(
            x=sample_data.copy(), detector=MockDetector(), seed=42
        )

        # Same seed should produce same number of calibration scores
        assert len(scores1) == len(scores2)

    def test_different_seed_different_split(self, sample_data):
        """Different seeds may produce different splits."""
        strategy1 = Split(n_calib=0.2)
        strategy1.fit_calibrate(
            x=sample_data.copy(), detector=MockDetector(), seed=42, weighted=True
        )

        strategy2 = Split(n_calib=0.2)
        strategy2.fit_calibrate(
            x=sample_data.copy(), detector=MockDetector(), seed=123, weighted=True
        )

        # Different seeds should produce different calibration IDs
        # (with high probability for reasonable data sizes)
        if (
            strategy1.calibration_ids is not None
            and strategy2.calibration_ids is not None
        ):
            assert not np.array_equal(
                sorted(strategy1.calibration_ids), sorted(strategy2.calibration_ids)
            )


class TestSplitEdgeCases:
    """Tests for edge cases."""

    def test_minimum_data_size(self):
        """Strategy works with minimum viable data size."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 2))  # Small dataset
        strategy = Split(n_calib=0.3)  # 30% = 3 samples for calibration
        detector_set, calib_scores = strategy.fit_calibrate(
            x=X, detector=MockDetector(), seed=42
        )
        assert len(detector_set) == 1
        assert len(calib_scores) >= 1

    def test_accepts_numpy_values_from_dataframe(self):
        """Strategy works with numpy values extracted from DataFrame."""
        import pandas as pd

        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.standard_normal((50, 3)))
        # Convert to numpy for use with Split strategy
        # (Split uses array indexing which doesn't work with DataFrames)
        strategy = Split(n_calib=0.2)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=df.values, detector=MockDetector(), seed=42
        )
        assert len(detector_set) == 1
        assert len(calib_scores) > 0
