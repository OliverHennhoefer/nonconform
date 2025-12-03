"""Unit tests for detection/conformal.py."""

from copy import deepcopy
from typing import Any, Self

import numpy as np
import pytest

from nonconform.detection import ConformalDetector
from nonconform.detection.protocol import AnomalyDetector
from nonconform.strategy import Split
from nonconform.utils.func.enums import Aggregation


class MockDetector:
    """Mock anomaly detector for testing."""

    def __init__(self, scores=None):
        self._fitted = False
        self._scores = scores if scores is not None else np.array([0.5])
        self._params = {"random_state": None, "n_jobs": 1, "contamination": 0.1}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self._scores is not None:
            return (
                self._scores[: len(X)]
                if len(self._scores) >= len(X)
                else np.tile(self._scores, len(X))[: len(X)]
            )
        rng = np.random.default_rng()
        return rng.standard_normal(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self

    def __copy__(self):
        """Create shallow copy."""
        new = MockDetector(self._scores.copy() if self._scores is not None else None)
        new._params = self._params.copy()
        return new

    def __deepcopy__(self, memo):
        """Create deep copy."""
        new = MockDetector(
            deepcopy(self._scores, memo) if self._scores is not None else None
        )
        new._params = deepcopy(self._params, memo)
        return new


class TestConformalDetectorInit:
    """Tests for ConformalDetector initialization."""

    def test_basic_init(self):
        """Basic initialization works."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )
        assert detector is not None
        assert not detector.is_fitted

    def test_init_with_seed(self):
        """Initialization with seed sets random state."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2), seed=42
        )
        assert detector.seed == 42

    def test_init_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ConformalDetector(
                detector=MockDetector(), strategy=Split(n_calib=0.2), seed=-1
            )

    def test_init_invalid_aggregation_raises(self):
        """Invalid aggregation type raises TypeError."""
        with pytest.raises(TypeError, match="Aggregation enum"):
            ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                aggregation="median",  # type: ignore  # Should be enum
            )

    def test_init_with_aggregation(self):
        """Initialization with aggregation parameter."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            aggregation=Aggregation.MEAN,
        )
        assert detector.aggregation == Aggregation.MEAN

    def test_init_adapts_detector(self):
        """Detector is adapted to AnomalyDetector protocol."""
        mock = MockDetector()
        detector = ConformalDetector(detector=mock, strategy=Split(n_calib=0.2))
        assert isinstance(detector.detector, AnomalyDetector)


class TestConformalDetectorFit:
    """Tests for ConformalDetector.fit()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 5))

    def test_fit_sets_is_fitted(self, sample_data):
        """fit() sets is_fitted property."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        assert not detector.is_fitted
        detector.fit(sample_data)
        assert detector.is_fitted

    def test_fit_populates_detector_set(self, sample_data):
        """fit() populates detector_set."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(sample_data)
        assert len(detector.detector_set) > 0

    def test_fit_populates_calibration_set(self, sample_data):
        """fit() populates calibration_set."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(sample_data)
        assert len(detector.calibration_set) > 0

    def test_fit_accepts_dataframe(self, sample_data):
        """fit() accepts pandas DataFrame."""
        import pandas as pd

        rng = np.random.default_rng(42)
        df = pd.DataFrame(sample_data)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(df)
        assert detector.is_fitted


class TestConformalDetectorPredict:
    """Tests for ConformalDetector.predict()."""

    @pytest.fixture
    def fitted_detector(self):
        """Pre-fitted conformal detector."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 5))
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(X_train)
        return detector

    def test_predict_before_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )
        with pytest.raises(RuntimeError, match="fit"):
            detector.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_predict_returns_p_values(self, fitted_detector):
        """predict() returns p-values by default."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        p_values = fitted_detector.predict(X_test)
        assert len(p_values) == 10
        assert all(0 <= p <= 1 for p in p_values)

    def test_predict_raw_returns_scores(self, fitted_detector):
        """predict(raw=True) returns raw scores."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        scores = fitted_detector.predict(X_test, raw=True)
        assert len(scores) == 10
        # Raw scores can be any value, not necessarily in [0, 1]

    def test_predict_accepts_dataframe(self, fitted_detector):
        """predict() accepts pandas DataFrame."""
        import pandas as pd

        rng = np.random.default_rng(42)
        X_test = pd.DataFrame(rng.standard_normal((10, 5)))
        p_values = fitted_detector.predict(X_test)
        assert len(p_values) == 10


class TestConformalDetectorProperties:
    """Tests for ConformalDetector properties."""

    @pytest.fixture
    def fitted_detector(self):
        """Pre-fitted conformal detector."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 5))
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(X_train)
        return detector

    def test_detector_set_returns_copy(self, fitted_detector):
        """detector_set returns defensive copy."""
        set1 = fitted_detector.detector_set
        set2 = fitted_detector.detector_set
        assert set1 is not set2  # Different list objects

    def test_calibration_set_returns_copy(self, fitted_detector):
        """calibration_set returns defensive copy."""
        set1 = fitted_detector.calibration_set
        set2 = fitted_detector.calibration_set
        assert set1 is not set2  # Different array objects

    def test_calibration_samples_empty_without_weights(self, fitted_detector):
        """calibration_samples is empty in standard mode."""
        samples = fitted_detector.calibration_samples
        assert len(samples) == 0

    def test_is_fitted_before_fit(self):
        """is_fitted is False before fit()."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )
        assert not detector.is_fitted

    def test_last_result_none_before_predict(self, fitted_detector):
        """last_result is None before predict()."""
        # Create new fitted detector without predict
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 5))
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(X_train)
        assert detector.last_result is None

    def test_last_result_populated_after_predict(self, fitted_detector):
        """last_result is populated after predict()."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        fitted_detector.predict(X_test)
        result = fitted_detector.last_result
        assert result is not None
        assert result.p_values is not None
        assert len(result.p_values) == 10


class TestConformalDetectorReproducibility:
    """Tests for reproducibility with seed."""

    def test_same_seed_same_results(self):
        """Same seed produces same results."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        X_test = rng.standard_normal((20, 5))

        # Fixed scores for consistent testing
        fixed_scores = rng.standard_normal(100)

        detector1 = ConformalDetector(
            detector=MockDetector(fixed_scores.copy()),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector1.fit(X_train.copy())
        p1 = detector1.predict(X_test.copy())

        detector2 = ConformalDetector(
            detector=MockDetector(fixed_scores.copy()),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector2.fit(X_train.copy())
        p2 = detector2.predict(X_test.copy())

        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_seed_different_results(self):
        """Different seeds produce different results."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        X_test = rng.standard_normal((20, 5))

        detector1 = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector1.fit(X_train.copy())
        p1 = detector1.predict(X_test.copy())

        detector2 = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=123,
        )
        detector2.fit(X_train.copy())
        p2 = detector2.predict(X_test.copy())

        # Results should differ (with high probability)
        assert not np.allclose(p1, p2)
