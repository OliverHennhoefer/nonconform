"""Unit tests for detector.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM

from nonconform import ConformalDetector as _ConformalDetector
from nonconform import JackknifeBootstrap, Split
from nonconform.enums import ScorePolarity
from nonconform.structures import AnomalyDetector
from nonconform.weighting import BaseWeightEstimator

# MockDetector is imported from tests/conftest.py via pytest fixture discovery
from tests.conftest import MockDetector


class ConformalDetector(_ConformalDetector):
    """Test helper that defaults custom detectors to anomalous-higher scores."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("score_polarity", "higher_is_anomalous")
        super().__init__(*args, **kwargs)


class CountingWeightEstimator(BaseWeightEstimator):
    """Minimal estimator that records fit calls for detector API testing."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self._is_fitted = False
        self._w_calib = np.array([])
        self._w_test = np.array([])

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        self.fit_calls += 1
        self._w_calib = np.ones(len(calibration_samples), dtype=float)
        self._w_test = np.ones(len(test_samples), dtype=float)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.ones(len(calibration_samples), dtype=float),
            np.ones(len(test_samples), dtype=float),
        )


class SeedAwareDetector:
    """Deterministic detector whose scores depend on random_state."""

    def __init__(self) -> None:
        self._params = {"random_state": None}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SeedAwareDetector":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self._params["random_state"])
        return rng.standard_normal(len(X))

    def get_params(self, deep: bool = True) -> dict[str, int | None]:
        return self._params.copy()

    def set_params(self, **params: int | None) -> "SeedAwareDetector":
        self._params.update(params)
        return self

    def __copy__(self) -> "SeedAwareDetector":
        new = type(self)()
        new._params = self._params.copy()
        return new

    def __deepcopy__(self, memo: dict) -> "SeedAwareDetector":
        new = type(self)()
        memo[id(self)] = new
        new._params = self._params.copy()
        return new


@pytest.fixture
def fitted_detector():
    """Pre-fitted conformal detector."""
    rng = np.random.default_rng(42)
    x_train = rng.standard_normal((100, 5))
    detector = ConformalDetector(
        detector=MockDetector(rng.standard_normal(100)),
        strategy=Split(n_calib=0.2),
        seed=42,
    )
    detector.fit(x_train)
    return detector


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

    def test_init_copies_strategy_to_isolate_external_mutation(self):
        """External strategy mutations after init do not affect detector."""
        strategy = Split(n_calib=0.2)
        detector = ConformalDetector(
            detector=MockDetector(), strategy=strategy, seed=42
        )
        strategy._calib_size = 0.5
        assert detector.strategy.calib_size == 0.2

    def test_init_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ConformalDetector(
                detector=MockDetector(), strategy=Split(n_calib=0.2), seed=-1
            )

    def test_init_invalid_aggregation_raises(self):
        """Invalid aggregation type raises TypeError."""
        with pytest.raises(TypeError, match="aggregation method must be a string"):
            ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                aggregation=1,  # type: ignore[arg-type]
            )

    def test_init_with_aggregation(self):
        """Initialization with aggregation parameter."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            aggregation="mean",
        )
        assert detector.aggregation == "mean"

    def test_init_normalizes_aggregation(self):
        """Aggregation strings are normalized."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            aggregation="  MEDIAN ",
        )
        assert detector.aggregation == "median"

    def test_init_invalid_aggregation_string_raises(self):
        """Unsupported aggregation strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                aggregation="avg",
            )

    def test_init_adapts_detector(self):
        """Detector is adapted to AnomalyDetector protocol."""
        mock = MockDetector()
        detector = ConformalDetector(detector=mock, strategy=Split(n_calib=0.2))
        assert isinstance(detector.detector, AnomalyDetector)

    def test_init_with_score_polarity_literal(self):
        """Initialization accepts score_polarity literal values."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            score_polarity="higher_is_anomalous",
        )
        assert detector.score_polarity is ScorePolarity.HIGHER_IS_ANOMALOUS

    def test_init_with_score_polarity_enum(self):
        """Initialization accepts score_polarity enum values."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            score_polarity=ScorePolarity.HIGHER_IS_ANOMALOUS,
        )
        assert detector.score_polarity is ScorePolarity.HIGHER_IS_ANOMALOUS

    def test_init_invalid_score_polarity_raises(self):
        """Invalid score_polarity string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid score_polarity value"):
            ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                score_polarity="invalid",  # type: ignore[arg-type]
            )

    def test_init_auto_unknown_detector_raises(self):
        """AUTO mode fails fast for unknown detector classes."""
        with pytest.raises(ValueError, match="Unable to infer score polarity"):
            _ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                score_polarity="auto",
            )


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
        rng = np.random.default_rng(42)
        df = pd.DataFrame(sample_data)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(df)
        assert detector.is_fitted

    def test_fit_returns_self(self, sample_data):
        """fit() returns self to support method chaining."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        result = detector.fit(sample_data)
        assert result is detector

    def test_fit_accepts_unused_y_for_sklearn_compatibility(self, sample_data):
        """fit() accepts y argument and ignores it."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        y = np.zeros(len(sample_data))
        detector.fit(sample_data, y=y)
        assert detector.is_fitted

    def test_fit_passes_n_jobs_to_supported_strategy(self, sample_data):
        """fit(n_jobs=...) works when strategy exposes n_jobs."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=JackknifeBootstrap(n_bootstraps=3),
            seed=42,
        )
        detector.fit(sample_data, n_jobs=1)
        assert detector.is_fitted

    def test_fit_n_jobs_with_unsupported_strategy_raises(self, sample_data):
        """fit(n_jobs=...) raises for strategies without n_jobs support."""
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        with pytest.raises(ValueError, match="does not support n_jobs"):
            detector.fit(sample_data, n_jobs=2)


class TestConformalDetectorPredict:
    """Tests for ConformalDetector prediction interfaces."""

    def test_predict_before_fit_raises(self):
        """compute_p_values() before fit() raises NotFittedError."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )
        with pytest.raises(NotFittedError, match="not fitted"):
            detector.compute_p_values(np.array([[1, 2, 3, 4, 5]]))

    def test_predict_returns_p_values(self, fitted_detector):
        """compute_p_values() returns p-values."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        p_values = fitted_detector.compute_p_values(X_test)
        assert len(p_values) == 10
        assert all(0 <= p <= 1 for p in p_values)

    def test_compute_p_values_repeated_calls_consistent(self, fitted_detector):
        """Repeated compute_p_values() calls produce consistent results."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        p_values_1 = fitted_detector.compute_p_values(X_test)
        p_values_2 = fitted_detector.compute_p_values(X_test)
        assert len(p_values_2) == 10
        np.testing.assert_array_equal(p_values_1, p_values_2)

    def test_score_samples_returns_scores(self, fitted_detector):
        """score_samples() returns raw scores."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        scores = fitted_detector.score_samples(X_test)
        assert len(scores) == 10

    def test_predict_accepts_dataframe(self, fitted_detector):
        """compute_p_values() with DataFrame returns indexed Series."""
        rng = np.random.default_rng(42)
        index = pd.RangeIndex(start=100, stop=110, step=1)
        X_test = pd.DataFrame(rng.standard_normal((10, 5)), index=index)
        p_values = fitted_detector.compute_p_values(X_test)
        assert isinstance(p_values, pd.Series)
        assert p_values.index.equals(index)
        assert len(p_values) == 10

    def test_score_samples_dataframe_returns_series(self, fitted_detector):
        """score_samples() with DataFrame returns indexed Series."""
        rng = np.random.default_rng(42)
        index = pd.RangeIndex(start=0, stop=10, step=1)
        X_test = pd.DataFrame(rng.standard_normal((10, 5)), index=index)
        scores = fitted_detector.score_samples(X_test)
        assert isinstance(scores, pd.Series)
        assert scores.index.equals(index)
        assert len(scores) == 10


class TestConformalDetectorProperties:
    """Tests for ConformalDetector properties."""

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
        fitted_detector.compute_p_values(X_test)
        result = fitted_detector.last_result
        assert result is not None
        assert result.p_values is not None
        assert len(result.p_values) == 10

    def test_repr_summarizes_state(self, fitted_detector):
        """repr() exposes concise high-level detector state."""
        repr_before = repr(
            ConformalDetector(detector=MockDetector(), strategy=Split(n_calib=0.2))
        )
        assert "fitted=False" in repr_before
        assert "aggregation='median'" in repr_before

        repr_after = repr(fitted_detector)
        assert "fitted=True" in repr_after
        assert "n_models=" in repr_after
        assert "n_calibration=" in repr_after


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
        p1 = detector1.compute_p_values(X_test.copy())

        detector2 = ConformalDetector(
            detector=MockDetector(fixed_scores.copy()),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector2.fit(X_train.copy())
        p2 = detector2.compute_p_values(X_test.copy())

        np.testing.assert_array_almost_equal(p1, p2)

    def test_different_seed_different_results(self):
        """Different seeds produce different results."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        X_test = rng.standard_normal((20, 5))

        detector1 = ConformalDetector(
            detector=SeedAwareDetector(),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector1.fit(X_train.copy())
        p1 = detector1.compute_p_values(X_test.copy())

        detector2 = ConformalDetector(
            detector=SeedAwareDetector(),
            strategy=Split(n_calib=0.2),
            seed=123,
        )
        detector2.fit(X_train.copy())
        p2 = detector2.compute_p_values(X_test.copy())

        # Results should differ (with high probability)
        assert not np.allclose(p1, p2)


class TestConformalDetectorWeightedPreparation:
    """Tests for explicit weighted preparation APIs."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(123)
        return rng.standard_normal((120, 4)), rng.standard_normal((30, 4))

    def test_prepare_weights_requires_weighted_mode(self, sample_data):
        """prepare_weights_for() raises when weighted mode is disabled."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            seed=42,
        )
        detector.fit(x_train)
        with pytest.raises(RuntimeError, match="requires weighted mode"):
            detector.prepare_weights_for(x_test)

    def test_prepare_weights_before_fit_raises(self, sample_data):
        """prepare_weights_for() before fit() raises NotFittedError."""
        _, x_test = sample_data
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2), seed=42
        )
        with pytest.raises(NotFittedError, match="not fitted"):
            detector.prepare_weights_for(x_test)

    def test_predict_without_refit_requires_prepared_weights(self, sample_data):
        """refit_weights=False requires prepared weights in weighted mode."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
            seed=42,
        )
        detector.fit(x_train)

        with pytest.raises(RuntimeError, match="Weights are not prepared"):
            detector.compute_p_values(x_test, refit_weights=False)

    def test_prepare_then_predict_without_refit(self, sample_data):
        """Prepared weights are reused when refit_weights=False."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        weight_estimator = CountingWeightEstimator()
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=weight_estimator,
            seed=42,
        )
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        assert weight_estimator.fit_calls == 1
        _ = detector.compute_p_values(x_test, refit_weights=False)
        assert weight_estimator.fit_calls == 1

    def test_refit_true_then_reuse_without_refit(self, sample_data):
        """A refit prediction prepares weights for later reuse."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        weight_estimator = CountingWeightEstimator()
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=weight_estimator,
            seed=42,
        )
        detector.fit(x_train)

        _ = detector.compute_p_values(x_test, refit_weights=True)
        assert weight_estimator.fit_calls == 1
        _ = detector.compute_p_values(x_test, refit_weights=False)
        assert weight_estimator.fit_calls == 1

    def test_reuse_with_different_batch_size_raises(self, sample_data):
        """Prepared weights cannot be reused for a different batch size."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
            seed=42,
        )
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        with pytest.raises(ValueError, match="batch size"):
            detector.compute_p_values(x_test[:10], refit_weights=False)

    def test_reuse_with_different_batch_content_raises(self, sample_data):
        """Prepared weights cannot be reused for same-size different batches."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
            seed=42,
        )
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        with pytest.raises(ValueError, match="batch content"):
            detector.compute_p_values(x_test + 1.0, refit_weights=False)

    def test_reuse_with_different_content_allowed_when_verification_disabled(
        self, sample_data
    ):
        """Same-size reuse can skip content check when explicitly disabled."""
        x_train, x_test = sample_data
        rng = np.random.default_rng(42)
        detector = ConformalDetector(
            detector=MockDetector(rng.standard_normal(100)),
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
            seed=42,
            verify_prepared_batch_content=False,
        )
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        p_values = detector.compute_p_values(x_test + 1.0, refit_weights=False)
        assert len(p_values) == len(x_test)


class TestConformalDetectorSklearnParams:
    """Tests for sklearn-style parameter protocol integration."""

    def test_get_params_exposes_nested_detector_params(self):
        """get_params(deep=True) includes detector__* nested keys."""
        detector = _ConformalDetector(
            detector=OneClassSVM(gamma=0.1),
            strategy=Split(n_calib=0.2),
            score_polarity="auto",
        )
        params = detector.get_params(deep=True)
        assert "detector__gamma" in params
        assert "strategy" in params

    def test_set_params_updates_nested_detector_params(self):
        """set_params(detector__...) delegates to wrapped detector."""
        detector = _ConformalDetector(
            detector=OneClassSVM(gamma=0.1),
            strategy=Split(n_calib=0.2),
            score_polarity="auto",
        )
        detector.set_params(detector__gamma=0.5)
        assert detector.get_params(deep=True)["detector__gamma"] == 0.5

    def test_clone_compatibility(self):
        """Estimator can be cloned via sklearn.base.clone."""
        detector = _ConformalDetector(
            detector=OneClassSVM(gamma=0.1),
            strategy=Split(n_calib=0.2),
            score_polarity="auto",
        )
        cloned = clone(detector)
        assert isinstance(cloned, _ConformalDetector)
