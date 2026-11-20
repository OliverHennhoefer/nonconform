"""Unit tests for detector.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM

from nonconform import ConformalDetector as _ConformalDetector
from nonconform import JackknifeBootstrap, Split
from nonconform.detector import BaseConformalDetector, _derive_wcs_pruning_seed
from nonconform.enums import Pruning, ScorePolarity
from nonconform.scoring import Empirical, calculate_weighted_p_val
from nonconform.structures import AnomalyDetector, ConformalResult
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


class StrictFittableDetector:
    """Detector that enforces fit-before-decision_function behavior."""

    def __init__(self) -> None:
        self._fitted = False
        self._params = {"random_state": None}

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "StrictFittableDetector":
        self._fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("StrictFittableDetector is not fitted.")
        return np.linalg.norm(X, axis=1)

    def get_params(self, deep: bool = True) -> dict[str, int | None]:
        return self._params.copy()

    def set_params(self, **params: int | None) -> "StrictFittableDetector":
        self._params.update(params)
        return self

    def __copy__(self) -> "StrictFittableDetector":
        new = type(self)()
        new._fitted = self._fitted
        new._params = self._params.copy()
        return new

    def __deepcopy__(self, memo: dict) -> "StrictFittableDetector":
        new = type(self)()
        memo[id(self)] = new
        new._fitted = self._fitted
        new._params = self._params.copy()
        return new


class NoDeepParamsDetector:
    """Detector whose get_params() does not accept the ``deep`` argument."""

    def __init__(self) -> None:
        self.threshold = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "NoDeepParamsDetector":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def get_params(self) -> dict[str, float]:
        return {"threshold": self.threshold}

    def set_params(self, **params: float) -> "NoDeepParamsDetector":
        for key, value in params.items():
            setattr(self, key, value)
        return self


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


def _split_detector(**kwargs) -> ConformalDetector:
    """Seeded split detector over fixed mock scores."""
    kwargs.setdefault("strategy", Split(n_calib=0.2))
    kwargs.setdefault("seed", 42)
    scores = np.random.default_rng(42).standard_normal(100)
    return ConformalDetector(detector=MockDetector(scores), **kwargs)


class TestConformalDetectorInit:
    """Tests for ConformalDetector initialization."""

    def test_init_defaults(self):
        """A new detector adapts its model, keeps the seed, and is unfitted."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2), seed=42
        )
        assert not detector.is_fitted
        assert detector.seed == 42
        assert isinstance(detector.detector, AnomalyDetector)

    def test_empirical_default_is_classical_in_all_modes(self):
        """Weighting does not change the public empirical default."""
        weighted = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
        )
        unweighted = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )

        for detector in (weighted, unweighted):
            assert isinstance(detector.estimation, Empirical)
            assert detector.estimation._tie_break.value == "classical"

    def test_init_copies_strategy_to_isolate_external_mutation(self):
        """External strategy mutations after init do not affect detector."""
        strategy = Split(n_calib=0.2)
        detector = ConformalDetector(
            detector=MockDetector(), strategy=strategy, seed=42
        )
        strategy._calib_size = 0.5
        assert detector.strategy.calib_size == 0.2

    @pytest.mark.parametrize(
        ("aggregation", "expected"), [("mean", "mean"), ("  MEDIAN ", "median")]
    )
    def test_init_normalizes_aggregation(self, aggregation, expected):
        """Aggregation strings are stored in normalized form."""
        detector = ConformalDetector(
            detector=MockDetector(),
            strategy=Split(n_calib=0.2),
            aggregation=aggregation,
        )
        assert detector.aggregation == expected

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"seed": -1}, ValueError, "non-negative"),
            ({"aggregation": 1}, TypeError, "aggregation method must be a string"),
            ({"aggregation": "avg"}, ValueError, "Unsupported aggregation method"),
            ({"verbose": 1}, TypeError, "verbose must be a boolean value"),
            (
                {"verify_prepared_batch_content": 1},
                TypeError,
                "verify_prepared_batch_content must be a boolean value",
            ),
            (
                {"score_polarity": "invalid"},
                ValueError,
                "Invalid score_polarity value",
            ),
        ],
        ids=[
            "negative-seed",
            "non-string-aggregation",
            "unknown-aggregation",
            "non-boolean-verbose",
            "non-boolean-batch-verification",
            "invalid-score-polarity",
        ],
    )
    def test_init_rejects_invalid_arguments(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            ConformalDetector(
                detector=MockDetector(), strategy=Split(n_calib=0.2), **kwargs
            )

    @pytest.mark.parametrize(
        ("make_detector", "polarity", "expected"),
        [
            (MockDetector, "higher_is_anomalous", ScorePolarity.HIGHER_IS_ANOMALOUS),
            (
                MockDetector,
                ScorePolarity.HIGHER_IS_ANOMALOUS,
                ScorePolarity.HIGHER_IS_ANOMALOUS,
            ),
            (MockDetector, None, ScorePolarity.HIGHER_IS_ANOMALOUS),
            (OneClassSVM, None, ScorePolarity.HIGHER_IS_NORMAL),
        ],
        ids=["literal", "enum", "omitted-custom", "omitted-sklearn-normality"],
    )
    def test_init_resolves_score_polarity(self, make_detector, polarity, expected):
        """Explicit polarity is parsed; omitted polarity uses safe defaults."""
        kwargs = {} if polarity is None else {"score_polarity": polarity}
        detector = _ConformalDetector(
            detector=make_detector(), strategy=Split(n_calib=0.2), **kwargs
        )
        assert detector.score_polarity is expected
        assert detector.get_params(deep=False)["score_polarity"] == polarity

    def test_init_auto_unknown_detector_raises(self):
        """AUTO mode fails fast for unknown detector classes."""
        with pytest.raises(ValueError) as exc_info:
            _ConformalDetector(
                detector=MockDetector(),
                strategy=Split(n_calib=0.2),
                score_polarity="auto",
            )
        message = str(exc_info.value)
        assert (
            "Unable to infer score polarity automatically in strict auto mode"
            in message
        )
        assert "higher_is_anomalous" in message
        assert "higher_is_normal" in message


class TestConformalDetectorFit:
    """Tests for ConformalDetector.fit()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 5))

    @pytest.mark.parametrize(
        ("make_input", "fit_kwargs"),
        [
            (np.asarray, {}),
            (pd.DataFrame, {}),
            (np.asarray, {"y": np.zeros(100)}),
        ],
        ids=["array", "dataframe", "unused-y"],
    )
    def test_fit_returns_self_with_models_and_calibration(
        self, sample_data, make_input, fit_kwargs
    ):
        """fit() accepts arrays, DataFrames, and an ignored sklearn-style y."""
        detector = _split_detector()

        assert detector.fit(make_input(sample_data), **fit_kwargs) is detector
        assert detector.is_fitted
        assert len(detector.detector_set) == 1
        assert len(detector.calibration_set) == 20

    def test_fit_passes_n_jobs_to_supported_strategy(self, sample_data):
        """fit(n_jobs=...) works when strategy exposes n_jobs."""
        detector = _split_detector(strategy=JackknifeBootstrap(n_bootstraps=3))
        detector.fit(sample_data, n_jobs=1)
        assert detector.is_fitted

    def test_fit_n_jobs_with_unsupported_strategy_raises(self, sample_data):
        """fit(n_jobs=...) raises for strategies without n_jobs support."""
        with pytest.raises(ValueError, match="does not support n_jobs"):
            _split_detector().fit(sample_data, n_jobs=2)


class TestConformalDetectorCalibrate:
    """Tests for detached calibration on pre-fitted detectors."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(1337)
        x_train = rng.standard_normal((80, 4))
        x_calib = rng.standard_normal((30, 4))
        x_test = rng.standard_normal((12, 4))
        return x_train, x_calib, x_test

    @pytest.mark.parametrize("make_input", [np.asarray, pd.DataFrame])
    def test_calibrate_prefitted_detector_returns_self(self, sample_data, make_input):
        x_train, x_calib, _ = sample_data
        base = StrictFittableDetector().fit(x_train)
        detector = ConformalDetector(detector=base, strategy=Split(n_calib=0.2), seed=7)

        assert detector.calibrate(make_input(x_calib)) is detector
        assert detector.is_fitted
        assert len(detector.detector_set) == 1
        assert len(detector.calibration_set) == len(x_calib)

    def test_calibrate_rejects_non_split_strategy(self, sample_data):
        x_train, x_calib, _ = sample_data
        base = StrictFittableDetector().fit(x_train)
        detector = ConformalDetector(
            detector=base,
            strategy=JackknifeBootstrap(n_bootstraps=3),
            seed=7,
        )

        with pytest.raises(ValueError, match="supported only with Split"):
            detector.calibrate(x_calib)

    @pytest.mark.parametrize(
        ("error", "expected", "match"),
        [
            (NotFittedError("unfitted"), NotFittedError, "Base detector is not fitted"),
            (
                AttributeError("'Model' object has no attribute 'center_'"),
                NotFittedError,
                "Base detector is not fitted",
            ),
            (RuntimeError("scoring backend failed"), RuntimeError, "backend failed"),
        ],
        ids=["not-fitted-error", "missing-fitted-attribute", "unrelated-error"],
    )
    def test_calibrate_translates_only_unfitted_detector_errors(
        self, sample_data, monkeypatch, error, expected, match
    ):
        _, x_calib, _ = sample_data
        detector = ConformalDetector(
            detector=StrictFittableDetector(),
            strategy=Split(n_calib=0.2),
            seed=7,
        )

        def failing_decision_function(X):
            raise error

        monkeypatch.setattr(
            detector.detector, "decision_function", failing_decision_function
        )
        with pytest.raises(expected, match=match):
            detector.calibrate(x_calib)

    def test_calibrate_is_optional_for_base_detector_subclasses(self, sample_data):
        class MinimalDetector(BaseConformalDetector):
            def fit(self, x, y=None, *, n_jobs=None):
                return self

            def compute_p_values(self, x):
                return np.ones(len(x))

            def score_samples(self, x):
                return np.zeros(len(x))

        with pytest.raises(NotImplementedError, match="calibrate"):
            MinimalDetector().calibrate(sample_data[1])

    def test_compute_p_values_after_calibrate(self, sample_data):
        x_train, x_calib, x_test = sample_data
        base = StrictFittableDetector().fit(x_train)
        detector = ConformalDetector(detector=base, strategy=Split(n_calib=0.2), seed=7)

        detector.calibrate(x_calib)
        p_values = detector.compute_p_values(x_test)
        assert len(p_values) == len(x_test)
        assert np.all((0 <= p_values) & (p_values <= 1))
        assert detector.compute_p_value(x_test[0]) == float(p_values[0])

    def test_calibrate_weighted_mode_stores_calibration_samples(self, sample_data):
        x_train, x_calib, x_test = sample_data
        base = StrictFittableDetector().fit(x_train)
        detector = ConformalDetector(
            detector=base,
            strategy=Split(n_calib=0.2),
            weight_estimator=CountingWeightEstimator(),
            seed=7,
        )

        detector.calibrate(x_calib)
        np.testing.assert_array_equal(detector.calibration_samples, x_calib)

        p_values = detector.compute_p_values(x_test)
        assert len(p_values) == len(x_test)


class TestConformalDetectorPredict:
    """Tests for ConformalDetector prediction interfaces."""

    def test_predict_before_fit_raises(self):
        """compute_p_values() before fit() raises NotFittedError."""
        detector = ConformalDetector(
            detector=MockDetector(), strategy=Split(n_calib=0.2)
        )
        with pytest.raises(NotFittedError, match="not fitted"):
            detector.compute_p_values(np.array([[1, 2, 3, 4, 5]]))
        with pytest.raises(NotFittedError, match="not fitted"):
            detector.compute_p_value(np.array([1, 2, 3, 4, 5]))

    def test_compute_p_values_is_repeatable_and_records_last_result(
        self, fitted_detector
    ):
        """compute_p_values() returns stable p-values and stores last_result."""
        rng = np.random.default_rng(42)
        X_test = rng.standard_normal((10, 5))
        assert fitted_detector.last_result is None

        p_values_1 = fitted_detector.compute_p_values(X_test)
        p_values_2 = fitted_detector.compute_p_values(X_test)

        assert len(p_values_1) == 10
        assert np.all((0 <= p_values_1) & (p_values_1 <= 1))
        np.testing.assert_array_equal(p_values_1, p_values_2)
        np.testing.assert_array_equal(fitted_detector.last_result.p_values, p_values_2)
        assert len(fitted_detector.score_samples(X_test)) == 10

    def test_compute_p_value_matches_one_row_batch(self, fitted_detector):
        """compute_p_value() returns the equivalent one-row batch value."""
        x = pd.Series(np.arange(5, dtype=float))

        expected = float(fitted_detector.compute_p_values(x.to_numpy()[None, :])[0])
        actual = fitted_detector.compute_p_value(x)

        assert isinstance(actual, float)
        assert actual == expected
        assert fitted_detector.last_result is not None
        assert fitted_detector.last_result.p_values.shape == (1,)

    @pytest.mark.parametrize(
        ("x", "match"),
        [
            (np.array(1.0), "one-dimensional feature vector"),
            (np.zeros((1, 5)), "one-dimensional feature vector"),
            (np.zeros((2, 5)), "one-dimensional feature vector"),
            (np.zeros(4), r"4 features.*5 features"),
        ],
        ids=["scalar", "one-row-matrix", "two-row-matrix", "wrong-feature-count"],
    )
    def test_compute_p_value_rejects_invalid_vectors(self, fitted_detector, x, match):
        """compute_p_value() accepts only fitted-width feature vectors."""
        with pytest.raises(ValueError, match=match):
            fitted_detector.compute_p_value(x)

    @pytest.mark.parametrize(
        ("method", "kwargs", "name"),
        [
            ("compute_p_values", {}, "p_value"),
            ("score_samples", {}, "score"),
            ("select", {"alpha": 0.2}, "selected"),
        ],
    )
    def test_dataframe_input_returns_indexed_series(
        self, fitted_detector, method, kwargs, name
    ):
        """DataFrame inputs return named Series aligned to the input index."""
        rng = np.random.default_rng(42)
        index = pd.RangeIndex(start=100, stop=110, step=1)
        x_test = pd.DataFrame(rng.standard_normal((10, 5)), index=index)

        output = getattr(fitted_detector, method)(x_test, **kwargs)

        assert isinstance(output, pd.Series)
        assert output.name == name
        assert output.index.equals(index)
        np.testing.assert_array_equal(
            output.to_numpy(),
            getattr(fitted_detector, method)(x_test.to_numpy(), **kwargs),
        )

    def test_select_matches_bh_from_scipy(self, fitted_detector):
        """select() should match BH selection from SciPy on p-values."""
        rng = np.random.default_rng(42)
        x_test = rng.standard_normal((10, 5))
        p_values = fitted_detector.compute_p_values(x_test)
        expected = false_discovery_control(p_values, method="bh") <= 0.2
        selected = fitted_detector.select(x_test, alpha=0.2)
        assert selected.dtype == bool
        np.testing.assert_array_equal(selected, expected)

    def test_select_matches_bh_with_tied_p_values(self, fitted_detector, monkeypatch):
        """select() should match BH behavior when p-values contain ties."""
        tied_p_values = np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2, 0.7, 1.0])

        def fake_compute_p_values(
            x: np.ndarray,
            *,
            refit_weights: bool = True,
        ) -> np.ndarray:
            _ = refit_weights
            fitted_detector._last_result = ConformalResult(
                p_values=tied_p_values.copy(),
                test_scores=np.zeros(len(tied_p_values), dtype=float),
                calib_scores=np.zeros(1, dtype=float),
                test_weights=None,
                calib_weights=None,
                metadata={},
            )
            return tied_p_values.copy()

        monkeypatch.setattr(fitted_detector, "compute_p_values", fake_compute_p_values)
        expected = false_discovery_control(tied_p_values, method="bh") <= 0.2
        selected = fitted_detector.select(
            np.zeros((len(tied_p_values), 5), dtype=float),
            alpha=0.2,
        )
        np.testing.assert_array_equal(selected, expected)

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.2])
    def test_select_rejects_invalid_alpha_before_prediction(
        self, fitted_detector, alpha
    ):
        """select() should fail fast for alpha outside (0, 1)."""
        rng = np.random.default_rng(42)
        x_test = rng.standard_normal((10, 5))

        assert fitted_detector.last_result is None
        with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\)"):
            fitted_detector.select(x_test, alpha=alpha)
        assert fitted_detector.last_result is None


class TestConformalDetectorProperties:
    """Tests for ConformalDetector properties."""

    def test_fitted_state_properties_return_defensive_copies(self, fitted_detector):
        """detector_set and calibration_set return fresh containers."""
        assert fitted_detector.detector_set is not fitted_detector.detector_set
        assert fitted_detector.calibration_set is not fitted_detector.calibration_set
        assert len(fitted_detector.calibration_samples) == 0

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

    @pytest.mark.parametrize(
        ("seeds", "same"),
        [((42, 42), True), ((42, 123), False)],
        ids=["same-seed", "different-seed"],
    )
    def test_seed_controls_p_values(self, seeds, same):
        """The detector seed reaches the base model and fixes the p-values."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((100, 5))
        X_test = rng.standard_normal((20, 5))

        p_values = [
            ConformalDetector(
                detector=SeedAwareDetector(), strategy=Split(n_calib=0.2), seed=seed
            )
            .fit(X_train.copy())
            .compute_p_values(X_test.copy())
            for seed in seeds
        ]

        assert np.allclose(*p_values) is same


class TestConformalDetectorWeightedPreparation:
    """Tests for explicit weighted preparation APIs."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(123)
        return rng.standard_normal((120, 4)), rng.standard_normal((30, 4))

    def test_prepare_weights_requires_weighted_mode(self, sample_data):
        """prepare_weights_for() raises when weighted mode is disabled."""
        x_train, x_test = sample_data
        detector = _split_detector()
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
        detector = _split_detector(weight_estimator=CountingWeightEstimator())
        detector.fit(x_train)

        with pytest.raises(RuntimeError, match="Weights are not prepared"):
            detector.compute_p_values(x_test, refit_weights=False)

    def test_compute_p_value_rejects_weighted_mode(self, sample_data):
        """Single-observation inference must not fit unstable density ratios."""
        x_train, x_test = sample_data
        weight_estimator = CountingWeightEstimator()
        detector = _split_detector(weight_estimator=weight_estimator)
        detector.fit(x_train)

        with pytest.raises(RuntimeError, match="representative test batch"):
            detector.compute_p_value(x_test[0])
        assert weight_estimator.fit_calls == 0

    @pytest.mark.parametrize(
        "prepare",
        [
            lambda detector, x: detector.prepare_weights_for(x),
            lambda detector, x: detector.compute_p_values(x, refit_weights=True),
        ],
        ids=["prepare-weights-for", "refit-prediction"],
    )
    def test_prepared_weights_are_reused_without_refit(self, sample_data, prepare):
        """Explicit preparation and refit predictions both prepare reusable weights."""
        x_train, x_test = sample_data
        weight_estimator = CountingWeightEstimator()
        detector = _split_detector(weight_estimator=weight_estimator).fit(x_train)

        prepare(detector, x_test)
        assert weight_estimator.fit_calls == 1
        _ = detector.compute_p_values(x_test, refit_weights=False)
        assert weight_estimator.fit_calls == 1

    def test_prepare_nullable_dataframe_then_reuse_weights(self, sample_data):
        """Content verification accepts equal nullable numeric DataFrames."""
        x_train, x_test = sample_data
        batch = pd.DataFrame(x_test, dtype="Float64")
        weight_estimator = CountingWeightEstimator()
        detector = ConformalDetector(
            detector=MockDetector(np.arange(100, dtype=float)),
            strategy=Split(n_calib=0.2),
            weight_estimator=weight_estimator,
            seed=42,
        ).fit(x_train)
        detector.prepare_weights_for(batch)

        detector.compute_p_values(batch.copy(), refit_weights=False)

        assert weight_estimator.fit_calls == 1
        with pytest.raises(ValueError, match="batch content"):
            detector.compute_p_values(batch + 1.0, refit_weights=False)

    @pytest.mark.parametrize(
        ("change_batch", "match"),
        [(lambda x: x[:10], "batch size"), (lambda x: x + 1.0, "batch content")],
        ids=["size", "content"],
    )
    def test_reuse_with_different_batch_raises(self, sample_data, change_batch, match):
        """Prepared weights cannot be reused for a different batch."""
        x_train, x_test = sample_data
        detector = _split_detector(weight_estimator=CountingWeightEstimator())
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        with pytest.raises(ValueError, match=match):
            detector.compute_p_values(change_batch(x_test), refit_weights=False)

    def test_reuse_with_different_content_allowed_when_verification_disabled(
        self, sample_data
    ):
        """Same-size reuse can skip content check when explicitly disabled."""
        x_train, x_test = sample_data
        detector = _split_detector(
            weight_estimator=CountingWeightEstimator(),
            verify_prepared_batch_content=False,
        )
        detector.fit(x_train)
        detector.prepare_weights_for(x_test)

        p_values = detector.compute_p_values(x_test + 1.0, refit_weights=False)
        assert len(p_values) == len(x_test)

    @pytest.mark.parametrize(
        ("detector_seed", "select_seed", "expected_seed"),
        [(42, 7, _derive_wcs_pruning_seed(7)), (None, None, None)],
        ids=["explicit-seed", "unseeded"],
    )
    def test_select_passes_weighted_pruning_seed(
        self, sample_data, monkeypatch, detector_seed, select_seed, expected_seed
    ):
        """Weighted select() forwards pruning and seed to weighted FDR."""
        x_train, x_test = sample_data
        detector = _split_detector(
            weight_estimator=CountingWeightEstimator(), seed=detector_seed
        )
        detector.fit(x_train)

        call_args: dict[str, object] = {}

        def fake_weighted_fdr(
            *,
            result,
            alpha: float,
            pruning: Pruning,
            seed: int | None,
        ) -> np.ndarray:
            call_args["result"] = result
            call_args["alpha"] = alpha
            call_args["pruning"] = pruning
            call_args["seed"] = seed
            assert result is not None and result.p_values is not None
            return np.zeros(len(result.p_values), dtype=bool)

        monkeypatch.setattr(
            "nonconform.fdr.weighted_false_discovery_control", fake_weighted_fdr
        )
        selected = detector.select(
            x_test,
            alpha=0.1,
            pruning=Pruning.HETEROGENEOUS,
            seed=select_seed,
        )
        assert selected.dtype == bool
        assert len(selected) == len(x_test)
        assert call_args["alpha"] == 0.1
        assert call_args["pruning"] is Pruning.HETEROGENEOUS
        assert call_args["seed"] == expected_seed

    def test_select_accepts_classical_empirical_wcs(self, sample_data):
        """Weighted select accepts classical empirical p-values."""
        x_train, x_test = sample_data
        detector = ConformalDetector(
            detector=MockDetector(np.arange(100, dtype=float)),
            strategy=Split(n_calib=0.2),
            estimation=Empirical(tie_break="classical"),
            weight_estimator=CountingWeightEstimator(),
            seed=42,
        )
        detector.fit(x_train)

        selected = detector.select(x_test, alpha=0.1)

        assert selected.shape == (len(x_test),)
        assert selected.dtype == bool

    def test_select_uses_detector_seed_by_default_in_weighted_mode(
        self, sample_data, monkeypatch
    ):
        """Weighted select derives a stable pruning stream from detector seed."""
        x_train, x_test = sample_data
        detector = _split_detector(
            estimation=Empirical(tie_break="randomized"),
            weight_estimator=CountingWeightEstimator(),
            seed=123,
        )
        detector.fit(x_train)

        captured_seed: dict[str, int | None] = {"value": None}

        def fake_weighted_fdr(
            *,
            result,
            alpha: float,
            pruning: Pruning,
            seed: int | None,
        ) -> np.ndarray:
            captured_seed["value"] = seed
            assert result is not None and result.p_values is not None
            expected = calculate_weighted_p_val(
                result.test_scores,
                result.calib_scores,
                result.test_weights,
                result.calib_weights,
                tie_break="randomized",
                rng=np.random.default_rng(123),
            )
            np.testing.assert_allclose(result.p_values, expected)
            return np.zeros(len(result.p_values), dtype=bool)

        monkeypatch.setattr(
            "nonconform.fdr.weighted_false_discovery_control", fake_weighted_fdr
        )
        _ = detector.select(x_test, alpha=0.1, pruning=Pruning.HOMOGENEOUS)
        assert captured_seed["value"] == _derive_wcs_pruning_seed(123)
        assert captured_seed["value"] != 123


class TestConformalDetectorSklearnParams:
    """Tests for sklearn-style parameter protocol integration."""

    def test_nested_detector_params_are_exposed_and_updatable(self):
        """get_params/set_params handle detector__* nested keys."""
        detector = _ConformalDetector(
            detector=OneClassSVM(gamma=0.1),
            strategy=Split(n_calib=0.2),
            score_polarity="auto",
        )
        params = detector.get_params(deep=True)
        assert params["detector__gamma"] == 0.1
        assert "strategy" in params

        detector.set_params(detector__gamma=0.5)
        assert detector.get_params(deep=True)["detector__gamma"] == 0.5

    def test_nested_params_support_get_params_without_deep_argument(self):
        detector = ConformalDetector(
            detector=NoDeepParamsDetector(), strategy=Split(n_calib=0.2)
        )
        assert detector.get_params(deep=True)["detector__threshold"] == 1.0

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"unknown": 1}, "Invalid parameter 'unknown' for ConformalDetector"),
            ({"unknown__value": 1}, "Invalid parameter 'unknown'"),
            ({"weight_estimator__clip_quantile": 0.1}, "Cannot set nested parameters"),
        ],
        ids=["top-level", "nested-component", "missing-component"],
    )
    def test_set_params_rejects_unknown_parameters(self, params, match):
        detector = ConformalDetector(detector=MockDetector(), strategy=Split(0.2))
        with pytest.raises(ValueError, match=match):
            detector.set_params(**params)

    def test_params_follow_sklearn_and_runtime_is_independent(self):
        """get_params returns the passed objects; fitting uses separate copies."""
        external_detector = OneClassSVM(gamma=0.1)
        external_strategy = Split(n_calib=0.2)
        detector = _ConformalDetector(
            detector=external_detector,
            strategy=external_strategy,
            score_polarity="auto",
        )
        params = detector.get_params(deep=False)
        assert params["detector"] is external_detector
        assert params["strategy"] is external_strategy

        external_detector.set_params(gamma=0.9)
        external_strategy._calib_size = 0.5

        assert detector.get_params(deep=True)["detector__gamma"] == 0.1
        assert detector.strategy.calib_size == 0.2
        cloned = clone(detector)
        assert isinstance(cloned, _ConformalDetector)
        assert cloned.get_params(deep=True)["detector__gamma"] == 0.9
        assert cloned.strategy.calib_size == 0.5
