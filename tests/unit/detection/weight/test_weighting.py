import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from nonconform.weighting import (
    DEFAULT_CLIP_BOUNDS,
    EPSILON,
    BaseWeightEstimator,
    BootstrapBaggedWeightEstimator,
    IdentityWeightEstimator,
    SklearnWeightEstimator,
    forest_weight_estimator,
    logistic_weight_estimator,
)


class DummyStoredEstimator(BaseWeightEstimator):
    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        self._stored_calib = np.full(calibration_samples.shape[0], 2.0)
        self._stored_test = np.full(test_samples.shape[0], 3.0)
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self._stored_calib.copy(), self._stored_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(calibration_samples.shape[0], 1.5),
            np.full(test_samples.shape[0], 2.5),
        )


class FixedProbClassifier(BaseEstimator):
    def __init__(
        self,
        proba0: float = 0.8,
        proba1: float = 0.2,
        class_order: tuple[int, int] = (0, 1),
    ) -> None:
        self.proba0 = proba0
        self.proba1 = proba1
        self.class_order = class_order
        self.random_state = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FixedProbClassifier":
        self.classes_ = np.array(self.class_order)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_map = {0: self.proba0, 1: self.proba1}
        return np.array(
            [[proba_map[c] for c in self.classes_] for _ in range(len(X))], dtype=float
        )


class SeedAwareClassifier(BaseEstimator):
    def __init__(self) -> None:
        self.random_state: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeedAwareClassifier":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        probs = rng.uniform(0.1, 0.9, size=len(X))
        return np.column_stack([1.0 - probs, probs])


class RecordingWeightEstimator(BaseWeightEstimator):
    def __init__(self, recorder: list[tuple[int, int]]) -> None:
        self.recorder = recorder
        self._is_fitted = False
        self._seed: int | None = None

    def __deepcopy__(self, memo):
        new = type(self)(self.recorder)
        memo[id(self)] = new
        new._seed = getattr(self, "_seed", None)
        return new

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        self.recorder.append((len(calibration_samples), len(test_samples)))
        self._w_calib = np.ones(len(calibration_samples))
        self._w_test = np.ones(len(test_samples))
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(len(calibration_samples)), np.ones(len(test_samples))


class DeterministicWeightEstimator(BaseWeightEstimator):
    def __init__(self) -> None:
        self._seed: int | None = None
        self._is_fitted = False

    def fit(self, calibration_samples: np.ndarray, test_samples: np.ndarray) -> None:
        rng = np.random.default_rng(self._seed)
        self._w_calib = rng.uniform(1.0, 2.0, size=len(calibration_samples))
        self._w_test = rng.uniform(1.5, 2.5, size=len(test_samples))
        self._is_fitted = True

    def _get_stored_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return self._w_calib.copy(), self._w_test.copy()

    def _score_new_data(
        self, calibration_samples: np.ndarray, test_samples: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self._seed)
        w_calib = rng.uniform(1.0, 2.0, size=len(calibration_samples))
        w_test = rng.uniform(1.5, 2.5, size=len(test_samples))
        return w_calib, w_test


class NoPredictProbaEstimator:
    def fit(self, X, y):
        return self


def _simple_data(n_calib: int = 6, n_test: int = 4) -> tuple[np.ndarray, np.ndarray]:
    calib = np.stack([np.linspace(0, 1, n_calib), np.linspace(1, 2, n_calib)], axis=1)
    test = np.stack([np.linspace(2, 3, n_test), np.linspace(3, 4, n_test)], axis=1)
    return calib, test


class TestBaseWeightEstimatorHelpers:
    def test_get_weights_requires_fit(self):
        estimator = IdentityWeightEstimator()
        with pytest.raises(RuntimeError):
            estimator.get_weights()

    def test_get_weights_requires_matching_inputs(self):
        estimator = DummyStoredEstimator()
        calib, test = _simple_data()
        estimator.fit(calib, test)
        with pytest.raises(ValueError):
            estimator.get_weights(calib)

    def test_get_weights_returns_stored_and_new_scores(self):
        estimator = DummyStoredEstimator()
        calib, test = _simple_data()
        estimator.fit(calib, test)

        stored_calib, stored_test = estimator.get_weights()
        assert np.allclose(stored_calib, 2.0)
        assert np.allclose(stored_test, 3.0)

        new_calib = np.ones((3, 2))
        new_test = np.ones((5, 2))
        scored_calib, scored_test = estimator.get_weights(new_calib, new_test)
        assert np.allclose(scored_calib, 1.5)
        assert np.allclose(scored_test, 2.5)

        stored_calib[0] = 99.0
        refreshed_calib, _ = estimator.get_weights()
        assert refreshed_calib[0] == 2.0

    def test_prepare_training_data_labels_and_seed(self):
        calib = np.array([[0.0], [1.0], [2.0]])
        test = np.array([[10.0], [11.0], [12.0]])
        x1, y1 = BaseWeightEstimator._prepare_training_data(calib, test, seed=42)
        x2, y2 = BaseWeightEstimator._prepare_training_data(calib, test, seed=42)

        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)
        assert (y1 == 0).sum() == len(calib)
        assert (y1 == 1).sum() == len(test)
        assert {tuple(row) for row in x1[y1 == 0]} == {tuple(row) for row in calib}
        assert {tuple(row) for row in x1[y1 == 1]} == {tuple(row) for row in test}

    def test_compute_clip_bounds_and_clipping(self):
        w_calib = np.array([0.1, 0.5, 1.0, 2.0])
        w_test = np.array([3.0, 4.0])
        lower, upper = BaseWeightEstimator._compute_clip_bounds(
            w_calib, w_test, clip_quantile=0.2
        )
        assert lower == np.percentile(np.concatenate([w_calib, w_test]), 20)
        assert upper == np.percentile(np.concatenate([w_calib, w_test]), 80)

        fallback = BaseWeightEstimator._compute_clip_bounds(
            w_calib, w_test, clip_quantile=None
        )
        assert fallback == DEFAULT_CLIP_BOUNDS

        clipped_calib, clipped_test = BaseWeightEstimator._clip_weights(
            w_calib, w_test, (0.5, 2.5)
        )
        assert np.all(clipped_calib >= 0.5)
        assert np.all(clipped_test <= 2.5)


class TestIdentityWeightEstimator:
    def test_identity_returns_unit_weights(self):
        calib, test = _simple_data(4, 3)
        estimator = IdentityWeightEstimator()
        estimator.fit(calib, test)
        w_calib, w_test = estimator.get_weights()
        assert np.allclose(w_calib, 1.0)
        assert np.allclose(w_test, 1.0)
        assert len(w_calib) == len(calib)
        assert len(w_test) == len(test)

    def test_identity_scores_new_data_and_returns_copies(self):
        calib, test = _simple_data(2, 2)
        estimator = IdentityWeightEstimator()
        estimator.fit(calib, test)

        new_calib = np.ones((5, 1))
        new_test = np.ones((6, 1))
        w_calib, w_test = estimator.get_weights(new_calib, new_test)
        assert w_calib.shape[0] == 5
        assert w_test.shape[0] == 6
        assert np.allclose(w_calib, 1.0)
        assert np.allclose(w_test, 1.0)

        w_calib[0] = 5.0
        fresh_calib, _ = estimator.get_weights()
        assert fresh_calib[0] == 1.0


class TestSklearnWeightEstimator:
    def test_fit_sets_weights_and_clip_bounds(self):
        calib, test = _simple_data(10, 6)
        estimator = SklearnWeightEstimator(clip_quantile=0.1)
        estimator.fit(calib, test)

        assert estimator._is_fitted is True
        w_calib, w_test = estimator.get_weights()
        assert np.all(w_calib > 0)
        assert np.all(w_test > 0)
        raw_w_calib, raw_w_test = estimator._compute_weights(calib, test)
        expected_bounds = BaseWeightEstimator._compute_clip_bounds(
            raw_w_calib, raw_w_test, clip_quantile=0.1
        )
        assert estimator._clip_bounds == expected_bounds

        w_calib[0] = 123.0
        refreshed, _ = estimator.get_weights()
        assert refreshed[0] != 123.0

    def test_score_new_data_respects_clipping(self):
        calib, test = _simple_data(4, 4)
        estimator = SklearnWeightEstimator(
            base_estimator=FixedProbClassifier(proba0=0.01, proba1=0.99),
            clip_quantile=None,
        )
        estimator.fit(calib, test)

        w_calib, w_test = estimator.get_weights(np.ones((3, 2)), np.ones((3, 2)))
        assert np.all(w_calib <= DEFAULT_CLIP_BOUNDS[1] + 1e-12)
        assert np.all(w_test <= DEFAULT_CLIP_BOUNDS[1] + 1e-12)

    def test_init_without_predict_proba_raises(self):
        with pytest.raises(ValueError):
            SklearnWeightEstimator(base_estimator=NoPredictProbaEstimator())

    def test_fit_with_empty_calibration_raises(self):
        estimator = SklearnWeightEstimator()
        with pytest.raises(ValueError):
            estimator.fit(np.empty((0, 2)), np.ones((3, 2)))

    def test_respects_class_ordering(self):
        calib, test = _simple_data(3, 2)
        estimator = SklearnWeightEstimator(
            base_estimator=FixedProbClassifier(
                proba0=0.2, proba1=0.8, class_order=(1, 0)
            ),
            clip_quantile=None,
        )
        estimator.fit(calib, test)
        w_calib, w_test = estimator.get_weights()
        assert np.allclose(w_calib, 4.0)
        assert np.allclose(w_test, 4.0)

    def test_eps_prevents_division_by_zero(self):
        calib, test = _simple_data(2, 2)
        estimator = SklearnWeightEstimator(
            base_estimator=FixedProbClassifier(proba0=0.0, proba1=1.0),
            clip_quantile=0.05,
        )
        estimator.fit(calib, test)
        raw_calib, raw_test = estimator._compute_weights(calib, test)
        assert np.isfinite(raw_calib).all()
        assert np.isclose(raw_calib[0], 1.0 / EPSILON)
        assert np.isclose(raw_test[0], 1.0 / EPSILON)

    def test_seed_propagation_and_determinism(self):
        calib, test = _simple_data(5, 4)
        est1 = SklearnWeightEstimator(
            base_estimator=SeedAwareClassifier(), clip_quantile=None
        )
        est1.set_seed(99)
        est1.fit(calib, test)
        w1_calib, w1_test = est1.get_weights()

        est2 = SklearnWeightEstimator(
            base_estimator=SeedAwareClassifier(), clip_quantile=None
        )
        est2.set_seed(99)
        est2.fit(calib, test)
        w2_calib, w2_test = est2.get_weights()

        assert est1.estimator_.random_state == 99
        assert np.allclose(w1_calib, w2_calib)
        assert np.allclose(w1_test, w2_test)

    def test_seed_applied_inside_pipeline(self):
        calib, test = _simple_data(6, 4)
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=50))
        estimator = SklearnWeightEstimator(base_estimator=pipeline, clip_quantile=None)
        estimator.set_seed(7)
        estimator.fit(calib, test)
        assert isinstance(estimator.estimator_, Pipeline)
        lr = estimator.estimator_.named_steps["logisticregression"]
        assert lr.random_state == 7


class TestBootstrapBaggedWeightEstimator:
    def test_invalid_constructor_args_raise(self):
        with pytest.raises(ValueError):
            BootstrapBaggedWeightEstimator(IdentityWeightEstimator(), n_bootstrap=0)
        with pytest.raises(ValueError):
            BootstrapBaggedWeightEstimator(
                IdentityWeightEstimator(), n_bootstrap=2, clip_quantile=0.5
            )

    def test_fit_requires_nonempty_calibration(self):
        bagged = BootstrapBaggedWeightEstimator(
            IdentityWeightEstimator(), n_bootstrap=2
        )
        with pytest.raises(ValueError):
            bagged.fit(np.empty((0, 2)), np.ones((2, 2)))

    def test_bootstrap_samples_are_balanced(self):
        recorder: list[tuple[int, int]] = []
        base = RecordingWeightEstimator(recorder)
        bagged = BootstrapBaggedWeightEstimator(
            base_estimator=base, n_bootstrap=3, clip_quantile=None
        )
        bagged.fit(np.ones((5, 2)), np.ones((3, 2)))
        assert recorder == [(3, 3), (3, 3), (3, 3)]

    def test_score_new_data_shape_mismatch_raises(self):
        bagged = BootstrapBaggedWeightEstimator(
            base_estimator=IdentityWeightEstimator(), n_bootstrap=2
        )
        bagged.fit(np.ones((4, 1)), np.ones((2, 1)))
        with pytest.raises(NotImplementedError):
            bagged.get_weights(np.ones((5, 1)), np.ones((2, 1)))

    def test_weight_counts_and_copy(self):
        bagged = BootstrapBaggedWeightEstimator(
            base_estimator=IdentityWeightEstimator(), n_bootstrap=2
        )
        bagged.fit(np.ones((3, 1)), np.ones((3, 1)))
        w_calib, _ = bagged.get_weights()
        w_calib[0] = 10.0
        fresh_calib, _ = bagged.get_weights()
        assert fresh_calib[0] == 1.0
        info = bagged.weight_counts
        assert "Bootstrap iterations: 2" in info
        assert "Calibration instances: 3" in info
        assert "Test instances: 3" in info

    def test_geometric_mean_aggregation_and_clipping(self):
        base = DeterministicWeightEstimator()
        bagged = BootstrapBaggedWeightEstimator(
            base_estimator=base,
            n_bootstrap=3,
            clip_bounds=(0.5, 2.0),
            clip_quantile=None,
        )
        bagged.set_seed(7)
        n_calib, n_test = 4, 3
        calib = np.ones((n_calib, 1))
        test = np.ones((n_test, 1))
        bagged.fit(calib, test)

        w_calib, w_test = bagged.get_weights()
        seeds = [hash((i, 7)) % (2**32) for i in range(3)]
        log_c = np.zeros(n_calib)
        log_t = np.zeros(n_test)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            w_c_iter = rng.uniform(1.0, 2.0, size=n_calib)
            w_t_iter = rng.uniform(1.5, 2.5, size=n_test)
            log_c += np.log(w_c_iter)
            log_t += np.log(w_t_iter)
        expected_c = np.clip(np.exp(log_c / 3), 0.5, 2.0)
        expected_t = np.clip(np.exp(log_t / 3), 0.5, 2.0)
        assert np.allclose(w_calib, expected_c)
        assert np.allclose(w_test, expected_t)


class TestFactoryHelpers:
    def test_logistic_weight_estimator_factory(self):
        estimator = logistic_weight_estimator(
            regularization=0.5, clip_quantile=0.2, max_iter=50
        )
        assert isinstance(estimator, SklearnWeightEstimator)
        assert estimator.clip_quantile == 0.2
        assert isinstance(estimator.base_estimator, Pipeline)
        lr = estimator.base_estimator.named_steps["logisticregression"]
        assert isinstance(lr, LogisticRegression)
        assert lr.C == 0.5
        assert lr.max_iter == 50

    def test_forest_weight_estimator_factory(self):
        estimator = forest_weight_estimator(
            n_estimators=7, max_depth=4, min_samples_leaf=2, clip_quantile=0.15
        )
        assert isinstance(estimator, SklearnWeightEstimator)
        assert estimator.clip_quantile == 0.15
        assert isinstance(estimator.base_estimator, RandomForestClassifier)
        assert estimator.base_estimator.n_estimators == 7
        assert estimator.base_estimator.max_depth == 4
        assert estimator.base_estimator.min_samples_leaf == 2
