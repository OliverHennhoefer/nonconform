"""Regression coverage for configuration and learned-state ownership."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM

from nonconform import (
    ConformalDetector,
    CrossValidation,
    DerandomizedSplits,
    Split,
    logistic_weight_estimator,
)
from tests.unit.detection.test_conformal import CountingWeightEstimator


@pytest.fixture
def batches():
    rng = np.random.default_rng(42)
    return rng.normal(size=(40, 2)), rng.normal(size=(8, 2))


@pytest.mark.parametrize("strategy", [Split(10), DerandomizedSplits(2, 10)])
def test_rejected_updates_preserve_live_state_and_clone_parameters(strategy, batches):
    reference, test = batches
    detector = ConformalDetector(OneClassSVM(gamma=0.1), strategy, seed=1).fit(
        reference
    )
    expected = detector.select(test)
    with pytest.raises(ValueError):
        detector.set_params(detector__gamma=0.7, aggregation="unknown")

    assert detector.is_fitted
    assert detector.get_params(deep=False)["aggregation"] == "median"
    assert detector.get_params()["detector__gamma"] == 0.1
    assert clone(detector).get_params()["detector__gamma"] == 0.1
    np.testing.assert_array_equal(detector.select(test), expected)


def test_rejected_strategy_switch_preserves_original_calibration(batches):
    reference, test = batches
    detector = ConformalDetector(
        OneClassSVM(gamma=0.1), Split(10), weight_estimator=CountingWeightEstimator()
    ).fit(reference)
    expected = detector.compute_p_values(test)
    with pytest.raises(ValueError, match="does not support weighting"):
        detector.set_params(strategy=DerandomizedSplits(2, 10))
    assert isinstance(detector.strategy, Split)
    assert detector.is_fitted
    np.testing.assert_array_equal(detector.compute_p_values(test), expected)


def test_constructor_parameters_do_not_expose_owned_clone_snapshots():
    detector = ConformalDetector(OneClassSVM(gamma=0.1), Split(10))
    params = detector.get_params(deep=False)
    params["detector"].set_params(gamma=0.9)
    params["strategy"]._calib_size = 3
    cloned = clone(detector)
    assert cloned.get_params()["detector__gamma"] == 0.1
    assert cloned.strategy.calib_size == 10


@pytest.mark.parametrize(
    "strategy", [Split(10), CrossValidation(3), DerandomizedSplits(2, 10)]
)
def test_failed_refit_cannot_expose_previous_or_partial_calibration(
    strategy, batches, monkeypatch
):
    reference, test = batches
    detector = ConformalDetector(OneClassSVM(), strategy, seed=1).fit(reference)
    expected = detector.select(test)

    def fail_fit(self, *args, **kwargs):
        raise RuntimeError("model fit failed")

    with monkeypatch.context() as patch:
        patch.setattr(OneClassSVM, "fit", fail_fit)
        with pytest.raises(RuntimeError, match="model fit failed"):
            detector.fit(reference)
    assert not detector.is_fitted
    assert detector.detector_set == []
    assert detector.calibration_set.size == detector.calibration_samples.size == 0
    assert detector.last_result is detector.last_selection_result is None
    with pytest.raises(NotFittedError):
        detector.select(test)
    np.testing.assert_array_equal(detector.fit(reference).select(test), expected)


def test_failed_detached_calibration_clears_learned_state(batches, monkeypatch):
    reference, test = batches
    detector = ConformalDetector(OneClassSVM(), Split(10)).fit(reference)
    detector.compute_p_values(test)
    monkeypatch.setattr(detector.detector, "decision_function", lambda x: np.zeros(2))
    with pytest.raises(ValueError, match="one value per calibration sample"):
        detector.calibrate(reference)
    assert not detector.is_fitted
    assert detector.last_result is None


def test_prepared_weights_own_arrays_even_if_external_estimator_is_refitted(batches):
    reference, test = batches
    estimator = CountingWeightEstimator()
    detector = ConformalDetector(
        OneClassSVM(), Split(10), weight_estimator=estimator
    ).fit(reference)
    detector.prepare_weights_for(test)
    expected = detector.compute_p_values(test, refit_weights=False)
    estimator.fit(reference[:2], test[:1])
    np.testing.assert_array_equal(
        detector.compute_p_values(test, refit_weights=False), expected
    )


def test_failed_weight_preparation_invalidates_previous_prepared_batch(
    batches, monkeypatch
):
    reference, test = batches
    estimator = CountingWeightEstimator()
    detector = ConformalDetector(
        OneClassSVM(), Split(10), weight_estimator=estimator
    ).fit(reference)
    detector.prepare_weights_for(test)

    def fail_fit(*args, **kwargs):
        raise RuntimeError("weight fit failed")

    monkeypatch.setattr(estimator, "fit", fail_fit)
    with pytest.raises(RuntimeError, match="weight fit failed"):
        detector.prepare_weights_for(test)
    with pytest.raises(RuntimeError, match="Weights are not prepared"):
        detector.compute_p_values(test, refit_weights=False)


@pytest.mark.parametrize("seed", [True, 1.5])
def test_non_integer_seed_is_rejected_before_configuration(seed):
    with pytest.raises(TypeError, match="non-negative integer"):
        ConformalDetector(OneClassSVM(), Split(10), seed=seed)


@pytest.mark.parametrize("strategy", [Split(10), DerandomizedSplits(2, 10)])
def test_owned_calibration_and_weight_state_support_current_pickle_roundtrips(
    strategy, batches
):
    reference, test = batches
    weighted = isinstance(strategy, Split)
    detector = ConformalDetector(
        OneClassSVM(),
        strategy,
        weight_estimator=logistic_weight_estimator() if weighted else None,
        seed=1,
    ).fit(reference)
    if weighted:
        detector.prepare_weights_for(test)
    expected = detector.select(test, refit_weights=False)
    restored = pickle.loads(pickle.dumps(detector))
    np.testing.assert_array_equal(restored.calibration_set, detector.calibration_set)
    np.testing.assert_array_equal(restored.select(test, refit_weights=False), expected)
