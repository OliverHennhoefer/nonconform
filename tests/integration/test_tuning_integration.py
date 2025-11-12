"""Integration tests for KDE tuning inside probabilistic estimation."""

from __future__ import annotations

import numpy as np

from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import CrossValidation, Split
from nonconform.strategy.estimation.probabilistic import Probabilistic
from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.tuning import tune_kde_hyperparameters
from pyod.models.iforest import IForest


def test_probabilistic_tuning_records_metadata(simple_dataset):
    """Tuned probabilistic estimation should expose KDE metadata."""
    x_train, x_test, _ = simple_dataset(n_train=70, n_test=25, n_features=4)
    estimator = Probabilistic(
        kernel=[Kernel.GAUSSIAN, Kernel.TRIANGULAR],
        n_trials=3,
        cv_folds=2,
    )
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=estimator,
        seed=31,
    )
    detector.fit(x_train)
    detector.predict(x_test)

    assert estimator._tuned_params is not None
    result = detector.last_result
    assert result is not None
    assert "kde" in result.metadata
    kde_meta = result.metadata["kde"]
    assert "eval_grid" in kde_meta and "cdf_values" in kde_meta


def test_weighted_tuning_matches_total_weight(shifted_dataset):
    """Metadata total weight should equal the sum of calibration weights."""
    x_train, x_test, _ = shifted_dataset(n_train=110, n_test=36, n_features=4)
    detector = ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0),
        weight_estimator=LogisticWeightEstimator(seed=12),
        seed=12,
    )
    detector.fit(x_train)
    detector.predict(x_test)

    result = detector.last_result
    assert result is not None and result.metadata
    total_weight = result.metadata["kde"]["total_weight"]
    assert np.isclose(total_weight, np.sum(result.calib_weights))


def test_tuning_with_cross_validation_strategy(simple_dataset):
    """Probabilistic estimation with tuning should work with CV strategies."""
    x_train, x_test, _ = simple_dataset(n_train=66, n_test=24, n_features=4)
    detector = ConformalDetector(
        detector=IForest(n_estimators=20, max_samples=0.8, random_state=0),
        strategy=CrossValidation(k=3, plus=True),
        estimation=Probabilistic(
            kernel=[Kernel.GAUSSIAN, Kernel.BOX], n_trials=2, cv_folds=2
        ),
        seed=33,
    )
    detector.fit(x_train)
    p_values = detector.predict(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0 <= p_values) & (p_values <= 1))


def test_tuning_helper_respects_kernel_options():
    """Direct tune_kde_hyperparameters call should honor provided kernels."""
    rng = np.random.default_rng(0)
    calibration_set = rng.normal(size=80)
    result = tune_kde_hyperparameters(
        calibration_set=calibration_set,
        kernel_options=[Kernel.GAUSSIAN, Kernel.BOX],
        n_trials=1,
        cv_folds=2,
        seed=0,
    )
    assert result["kernel"] in {Kernel.GAUSSIAN, Kernel.BOX}
