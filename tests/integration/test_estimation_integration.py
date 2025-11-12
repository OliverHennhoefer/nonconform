"""Integration tests for empirical vs probabilistic p-value estimators."""

from __future__ import annotations

import numpy as np
import pytest

from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.strategy.estimation.empirical import Empirical
from nonconform.strategy.estimation.probabilistic import Probabilistic
from nonconform.utils.func.enums import Kernel
from pyod.models.iforest import IForest


def _build_detector(estimation):
    return ConformalDetector(
        detector=IForest(n_estimators=20, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=estimation,
        seed=19,
    )


ESTIMATION_FACTORIES = [
    ("empirical", lambda: Empirical()),
    ("probabilistic", lambda: Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0)),
]


@pytest.mark.parametrize(
    ("name", "factory"),
    ESTIMATION_FACTORIES,
    ids=[case[0] for case in ESTIMATION_FACTORIES],
)
def test_estimation_methods_return_unit_interval(simple_dataset, name, factory):
    """Both estimation strategies should output valid p-values."""
    x_train, x_test, _ = simple_dataset(n_train=72, n_test=24, n_features=4)
    detector = _build_detector(factory())

    detector.fit(x_train)
    p_values = detector.predict(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))


def test_empirical_and_probabilistic_differ(simple_dataset):
    """Discrete vs smooth estimators should not be identical."""
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=30, n_features=5)

    empirical = _build_detector(Empirical())
    probabilistic = _build_detector(
        Probabilistic(
            kernel=[Kernel.GAUSSIAN, Kernel.TRIANGULAR], n_trials=2, cv_folds=2
        )
    )

    empirical.fit(x_train)
    probabilistic.fit(x_train)

    emp_vals = empirical.predict(x_test)
    prob_vals = probabilistic.predict(x_test)

    assert not np.allclose(emp_vals, prob_vals)


@pytest.mark.parametrize(
    ("name", "factory"),
    ESTIMATION_FACTORIES,
    ids=[case[0] for case in ESTIMATION_FACTORIES],
)
def test_weighted_estimation_populates_metadata(shifted_dataset, name, factory):
    """Weighted conformal runs should store weights and (for KDE) metadata."""
    x_train, x_test, _ = shifted_dataset(n_train=120, n_test=40, n_features=4)
    detector = ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=factory(),
        weight_estimator=LogisticWeightEstimator(seed=3),
        seed=3,
    )

    detector.fit(x_train)
    detector.predict(x_test)

    result = detector.last_result
    assert result is not None
    assert result.test_weights is not None
    assert result.calib_weights is not None
    if name == "probabilistic":
        assert "kde" in result.metadata
        assert "eval_grid" in result.metadata["kde"]
    else:
        assert result.metadata == {}
