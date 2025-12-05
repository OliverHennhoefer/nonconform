"""Integration tests for covariate-shift weight estimators."""

import numpy as np
import pytest
from pyod.models.iforest import IForest

from nonconform import (
    ConformalDetector,
    Empirical,
    Split,
    forest_weight_estimator,
    logistic_weight_estimator,
)


def _build_weighted_detector(weight_estimator):
    return ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Empirical(),
        weight_estimator=weight_estimator,
        seed=13,
    )


@pytest.mark.parametrize(
    "estimator_factory",
    [logistic_weight_estimator, forest_weight_estimator],
    ids=["logistic", "forest"],
)
def test_weight_estimators_attach_weights(shifted_dataset, estimator_factory):
    """Both estimators should populate weights in the conformal result."""
    x_train, x_test, _ = shifted_dataset(n_train=180, n_test=60, n_features=4)
    estimator = estimator_factory(clip_quantile=0.1)
    detector = _build_weighted_detector(estimator)

    detector.fit(x_train)
    p_values = detector.predict(x_test)

    result = detector.last_result
    assert result is not None
    assert result.test_weights is not None
    assert result.calib_weights is not None
    assert np.all(result.test_weights > 0)
    assert np.all(result.calib_weights > 0)
    assert result.test_weights.shape[0] == len(p_values)
    # Ensure the estimator actually fitted and produced non-constant weights
    assert getattr(estimator, "_is_fitted", False) is True
    assert not np.allclose(result.test_weights, result.test_weights[0])


def test_weighted_vs_unweighted_predictions_differ(shifted_dataset):
    """Importance weighting should change p-values under covariate shift."""
    x_train, x_test, _ = shifted_dataset(n_train=200, n_test=80, n_features=5)

    weighted = _build_weighted_detector(logistic_weight_estimator())
    standard = ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Empirical(),
        seed=21,
    )

    weighted.fit(x_train)
    standard.fit(x_train)

    weighted_p = weighted.predict(x_test)
    standard_p = standard.predict(x_test)

    assert not np.allclose(weighted_p, standard_p)


def test_weight_clipping_bounds_propagate(shifted_dataset):
    """Verify estimator clipping bounds constrain stored weights."""
    estimator = logistic_weight_estimator(clip_quantile=0.2)
    detector = _build_weighted_detector(estimator)

    x_train, x_test, _ = shifted_dataset(n_train=160, n_test=64, n_features=3)
    detector.fit(x_train)
    detector.predict(x_test)

    assert estimator._clip_bounds is not None
    lower, upper = estimator._clip_bounds

    result = detector.last_result
    assert result is not None
    assert result.test_weights is not None
    assert result.calib_weights is not None
    assert np.all(result.test_weights <= upper + 1e-9)
    assert np.all(result.test_weights >= lower - 1e-9)
    assert np.all(result.calib_weights <= upper + 1e-9)
    assert np.all(result.calib_weights >= lower - 1e-9)
