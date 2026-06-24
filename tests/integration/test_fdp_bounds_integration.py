"""Integration tests for post-hoc conformal FDP bounds."""

from __future__ import annotations

import numpy as np
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.fdr import conformal_fdp_upper_bound_from_result


def test_fdp_bounds_do_not_change_existing_unweighted_selection(simple_dataset):
    """FDP bounds are an opt-in report over existing conformal p-values."""
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=50, n_features=4)
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.25),
        seed=14,
    )

    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)
    result = detector.last_result
    assert result is not None

    bh_mask = false_discovery_control(p_values, method="bh") <= 0.2
    bounds = conformal_fdp_upper_bound_from_result(
        result,
        confidence=0.8,
        n_resamples=25,
        seed=14,
        thresholds=np.array([0.01, 0.05, 0.1, 0.2]),
    )
    select_mask = detector.select(x_test, alpha=0.2)

    np.testing.assert_array_equal(select_mask, bh_mask)
    np.testing.assert_array_equal(bounds.select(0.1), p_values <= 0.1)
    assert np.all((0.0 <= bounds.fdp_upper_bounds) & (bounds.fdp_upper_bounds <= 1.0))
