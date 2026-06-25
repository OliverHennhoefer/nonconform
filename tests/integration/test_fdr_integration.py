"""Integration tests for weighted FDR control utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import (
    ConformalDetector,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform.enums import Kernel, Pruning
from nonconform.fdr import conformal_e_value_selection


def _fit_weighted_detector(x_train):
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0),
        weight_estimator=logistic_weight_estimator(),
        seed=4,
    )
    detector.fit(x_train)
    return detector


@pytest.mark.parametrize("pruning", list(Pruning))
def test_pruning_modes_control_false_discoveries(simple_dataset, pruning):
    """select() should support weighted pruning modes end-to-end."""
    x_train, x_test, y_test = simple_dataset(n_train=120, n_test=60, n_features=5)
    detector = _fit_weighted_detector(x_train)

    selections = detector.select(
        x_test,
        alpha=0.25,
        pruning=pruning,
        seed=0,
    )
    assert selections.dtype == bool
    assert selections.shape == (len(x_test),)

    discoveries = int(np.count_nonzero(selections))
    if discoveries > 0:
        false_pos = int(np.count_nonzero(selections & (y_test == 0)))
        observed_fdr = false_pos / discoveries
        assert observed_fdr <= 0.35  # empirical control with generous slack


def test_standard_bh_on_weighted_pvalues_respects_ordering(simple_dataset):
    """Selected discoveries must correspond to the smallest p-values."""
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=50, n_features=4)
    detector = _fit_weighted_detector(x_train)
    detector.compute_p_values(x_test)
    result = detector.last_result
    assert result is not None and result.p_values is not None

    mask = false_discovery_control(result.p_values, method="bh") <= 0.2
    assert mask.shape == (len(x_test),)

    if np.any(mask):
        max_sel = np.max(result.p_values[mask])
        assert np.all(result.p_values[~mask] >= max_sel - 1e-12)


def test_repeated_split_scores_feed_conformal_e_value_selection(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=120, n_test=40, n_features=4)

    def collect_result():
        test_scores = []
        calib_scores = []
        for seed in range(4):
            detector = ConformalDetector(
                detector=IsolationForest(n_estimators=20, random_state=seed),
                strategy=Split(n_calib=0.3),
                score_polarity="auto",
                seed=seed,
            )
            detector.fit(x_train)
            detector.score_samples(x_test)
            result = detector.last_result
            assert result is not None
            test_scores.append(result.test_scores)
            calib_scores.append(result.calib_scores)

        return conformal_e_value_selection(
            np.vstack(test_scores),
            np.vstack(calib_scores),
            alpha=0.2,
        )

    first = collect_result()
    second = collect_result()

    assert first.selected.dtype == bool
    assert first.selected.shape == (len(x_test),)
    assert first.e_values.shape == (len(x_test),)
    assert first.n_repetitions == 4
    np.testing.assert_array_equal(first.selected, second.selected)
    np.testing.assert_allclose(first.e_values, second.e_values)
