"""Integration tests for weighted FDR control utilities."""

from __future__ import annotations

import numpy as np
import pytest

from nonconform.detection import ConformalDetector
from nonconform.detection.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.strategy.estimation.probabilistic import Probabilistic
from nonconform.utils.func.enums import Kernel, Pruning
from nonconform.utils.stat.weighted_fdr import (
    weighted_bh,
    weighted_false_discovery_control,
)
from pyod.models.iforest import IForest


def _fit_weighted_detector(x_train):
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0),
        weight_estimator=LogisticWeightEstimator(seed=4),
        seed=4,
    )
    detector.fit(x_train)
    return detector


@pytest.mark.parametrize("pruning", list(Pruning))
def test_pruning_modes_control_false_discoveries(simple_dataset, pruning):
    """weighted_false_discovery_control should run for all pruning modes."""
    x_train, x_test, y_test = simple_dataset(n_train=120, n_test=60, n_features=5)
    detector = _fit_weighted_detector(x_train)
    detector.predict(x_test)
    result = detector.last_result
    assert result is not None

    selections = weighted_false_discovery_control(
        result=result,
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


def test_weighted_bh_respects_pvalue_ordering(simple_dataset):
    """Selected discoveries must correspond to the smallest p-values."""
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=50, n_features=4)
    detector = _fit_weighted_detector(x_train)
    detector.predict(x_test)
    result = detector.last_result
    assert result is not None and result.p_values is not None

    mask = weighted_bh(result, alpha=0.2)
    assert mask.shape == (len(x_test),)

    if np.any(mask):
        max_sel = np.max(result.p_values[mask])
        assert np.all(result.p_values[~mask] >= max_sel - 1e-12)
