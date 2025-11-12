"""Integration tests for multi-detector aggregation logic."""

from __future__ import annotations

import numpy as np
import pytest

from nonconform.detection import ConformalDetector
from nonconform.strategy import CrossValidation
from nonconform.utils.func.enums import Aggregation
from nonconform.utils.stat.aggregation import aggregate
from pyod.models.iforest import IForest


def _build_detector(aggregation: Aggregation):
    return ConformalDetector(
        detector=IForest(n_estimators=20, max_samples=0.8, random_state=0),
        strategy=CrossValidation(k=3, plus=True),
        aggregation=aggregation,
        seed=23,
    )


@pytest.mark.parametrize("aggregation", list(Aggregation))
def test_aggregation_matches_manual_scores(simple_dataset, aggregation):
    """Aggregated raw scores should match manual aggregation for ensembles."""
    x_train, x_test, _ = simple_dataset(n_train=54, n_test=18, n_features=4)
    detector = _build_detector(aggregation)
    detector.fit(x_train)

    raw_scores = detector.predict(x_test, raw=True)
    detectors = detector.detector_set
    assert len(detectors) >= 3  # ensemble created by CrossValidation
    stacked = np.vstack([model.decision_function(x_test) for model in detectors])

    manual_scores = aggregate(aggregation, stacked)
    np.testing.assert_allclose(raw_scores, manual_scores)


def test_aggregation_choice_changes_pvalues(simple_dataset):
    """Different aggregation methods should impact downstream p-values."""
    x_train, x_test, _ = simple_dataset(n_train=60, n_test=20, n_features=4)
    mean_detector = _build_detector(Aggregation.MEAN)
    max_detector = _build_detector(Aggregation.MAXIMUM)

    mean_detector.fit(x_train)
    max_detector.fit(x_train)

    mean_p = mean_detector.predict(x_test)
    max_p = max_detector.predict(x_test)

    assert not np.allclose(mean_p, max_p)
