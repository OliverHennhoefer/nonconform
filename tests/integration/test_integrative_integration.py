"""Integration tests for integrative conformal workflows."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from nonconform import (
    IntegrativeConformalDetector,
    IntegrativeModel,
    IntegrativeSplit,
    TransductiveCVPlus,
)
from nonconform.metrics import false_discovery_rate, statistical_power
from tests.unit.integrative.conftest import MeanDistanceDetector


def _integrative_models() -> list[IntegrativeModel]:
    return [
        IntegrativeModel.one_class(
            reference="inlier",
            estimator=MeanDistanceDetector(),
            score_polarity="higher_is_anomalous",
            name="inlier_distance",
        ),
        IntegrativeModel.one_class(
            reference="outlier",
            estimator=MeanDistanceDetector(),
            score_polarity="higher_is_anomalous",
            name="outlier_distance",
        ),
        IntegrativeModel.binary(
            estimator=LogisticRegression(solver="liblinear"),
            inlier_label=0,
            name="logistic",
        ),
    ]


def test_integrative_split_end_to_end(labeled_ood_dataset) -> None:
    x_in, x_out, x_test, y_test = labeled_ood_dataset(n_test=20, seed=21)
    detector = IntegrativeConformalDetector(
        models=_integrative_models(),
        strategy=IntegrativeSplit(n_calib=0.25),
        seed=21,
    )
    detector.fit(x_in, x_out)

    p_values = detector.compute_p_values(x_test)
    selected = detector.select(x_test, alpha=0.2)

    assert p_values.shape == (len(x_test),)
    assert selected.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    # The method should surface meaningful signal on the separable synthetic task.
    assert statistical_power(y_test, selected) > 0.2
    assert false_discovery_rate(y_test, selected) <= 0.5


def test_integrative_tcv_plus_computes_valid_p_values(labeled_ood_dataset) -> None:
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=8, seed=31)
    detector = IntegrativeConformalDetector(
        models=_integrative_models(),
        strategy=TransductiveCVPlus(k_in=4, k_out=4),
        seed=31,
    )
    detector.fit(x_in, x_out)

    p_values = detector.compute_p_values(x_test)
    result = detector.last_result

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    assert result is not None
    assert result.metadata["integrative"]["strategy"] == "tcv_plus"
