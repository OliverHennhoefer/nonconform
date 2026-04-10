import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from nonconform import (
    IntegrativeConformalDetector,
    IntegrativeModel,
    IntegrativeSplit,
)
from nonconform.fdr import integrative_false_discovery_control
from nonconform.structures import ConformalResult
from tests.unit.integrative.conftest import MeanDistanceDetector


def test_integrative_fdr_requires_result_bundle():
    with pytest.raises(ValueError, match="result must be a ConformalResult, got None"):
        integrative_false_discovery_control(None, alpha=0.1)  # type: ignore[arg-type]


def test_integrative_fdr_requires_integrative_metadata():
    result = ConformalResult(p_values=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="must contain integrative split caches"):
        integrative_false_discovery_control(result, alpha=0.1)


def test_integrative_fdr_returns_boolean_mask(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=12)
    detector = IntegrativeConformalDetector(
        models=[
            IntegrativeModel.one_class(
                reference="inlier",
                estimator=MeanDistanceDetector(),
                score_polarity="higher_is_anomalous",
            ),
            IntegrativeModel.one_class(
                reference="outlier",
                estimator=MeanDistanceDetector(),
                score_polarity="higher_is_anomalous",
            ),
            IntegrativeModel.binary(
                estimator=LogisticRegression(solver="liblinear"),
                inlier_label=0,
            ),
        ],
        strategy=IntegrativeSplit(n_calib=0.2),
        seed=5,
    )
    detector.fit(x_in, x_out)
    _ = detector.compute_p_values(x_test)

    mask = integrative_false_discovery_control(detector.last_result, alpha=0.2, seed=5)

    assert mask.dtype == bool
    assert mask.shape == (len(x_test),)
