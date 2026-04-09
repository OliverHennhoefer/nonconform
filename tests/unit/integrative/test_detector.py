import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nonconform import (
    IntegrativeConformalDetector,
    IntegrativeModel,
    IntegrativeSplit,
    TransductiveCVPlus,
)
from tests.unit.integrative.conftest import MeanDistanceDetector


def _split_detector() -> IntegrativeConformalDetector:
    models = [
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
    return IntegrativeConformalDetector(
        models=models,
        strategy=IntegrativeSplit(n_calib=0.25),
        seed=19,
    )


def _tcv_detector() -> IntegrativeConformalDetector:
    models = [
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
    return IntegrativeConformalDetector(
        models=models,
        strategy=TransductiveCVPlus(k_in=4, k_out=4),
        seed=23,
    )


def test_split_compute_p_values_populates_integrative_metadata(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset()
    detector = _split_detector().fit(x_in, x_out)

    p_values = detector.compute_p_values(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    result = detector.last_result
    assert result is not None
    assert result.test_scores is not None
    assert result.calib_scores is not None
    assert result.calib_scores.shape[0] == len(x_test)
    assert result.metadata["integrative"]["strategy"] == "split"


def test_split_score_samples_returns_ratio_scores(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset()
    detector = _split_detector().fit(x_in, x_out)

    scores = detector.score_samples(x_test)
    result = detector.last_result

    assert result is not None
    np.testing.assert_allclose(scores, result.test_scores)


def test_split_dataframe_input_preserves_index(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset()
    detector = _split_detector().fit(x_in, x_out)

    index = pd.RangeIndex(start=100, stop=100 + len(x_test))
    p_values = detector.compute_p_values(pd.DataFrame(x_test, index=index))
    selected = detector.select(pd.DataFrame(x_test, index=index), alpha=0.2)

    assert isinstance(p_values, pd.Series)
    assert isinstance(selected, pd.Series)
    assert p_values.index.equals(index)
    assert selected.index.equals(index)


def test_split_binary_model_sign_tuning_is_recorded(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset()
    detector = IntegrativeConformalDetector(
        models=IntegrativeModel.binary(
            estimator=LogisticRegression(solver="liblinear"),
            inlier_label=0,
            name="logistic",
        ),
        strategy=IntegrativeSplit(n_calib=0.25),
        seed=11,
    ).fit(x_in, x_out)

    _ = detector.compute_p_values(x_test)
    metadata = detector.last_result.metadata["integrative"]

    assert np.all(metadata["selected_u0_signs"] == 1)
    assert np.all(metadata["selected_u1_signs"] == -1)


def test_tcv_plus_compute_p_values_is_reproducible(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=10)
    first = _tcv_detector().fit(x_in, x_out)
    second = _tcv_detector().fit(x_in, x_out)

    p_first = first.compute_p_values(x_test)
    p_second = second.compute_p_values(x_test)

    np.testing.assert_allclose(p_first, p_second)


def test_tcv_plus_select_raises_not_implemented(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=10)
    detector = _tcv_detector().fit(x_in, x_out)

    with pytest.raises(
        NotImplementedError, match="currently implemented only for IntegrativeSplit"
    ):
        detector.select(x_test, alpha=0.1)
