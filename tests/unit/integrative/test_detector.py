import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
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


def test_tcv_plus_compute_p_values_are_batch_order_invariant(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=10, seed=31)
    detector = _tcv_detector().fit(x_in, x_out)

    original = detector.compute_p_values(x_test)
    permutation = np.array([3, 1, 5, 0, 9, 2, 8, 4, 7, 6])
    permuted = detector.compute_p_values(x_test[permutation])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))

    np.testing.assert_allclose(original, np.asarray(permuted)[inverse])


def test_tcv_plus_select_raises_not_implemented(labeled_ood_dataset):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=10)
    detector = _tcv_detector().fit(x_in, x_out)

    with pytest.raises(
        NotImplementedError, match="currently implemented only for IntegrativeSplit"
    ):
        detector.select(x_test, alpha=0.1)


def test_split_binary_model_supports_string_inlier_label_named_outlier(
    labeled_ood_dataset,
):
    x_in, x_out, x_test, _ = labeled_ood_dataset(n_test=8, seed=13)
    detector = IntegrativeConformalDetector(
        models=IntegrativeModel.binary(
            estimator=LogisticRegression(solver="liblinear"),
            inlier_label="__outlier__",
            name="logistic",
        ),
        strategy=IntegrativeSplit(n_calib=0.25),
        seed=13,
    ).fit(x_in, x_out)

    p_values = detector.compute_p_values(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))


def test_integrative_detector_supports_sklearn_clone():
    detector = _split_detector()

    cloned = clone(detector)

    assert isinstance(cloned, IntegrativeConformalDetector)
    assert cloned is not detector
    assert cloned.seed == detector.seed
    assert cloned.strategy.n_calib == detector.strategy.n_calib


def test_integrative_detector_set_params_updates_nested_strategy():
    detector = _split_detector()

    result = detector.set_params(strategy__n_calib=0.3, seed=29)

    assert result is detector
    assert detector.strategy.n_calib == 0.3
    assert detector.seed == 29


def test_integrative_detector_set_params_accepts_model_index_replacement():
    detector = _split_detector()
    replacement = IntegrativeModel.binary(
        estimator=LogisticRegression(solver="liblinear"),
        inlier_label=1,
        score_source="decision_function",
        name="replacement",
    )

    result = detector.set_params(models__2=replacement)

    assert result is detector
    assert detector.models[2] is not replacement
    assert detector.models[2].kind == "binary"
    assert detector.models[2].inlier_label == 1
    assert detector.models[2].score_source == "decision_function"
    assert detector.models[2].name == "replacement"
    assert detector.models[2].estimator is not replacement.estimator
    assert detector.models[2].estimator.get_params()["solver"] == "liblinear"


def test_integrative_detector_set_params_round_trips_deep_params():
    detector = _split_detector()

    result = detector.set_params(**detector.get_params(deep=True))

    assert result is detector
    assert detector.seed == 19
    assert detector.strategy.n_calib == 0.25
    assert len(detector.models) == 3
    assert detector.models[0].reference == "inlier"
    assert detector.models[1].reference == "outlier"
    assert detector.models[2].kind == "binary"
    assert detector.models[2].inlier_label == 0
    assert detector.models[2].estimator.get_params()["solver"] == "liblinear"
