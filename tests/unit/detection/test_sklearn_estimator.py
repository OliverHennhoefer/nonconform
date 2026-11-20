"""ConformalDetector as a scikit-learn estimator: checks, Pipelines, DataFrames."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import estimator_checks

from nonconform import ConformalDetector, CrossValidation, Split


@pytest.mark.parametrize(
    "check",
    [
        estimator_checks.check_get_params_invariance,
        estimator_checks.check_set_params,
        estimator_checks.check_no_attributes_set_in_init,
        estimator_checks.check_dont_overwrite_parameters,
        estimator_checks.check_n_features_in,
        estimator_checks.check_dataframe_column_names_consistency,
        estimator_checks.check_estimators_unfitted,
        estimator_checks.check_estimators_pickle,
    ],
    ids=lambda check: check.__name__,
)
def test_scikit_learn_estimator_contract(check):
    estimator = ConformalDetector(
        IsolationForest(random_state=0), Split(n_calib=0.3), seed=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check("ConformalDetector", estimator)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(2000, 3))
    test = np.vstack([rng.normal(size=(100, 3)), rng.normal(loc=6, size=(10, 3))])
    return reference, test


def test_strategy_defaults_to_split(data):
    reference, test = data
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    assert isinstance(detector.strategy, Split)
    assert detector.get_params(deep=False)["strategy"] is None
    assert detector.fit(reference).select(test, alpha=0.1)[-10:].all()


@pytest.mark.parametrize(
    ("strategy", "match"),
    [("split", "not the string 'split'"), (Split, r"pass Split\(\) instead")],
)
def test_strategy_mistakes_explain_the_fix(strategy, match):
    with pytest.raises(TypeError, match=match):
        ConformalDetector(IsolationForest(), strategy)


def test_predict_matches_select_and_leaves_results_untouched(data):
    reference, test = data
    detector = ConformalDetector(IsolationForest(random_state=0), alpha=0.1, seed=0)
    detector.fit(reference)
    assert detector.last_result is None
    labels = detector.predict(test)
    assert detector.last_result is None
    mask = detector.select(test, alpha=0.1)
    np.testing.assert_array_equal(labels, np.where(mask, -1, 1))
    assert labels.dtype.kind == "i"


def test_alpha_is_validated():
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        ConformalDetector(IsolationForest(), alpha=1.5)


def test_works_as_the_last_pipeline_step(data):
    reference, test = data
    pipeline = make_pipeline(
        StandardScaler(),
        ConformalDetector(IsolationForest(random_state=0), seed=0),
    ).fit(reference)
    labels = pipeline.predict(test)
    assert (labels[-10:] == -1).all()
    assert clone(pipeline).get_params()["conformaldetector__alpha"] == 0.05


@pytest.fixture
def frames():
    rng = np.random.default_rng(1)

    def frame(n, shift=0.0):
        return pd.DataFrame(
            {
                "amount": rng.lognormal(3 + shift, 1, n),
                "age": rng.normal(40, 10, n) + 10 * shift,
                "channel": rng.choice(["web", "app", "store"], n),
            }
        )

    test = pd.concat([frame(100), frame(10, shift=3.0)])
    test.index = [f"t{i}" for i in range(len(test))]
    return frame(2000), test


def test_dataframes_reach_the_wrapped_pipeline_unchanged(frames):
    reference, test = frames
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), ["amount", "age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel"]),
        ]
    )
    detector = ConformalDetector(
        make_pipeline(pre, IsolationForest(random_state=0)),
        score_polarity="higher_is_normal",
        seed=0,
    ).fit(reference)
    np.testing.assert_array_equal(
        detector.feature_names_in_, ["amount", "age", "channel"]
    )
    assert detector.n_features_in_ == 3
    selected = detector.select(test, alpha=0.1)
    assert list(selected.index) == list(test.index)
    assert selected.iloc[-10:].all()


class FrameRecorder:
    """Custom detector that records the input types it receives."""

    def __init__(self):
        self.seen = []

    def fit(self, X, y=None):
        self.seen.append(type(X))
        return self

    def decision_function(self, X):
        self.seen.append(type(X))
        return np.asarray(X, dtype=float)[:, 0]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class SklearnFrameRecorder(FrameRecorder, BaseEstimator):
    """The same recorder as a scikit-learn estimator."""


@pytest.mark.parametrize("strategy", [Split(0.3), CrossValidation(k=3)])
@pytest.mark.parametrize(
    ("recorder", "expected"),
    [(FrameRecorder, np.ndarray), (SklearnFrameRecorder, pd.DataFrame)],
    ids=["custom-gets-numpy", "sklearn-gets-dataframe"],
)
def test_only_sklearn_detectors_receive_dataframes(strategy, recorder, expected):
    frame = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)})
    detector = ConformalDetector(recorder(), strategy).fit(frame)
    detector.compute_p_values(frame)
    seen = {t for model in detector.detector_set for t in model.seen}
    assert seen == {expected}
    np.testing.assert_array_equal(detector.feature_names_in_, ["a", "b"])


def test_sparse_input_is_rejected(data):
    reference, _ = data
    with pytest.raises(TypeError, match="Sparse input is not supported"):
        ConformalDetector(IsolationForest()).fit(sparse.csr_matrix(reference))


def test_reordered_columns_are_rejected(frames):
    reference, test = frames
    numeric = ["amount", "age"]
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(reference[numeric])
    with pytest.raises(ValueError, match="same order"):
        detector.compute_p_values(test[["age", "amount"]])
    with pytest.raises(ValueError, match="unseen at fit time"):
        detector.compute_p_values(test[numeric].rename(columns={"age": "years"}))
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        detector.compute_p_values(test[numeric].to_numpy())


def test_series_row_keeps_feature_names(frames):
    reference, test = frames
    numeric = ["amount", "age"]
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(reference[numeric])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = detector.compute_p_value(test[numeric].iloc[0])
    assert 0 < value <= 1


def test_feature_names_survive_pickling(frames):
    reference, _ = frames
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(reference[["amount", "age"]])
    restored = pickle.loads(pickle.dumps(detector))
    np.testing.assert_array_equal(restored.feature_names_in_, ["amount", "age"])


def test_scoring_function_with_detached_calibration(data):
    reference, test = data

    def distance(x):
        return np.linalg.norm(np.asarray(x), axis=1)

    detector = ConformalDetector(distance).calibrate(reference)
    assert detector.n_features_in_ == 3
    assert detector.select(test, alpha=0.1)[-10:].all()


def test_result_to_frame_keeps_the_index(frames):
    reference, test = frames
    numeric = ["amount", "age"]
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(reference[numeric])
    p_values = detector.compute_p_values(test[numeric])
    table = detector.last_result.to_frame()
    assert list(table.columns) == ["score", "p_value"]
    assert list(table.index) == list(test.index)
    pd.testing.assert_series_equal(table["p_value"], p_values)
    assert table.index.equals(detector.last_result.copy().to_frame().index)

    detector.score_samples(test[numeric].to_numpy())
    assert list(detector.last_result.to_frame().columns) == ["score"]


def test_pipeline_of_conformal_detector_is_cloneable():
    pipeline = Pipeline([("conf", ConformalDetector(IsolationForest()))])
    assert clone(pipeline)[0].get_params()["strategy"] is None


def test_clone_shares_no_component_state():
    from nonconform import Empirical, logistic_weight_estimator

    estimation = Empirical(tie_break="randomized")
    weights = logistic_weight_estimator()
    detector = ConformalDetector(
        IsolationForest(), estimation=estimation, weight_estimator=weights, seed=1
    )
    cloned = clone(detector)
    assert cloned.estimation is not detector.estimation
    assert cloned.weight_estimator is not detector.weight_estimator
    assert cloned.get_params(deep=False)["detector"] is not estimation


@pytest.mark.parametrize(
    ("strategy", "match"),
    [(dict, "must be a strategy instance"), (object(), "nonconform strategy")],
)
def test_other_strategy_values_are_rejected(strategy, match):
    with pytest.raises(TypeError, match=match):
        ConformalDetector(IsolationForest(), strategy)


def test_fitted_attributes_are_absent_before_fit():
    detector = ConformalDetector(IsolationForest())
    assert not hasattr(detector, "n_features_in_")
    assert not hasattr(detector, "feature_names_in_")
    assert not detector.__sklearn_is_fitted__()


def test_series_reference_and_unnamed_fit_warns_on_named_test(frames):
    reference, test = frames
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(reference["amount"])
    np.testing.assert_array_equal(detector.feature_names_in_, ["amount"])
    assert detector.compute_p_values(test["amount"]).shape == (len(test),)

    detector.fit(reference[["amount", "age"]].to_numpy())
    with pytest.warns(UserWarning, match="fitted without feature names"):
        detector.compute_p_values(test[["amount", "age"]])


def test_scoring_adapter_protocol_details():
    from nonconform.adapters import adapt

    def score(x):
        return np.asarray(x)[:, 0]

    adapted = adapt(score)
    assert adapted.set_params() is adapted
    assert "_ScoringAdapter" in repr(adapted)
    assert adapted.__name__ == "score"
    with pytest.raises(AttributeError):
        adapted.missing_attribute


def test_to_frame_requires_scores_and_includes_weights(data):
    from nonconform import logistic_weight_estimator
    from nonconform.structures import ConformalResult

    with pytest.raises(ValueError, match="no test scores"):
        ConformalResult().to_frame()

    reference, test = data
    detector = ConformalDetector(
        IsolationForest(random_state=0),
        weight_estimator=logistic_weight_estimator(),
        seed=0,
    ).fit(reference)
    detector.compute_p_values(test)
    assert list(detector.last_result.to_frame().columns) == [
        "score",
        "p_value",
        "weight",
    ]


def test_split_strategy_accepts_a_series():
    seen = []

    class Recorder:
        def fit(self, X, y=None):
            seen.append(np.shape(X))
            return self

        def decision_function(self, X):
            return np.zeros(len(X))

    Split(0.5).fit_calibrate(pd.Series(np.arange(10.0)), Recorder())
    assert seen == [(5,)]
