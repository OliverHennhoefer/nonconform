"""ConformalDetector as a scikit-learn estimator: checks, Pipelines, DataFrames."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from nonconform import ConformalDetector, CrossValidation, Split

# Checks that assume a per-row threshold. predict() is a batch-level FDR
# decision: it may select nothing on clean training data, and a row's label
# depends on the rest of its batch.
BATCH_LEVEL_CHECKS = {
    "check_outliers_train",
    "check_outliers_fit_predict",
    "check_methods_subset_invariance",
}
# Parameters are validated when set, so rejected updates keep the fitted state.
EAGER_VALIDATION_CHECKS = {"check_do_not_raise_errors_in_init_or_set_params"}


@parametrize_with_checks(
    [ConformalDetector(IsolationForest(random_state=0), Split(n_calib=0.3), seed=0)]
)
def test_sklearn_compatible_estimator(estimator, check):
    name = getattr(check, "func", check).__name__
    if name in BATCH_LEVEL_CHECKS:
        pytest.xfail("predict() is a batch-level FDR decision by design")
    if name in EAGER_VALIDATION_CHECKS:
        pytest.xfail("parameters are validated eagerly by design")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check(estimator)


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
    np.testing.assert_array_equal(detector.decision_function(test) < 0, mask)


def test_alpha_is_validated():
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        ConformalDetector(IsolationForest(), alpha=1.5)


def test_decision_function_rejects_e_value_and_weighted_modes(data):
    from nonconform import DerandomizedSplits

    reference, test = data
    detector = ConformalDetector(IsolationForest(), DerandomizedSplits(2, 0.5))
    detector.fit(reference[:200])
    with pytest.raises(ValueError, match="adjusted p-values"):
        detector.decision_function(test)
    assert detector.predict(test).shape == (len(test),)


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


@pytest.mark.parametrize("strategy", [Split(0.3), CrossValidation(k=3)])
def test_strategies_pass_dataframes_through(strategy):
    seen = []

    class Recorder:
        def fit(self, X, y=None):
            seen.append(type(X))
            return self

        def decision_function(self, X):
            seen.append(type(X))
            return X["a"].to_numpy(dtype=float)

    frame = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)})
    ConformalDetector(Recorder(), strategy).fit(frame).compute_p_values(frame)
    assert set(seen) == {pd.DataFrame}


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


def test_sparse_input_is_supported(data):
    reference, test = data
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    dense = detector.fit(reference).compute_p_values(test)
    sparse_values = detector.fit(sparse.csr_matrix(reference)).compute_p_values(
        sparse.csr_matrix(test)
    )
    np.testing.assert_allclose(dense, sparse_values)


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


def test_sparse_batches_are_signed_without_densifying(data, monkeypatch):
    reference, test = data
    detector = ConformalDetector(IsolationForest(random_state=0), seed=0)
    detector.fit(sparse.csr_matrix(reference))

    def fail(*args, **kwargs):
        raise AssertionError("densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail)
    assert detector.score_samples(sparse.csr_matrix(test)).shape == (len(test),)


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


def test_sparse_input_in_weighted_mode(data):
    from nonconform import logistic_weight_estimator

    reference, test = data
    detector = ConformalDetector(
        IsolationForest(random_state=0),
        weight_estimator=logistic_weight_estimator(),
        seed=0,
    ).fit(sparse.csr_matrix(reference))
    test_matrix = sparse.csr_matrix(test)
    detector.prepare_weights_for(test_matrix)
    reused = detector.compute_p_values(test_matrix, refit_weights=False)
    assert reused.shape == (len(test),)


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
