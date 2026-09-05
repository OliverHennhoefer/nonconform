from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from nonconform import (
    ConformalDetector,
    DerandomizedSplits,
    Empirical,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform.fdr import conformal_e_values, select_conformal_e_values
from nonconform.scoring import ConditionalEmpirical
from nonconform.weighting import IdentityWeightEstimator


class RecordingDetector:
    def __init__(self, random_state=None, polarity=1):
        self.random_state = random_state
        self.polarity = polarity

    def fit(self, X, y=None):
        self.training = X.copy()
        self.center = X.mean(axis=0)
        return self

    def decision_function(self, X):
        # Discrete, model-dependent scores deliberately exercise automatic ties.
        return self.polarity * np.round(np.linalg.norm(X - self.center, axis=1))

    def get_params(self, deep=True):
        return {"random_state": self.random_state, "polarity": self.polarity}

    def set_params(self, **params):
        self.__dict__.update(params)
        return self


@pytest.fixture
def batches():
    rng = np.random.default_rng(18)
    return rng.normal(size=(120, 2)), np.r_[rng.normal(size=(25, 2)), [[8, 9]]]


def make_detector(**kwargs):
    return ConformalDetector(
        detector=RecordingDetector(),
        strategy=DerandomizedSplits(n_repetitions=3, n_calib=30),
        seed=42,
        **kwargs,
    )


@pytest.mark.parametrize("n_calib", [30, 0.25])
def test_replicas_preserve_disjoint_calibration_pairing(batches, n_calib):
    reference, _ = batches
    base = RecordingDetector()
    strategy = DerandomizedSplits(3, n_calib)
    models, calibration = strategy.fit_calibrate(reference, base, seed=42)
    streams = np.random.SeedSequence(42).spawn(2)[0].spawn(3)

    assert calibration.shape == (3, 30)
    assert not hasattr(base, "training")
    assert len({id(model) for model in models}) == 3
    for i, (model, stream) in enumerate(zip(models, streams, strict=True)):
        seed = int(stream.generate_state(1)[0])
        train, calib = train_test_split(
            np.arange(len(reference)), test_size=n_calib, random_state=seed
        )
        assert set(train).isdisjoint(calib)
        np.testing.assert_array_equal(model.training, reference[train])
        np.testing.assert_array_equal(
            calibration[i], model.decision_function(reference[calib])
        )
    original = models[1].training.copy()
    models[0].training[:] = 0
    np.testing.assert_array_equal(models[1].training, original)


@pytest.mark.parametrize("n_repetitions", [1, 4])
@pytest.mark.parametrize("alpha_bh", [None, 0.3])
def test_selection_matches_manual_split_workflow(batches, n_repetitions, alpha_bh):
    reference, test = batches
    automatic = ConformalDetector(
        RecordingDetector(),
        DerandomizedSplits(n_repetitions, 30, alpha_bh=alpha_bh, tie_seed=2026),
        seed=42,
    ).fit(reference)
    mask = automatic.select(test, alpha=0.2)
    results = []
    for stream in np.random.SeedSequence(42).spawn(2)[0].spawn(n_repetitions):
        seed = int(stream.generate_state(1)[0])
        manual = ConformalDetector(RecordingDetector(), Split(30), seed=seed).fit(
            reference
        )
        manual.score_samples(test)
        results.append(manual.last_result)
    expected = select_conformal_e_values(
        results, alpha=0.2, alpha_bh=alpha_bh, tie_seed=2026
    )
    actual = automatic.last_selection_result
    np.testing.assert_array_equal(actual.e_values, expected.e_values)
    np.testing.assert_array_equal(mask, expected.selected)
    assert actual.e_threshold == expected.e_threshold
    assert actual.alpha_bh == expected.alpha_bh
    assert actual.n_repetitions == n_repetitions
    assert actual.n_calibration == 30
    assert automatic.last_result is None


@pytest.mark.parametrize("seed", [42, None])
def test_automatic_ties_repeat_within_fit(batches, seed):
    reference, test = batches
    detector = ConformalDetector(
        RecordingDetector(), DerandomizedSplits(3, 30), seed=seed
    ).fit(reference)
    first = detector.select(test, alpha=0.2)
    result = detector.last_selection_result
    assert isinstance(result.tie_seed, int)
    np.testing.assert_array_equal(first, detector.select(test, alpha=0.2))
    np.testing.assert_array_equal(
        result.e_values, detector.last_selection_result.e_values
    )
    if seed is not None:
        replay = clone(detector).fit(reference)
        np.testing.assert_array_equal(first, replay.select(test, alpha=0.2))
        np.testing.assert_array_equal(
            result.e_values, replay.last_selection_result.e_values
        )
        assert result.tie_seed == replay.last_selection_result.tie_seed


def test_tie_override_does_not_change_fitting_and_is_used_for_evidence(batches):
    reference, test = batches
    fitted = []
    for tie_seed in [0, 1]:
        detector = ConformalDetector(
            RecordingDetector(), DerandomizedSplits(3, 30, tie_seed=tie_seed), seed=42
        ).fit(reference)
        detector.select(test, alpha=0.2)
        result = detector.last_selection_result
        rows = np.vstack(
            [model.decision_function(test) for model in detector.detector_set]
        )
        np.testing.assert_array_equal(
            result.e_values,
            conformal_e_values(
                rows, detector.calibration_set, alpha_bh=0.02, tie_seed=tie_seed
            ),
        )
        assert result.tie_seed == tie_seed
        fitted.append(detector)
    np.testing.assert_array_equal(fitted[0].calibration_set, fitted[1].calibration_set)
    for first, second in zip(
        fitted[0].detector_set, fitted[1].detector_set, strict=True
    ):
        np.testing.assert_array_equal(first.training, second.training)


def test_raw_aggregation_does_not_affect_e_values_or_fabricate_calibration(batches):
    reference, test = batches
    evidence = []
    for method in ["mean", "median", "minimum", "maximum"]:
        detector = make_detector(aggregation=method).fit(reference)
        detector.select(test)
        evidence.append(detector.last_selection_result.e_values)
        scores = detector.score_samples(test)
        assert scores.shape == (len(test),)
        assert detector.last_result.calib_scores is None
        assert detector.last_selection_result is None
    for values in evidence[1:]:
        np.testing.assert_array_equal(values, evidence[0])


@pytest.mark.parametrize("pandas_type", [pd.DataFrame, pd.Series])
def test_pandas_mask_and_scores_preserve_index(batches, pandas_type):
    reference, test = batches
    reference, test = reference[:, :1], test[:, :1]
    detector = make_detector().fit(reference)
    index = pd.Index([f"row-{i // 2}" for i in range(len(test))], name="sample")
    batch = pandas_type(
        test if pandas_type is pd.DataFrame else test[:, 0], index=index
    )
    expected = detector.select(test)
    selected = detector.select(batch)
    assert selected.name == "selected"
    assert selected.dtype == bool
    assert selected.index.equals(index)
    np.testing.assert_array_equal(selected, expected)
    scores = detector.score_samples(batch)
    assert scores.name == "score"
    assert scores.index.equals(index)


def test_result_snapshots_and_returned_mask_are_isolated(batches):
    reference, test = batches
    detector = make_detector().fit(reference)
    mask = detector.select(test)
    original = detector.last_selection_result
    snapshot = detector.last_selection_result
    assert not snapshot.e_values.flags.writeable
    snapshot.e_values.flags.writeable = True
    snapshot.e_values[:] = -1
    snapshot.selected.flags.writeable = True
    snapshot.selected[:] = ~snapshot.selected
    mask[:] = ~mask
    np.testing.assert_array_equal(
        detector.last_selection_result.e_values, original.e_values
    )
    np.testing.assert_array_equal(
        detector.last_selection_result.selected, original.selected
    )


def test_clone_parameters_and_refit_clear_state(batches):
    reference, test = batches
    detector = make_detector().fit(reference)
    detector.select(test)
    cloned = clone(detector)
    assert "n_calibration=30" in repr(detector)
    assert not cloned.is_fitted
    assert cloned.last_selection_result is None
    assert cloned.get_params()["strategy__n_repetitions"] == 3
    copied = deepcopy(detector)
    np.testing.assert_array_equal(copied.select(test), detector.select(test))
    detector.fit(reference)
    assert detector.last_selection_result is None
    detector.select(test)
    detector.set_params(strategy__n_repetitions=2)
    assert not detector.is_fitted
    assert detector.last_selection_result is None
    assert detector.last_result is None
    detector.fit(reference).select(test)
    assert detector.last_selection_result.n_repetitions == 2
    assert clone(detector.strategy).get_params() == detector.strategy.get_params()


@pytest.mark.parametrize("method", ["compute_p_value", "compute_p_values", "calibrate"])
def test_unsupported_operations_clear_selection(batches, method):
    reference, test = batches
    detector = make_detector().fit(reference)
    detector.select(test)
    with pytest.raises(ValueError, match=r"select|fit"):
        getattr(detector, method)(test[0] if method == "compute_p_value" else test)
    assert detector.last_selection_result is None


def test_failed_selection_and_refit_do_not_leave_stale_state(batches):
    reference, test = batches
    detector = make_detector().fit(reference)
    for invalid in [np.empty((0, 2)), np.zeros((3, 5)), np.full((3, 2), np.nan)]:
        detector.select(test)
        with pytest.raises(ValueError):
            detector.select(invalid)
        assert detector.last_selection_result is None
    detector.select(test)
    with pytest.raises(ValueError):
        detector.select(test, alpha=0)
    assert detector.last_selection_result is None
    detector.select(test)
    with pytest.raises(ValueError):
        detector.fit(reference[:3])
    assert not detector.is_fitted
    assert detector.last_selection_result is None
    with pytest.raises(NotFittedError):
        detector.select(test)


@pytest.mark.parametrize("estimation", [ConditionalEmpirical(), Probabilistic()])
def test_rejects_incompatible_estimators(estimation):
    with pytest.raises(ValueError, match="estimation is unused"):
        make_detector(estimation=estimation)


def test_weighting_restrictions_and_identity_compatibility(batches):
    with pytest.raises(ValueError, match="weighting"):
        make_detector(weight_estimator=logistic_weight_estimator())
    reference, test = batches
    detector = make_detector(
        estimation=Empirical(), weight_estimator=IdentityWeightEstimator()
    )
    detector.fit(reference).select(test)
    assert detector.last_selection_result is not None
    with pytest.raises(ValueError, match="weighted"):
        DerandomizedSplits().fit_calibrate(
            reference, RecordingDetector(), weighted=True
        )


@pytest.mark.parametrize(
    "value,error",
    [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_repetition_validation(value, error):
    with pytest.raises(error):
        DerandomizedSplits(n_repetitions=value)


@pytest.mark.parametrize(
    "kwargs", [{"alpha_bh": 0}, {"alpha_bh": 1}, {"alpha_bh": np.nan}, {"tie_seed": -1}]
)
def test_invalid_selection_configuration(kwargs):
    with pytest.raises(ValueError):
        DerandomizedSplits(**kwargs)


@pytest.mark.parametrize("value", [0, -1, 1.0, 120, 121])
def test_calibration_size_validation(batches, value):
    with pytest.raises(ValueError):
        DerandomizedSplits(n_calib=value).fit_calibrate(batches[0], RecordingDetector())


def test_no_discoveries_and_default_configuration():
    assert DerandomizedSplits().get_params() == {
        "n_repetitions": 5,
        "n_calib": 0.1,
        "alpha_bh": None,
        "tie_seed": None,
    }
    detector = make_detector().fit(np.zeros((120, 2)))
    mask = detector.select(np.zeros((5, 2)), alpha=0.001)
    assert not mask.any()
    assert detector.last_selection_result.e_threshold == float("inf")


def test_existing_workflow_has_no_e_value_diagnostics(batches):
    reference, test = batches
    detector = ConformalDetector(RecordingDetector(), Split(30), seed=42).fit(reference)
    detector.select(test)
    assert detector.last_result.p_values is not None
    assert detector.last_selection_result is None


def test_explicit_score_polarities_produce_identical_evidence(batches):
    reference, test = batches
    evidence = []
    for sign, polarity in [(1, "higher_is_anomalous"), (-1, "higher_is_normal")]:
        detector = ConformalDetector(
            RecordingDetector(polarity=sign),
            DerandomizedSplits(3, 30),
            seed=42,
            score_polarity=polarity,
        ).fit(reference)
        detector.select(test)
        evidence.append(detector.last_selection_result.e_values)
    np.testing.assert_array_equal(*evidence)


def test_partial_fit_failure_cannot_publish_models(batches, monkeypatch):
    reference, test = batches
    detector = make_detector().fit(reference)
    detector.select(test)
    original = RecordingDetector.fit
    calls = 0

    def fail_second(self, X, y=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("model fit failed")
        return original(self, X, y)

    monkeypatch.setattr(RecordingDetector, "fit", fail_second)
    with pytest.raises(RuntimeError, match="model fit failed"):
        detector.fit(reference)
    assert detector.detector_set == []
    assert not detector.is_fitted
    assert detector.last_selection_result is None


def test_malformed_score_shapes_are_rejected(batches, monkeypatch):
    reference, test = batches
    detector = make_detector().fit(reference)
    detector.select(test)
    monkeypatch.setattr(
        RecordingDetector, "decision_function", lambda self, X: np.zeros(2)
    )
    with pytest.raises(ValueError, match="one score per test row"):
        detector.select(test)
    assert detector.last_selection_result is None
    with pytest.raises(ValueError, match="one score per calibration row"):
        detector.fit(reference)
    assert not detector.is_fitted


def test_strategy_isolation_and_invalid_updates(batches):
    reference, test = batches
    strategy = DerandomizedSplits(3, 30)
    detector = ConformalDetector(RecordingDetector(), strategy, seed=42).fit(reference)
    strategy.set_params(n_repetitions=1)
    detector.select(test)
    assert detector.last_selection_result.n_repetitions == 3
    with pytest.raises(ValueError, match="Invalid"):
        strategy.set_params(unknown=1)
    with pytest.raises(ValueError):
        detector.set_params(strategy__n_repetitions=0)
    # Rejected nested changes leave the previously valid configuration intact.
    assert detector.strategy.n_repetitions == 3


@pytest.mark.parametrize("seed", [True, 1.5, -1])
def test_invalid_tie_seed_is_rejected(seed):
    with pytest.raises((TypeError, ValueError)):
        DerandomizedSplits(tie_seed=seed)


def test_refit_rejects_malformed_reference_and_parallelism(batches):
    reference, test = batches
    detector = make_detector().fit(reference)
    for invalid in [np.zeros(5), np.zeros((20, 0))]:
        detector.fit(reference).select(test)
        with pytest.raises(ValueError, match="two-dimensional"):
            detector.fit(invalid)
        assert not detector.is_fitted
        assert detector.last_selection_result is None
    with pytest.raises(ValueError, match="does not support n_jobs"):
        detector.fit(reference, n_jobs=2)


def test_repetition_seeds_support_detector_seed_alias(batches):
    class SeedAliasDetector(RecordingDetector):
        def get_params(self, deep=True):
            return {"seed": self.random_state}

        def set_params(self, **params):
            if params.keys() - {"seed"}:
                raise ValueError("unsupported parameter")
            self.random_state = params["seed"]
            return self

    reference, _ = batches
    detector = ConformalDetector(
        SeedAliasDetector(), DerandomizedSplits(3, 30), seed=42
    ).fit(reference)
    expected = [
        int(stream.generate_state(1)[0])
        for stream in np.random.SeedSequence(42).spawn(2)[0].spawn(3)
    ]
    assert [model.random_state for model in detector.detector_set] == expected
