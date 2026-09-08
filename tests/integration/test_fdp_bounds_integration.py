"""Integration tests for post-hoc conformal FDP bounds."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from nonconform import ConformalDetector, Split
from nonconform._internal import TieBreakMode
from nonconform._internal.provenance import (
    CalibrationMode,
    EstimationFamily,
    StrategyFamily,
)
from nonconform.fdr import FDPCertificate
from nonconform.resampling import (
    CrossValidation,
    DerandomizedSplits,
    JackknifeBootstrap,
)
from nonconform.scoring import ConditionalEmpirical, Empirical, Probabilistic
from nonconform.structures import ConformalResult


class ScoreDetector(BaseEstimator):
    """Expose the first input column as scores to control exact tie patterns."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        return np.asarray(X)[:, 0]


def _detached_score_detector(calibration_scores, tie_break):
    model = ScoreDetector().fit(np.zeros((2, 1)))
    return ConformalDetector(
        model,
        strategy=Split(),
        estimation=Empirical(tie_break=tie_break),
        seed=4,
    ).calibrate(np.asarray(calibration_scores).reshape(-1, 1))


def test_fdp_bounds_do_not_change_existing_unweighted_selection(simple_dataset):
    """FDP bounds are an opt-in report over existing conformal p-values."""
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=50, n_features=4)
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.25),
        seed=14,
    )

    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)
    result = detector.last_result
    assert result is not None

    bh_mask = false_discovery_control(p_values, method="bh") <= 0.2
    bounds = result.fdp_bounds(
        confidence=0.8,
        n_resamples=25,
        seed=14,
    )
    select_mask = detector.select(x_test, alpha=0.2)

    np.testing.assert_array_equal(select_mask, bh_mask)
    np.testing.assert_array_equal(bounds.select(0.1), p_values <= 0.1)
    assert np.all((0.0 <= bounds.fdp_upper_bounds) & (bounds.fdp_upper_bounds <= 1.0))


@pytest.fixture
def fitted_batch():
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(100, 3))
    batch = rng.normal(size=(20, 3))
    detector = ConformalDetector(
        IsolationForest(n_estimators=5), strategy=Split(n_calib=25), seed=4
    ).fit(reference)
    return detector, reference, batch


@pytest.mark.parametrize("pandas_input", [False, True])
def test_native_entry_points_equivalent_and_independent(
    fitted_batch, monkeypatch, pandas_input
):
    detector, reference, batch = fitted_batch
    if pandas_input:
        batch = pd.DataFrame(batch, index=np.arange(20) + 100)
    calls = 0
    original = detector.compute_p_values

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(detector, "compute_p_values", counted)
    options = dict(confidence=0.8, n_resamples=25, seed=5)
    direct = detector.fdp_bounds(batch, **options)
    assert calls == 1
    snapshot = detector.last_result

    def forbidden(*args, **kwargs):
        pytest.fail("snapshot certification rescored")

    monkeypatch.setattr(detector, "compute_p_values", forbidden)
    cached = snapshot.fdp_bounds(**options)
    raw = FDPCertificate.from_p_values(snapshot.p_values, n_calibration=25, **options)
    np.testing.assert_array_equal(direct.to_frame(), cached.to_frame())
    np.testing.assert_array_equal(direct.to_frame(), raw.to_frame())
    for target in [0, 0.1, 0.5, 1]:
        assert direct.threshold_for(max_fdp=target) == cached.threshold_for(
            max_fdp=target
        )
        assert direct.threshold_for(max_fdp=target) == raw.threshold_for(max_fdp=target)
    assert isinstance(direct.select(0.1), np.ndarray)
    expected = direct.to_frame()
    snapshot.p_values[:] = 0
    snapshot.calib_scores[:] = 42
    detector.fit(reference + 2)
    np.testing.assert_array_equal(direct.to_frame(), expected)
    np.testing.assert_array_equal(cached.to_frame(), expected)


def test_detached_scope_and_snapshot_validation(fitted_batch):
    _, reference, batch = fitted_batch
    fitted_model = IsolationForest(n_estimators=5, random_state=4).fit(reference[:50])
    detached = ConformalDetector(fitted_model, strategy=Split(), seed=4)
    detached.calibrate(reference[50:])
    certificate = detached.fdp_bounds(batch, method="ks")
    result = detached.last_result
    assert result._provenance.calibration_mode is CalibrationMode.DETACHED
    np.testing.assert_array_equal(
        certificate.to_frame(), result.fdp_bounds(method="ks").to_frame()
    )
    assert certificate.n_calibration == 50
    for field, value, match in [
        ("p_values", None, "p_values"),
        ("calib_scores", None, "calib_scores"),
        ("p_values", [0.1], "batch dimensions"),
        ("p_values", np.full(20, np.nan), "finite"),
        ("calib_scores", [], "at least one"),
        ("calib_scores", [[0.1]], "1D"),
        ("calib_scores", [np.nan], "finite"),
        ("test_scores", [0.1], "batch size"),
        ("test_weights", np.ones(20), "unweighted"),
        ("calib_weights", np.ones(50), "unweighted"),
    ]:
        invalid = result.copy()
        setattr(invalid, field, value)
        with pytest.raises(ValueError, match=match):
            invalid.fdp_bounds(method="ks")


def test_unknown_and_forged_legacy_metadata_are_rejected():
    for metadata in [
        {},
        {
            "nonconform": {
                "strategy": "Split",
                "estimation": "Empirical",
                "weighted": False,
            }
        },
    ]:
        result = ConformalResult(
            p_values=np.array([0.1]), calib_scores=np.arange(10), metadata=metadata
        )
        with pytest.raises(ValueError, match="native provenance"):
            result.fdp_bounds(method="ks")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"strategy_family": StrategyFamily.OTHER}, "split"),
        ({"estimation_family": EstimationFamily.CONDITIONAL_EMPIRICAL}, "empirical"),
        ({"estimation_family": EstimationFamily.OTHER}, "empirical"),
        ({"weighted": True}, "unweighted"),
        ({"test_batch_signature": None}, "batch dimensions"),
        ({"calibration_mode": None}, "fitted or calibrated"),
        ({"empirical_tie_break": None}, "tie"),
        ({"empirical_tie_break": "classical"}, "tie"),
        ({"empirical_tie_break": "unknown"}, "tie"),
    ],
)
def test_native_scope_cannot_be_overridden_by_metadata(fitted_batch, change, match):
    from dataclasses import replace

    detector, _, batch = fitted_batch
    detector.compute_p_values(batch)
    result = detector.last_result
    result._provenance = replace(result._provenance, **change)
    with pytest.raises(ValueError, match=match):
        result.fdp_bounds(method="ks")


def test_legacy_native_provenance_without_tie_mode_is_rejected(fitted_batch):
    detector, _, batch = fitted_batch
    detector.compute_p_values(batch)
    result = detector.last_result
    # Emulate a legacy native snapshot whose slotted provenance predates this field.
    object.__delattr__(result._provenance, "empirical_tie_break")
    with pytest.raises(ValueError, match="tie"):
        result.fdp_bounds(method="ks")


@pytest.mark.parametrize(
    "strategy,estimation,match",
    [
        (CrossValidation(k=2), None, "split"),
        (JackknifeBootstrap(n_bootstraps=2), None, "split"),
        (DerandomizedSplits(n_repetitions=1), None, "split"),
        (Split(), Probabilistic(), "empirical"),
        (Split(), ConditionalEmpirical(), "empirical"),
    ],
)
def test_detector_rejects_unsupported_scope_before_scoring(
    fitted_batch, monkeypatch, strategy, estimation, match
):
    _, reference, batch = fitted_batch
    detector = ConformalDetector(
        IsolationForest(n_estimators=5),
        strategy=strategy,
        estimation=estimation,
        seed=4,
    ).fit(reference)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported certification scored the batch")

    monkeypatch.setattr(detector, "compute_p_values", forbidden)
    with pytest.raises(ValueError, match=match):
        detector.fdp_bounds(batch)


def test_unfitted_detector_rejected():
    detector = ConformalDetector(IsolationForest(n_estimators=5), strategy=Split())
    with pytest.raises(NotFittedError):
        detector.fdp_bounds(np.zeros((2, 3)))


def test_certificate_seed_is_independent_of_fitting_seed(fitted_batch, monkeypatch):
    from nonconform._internal import fdp_bounds as core

    detector, _, batch = fitted_batch
    original = core._mc_summary_quantile
    seen = []

    def recorded(**kwargs):
        seen.append(kwargs["seed"])
        return original(**kwargs)

    monkeypatch.setattr(core, "_mc_summary_quantile", recorded)
    models = detector.detector_set
    calibration = detector.calibration_set
    fresh = detector.fdp_bounds(batch, n_resamples=25)
    seeded = detector.fdp_bounds(batch, n_resamples=25, seed=8)
    assert seen == [None, 8]
    assert fresh.seed is None
    assert seeded.seed == 8
    assert detector.seed == 4
    assert all(a is b for a, b in zip(models, detector.detector_set, strict=True))
    np.testing.assert_array_equal(calibration, detector.calibration_set)
    np.testing.assert_array_equal(fresh.p_values, seeded.p_values)


def test_weighted_detector_rejected_before_scoring(fitted_batch, monkeypatch):
    from nonconform.weighting import SklearnWeightEstimator

    _, reference, batch = fitted_batch
    detector = ConformalDetector(
        IsolationForest(n_estimators=5),
        strategy=Split(n_calib=25),
        weight_estimator=SklearnWeightEstimator(),
        seed=4,
    ).fit(reference)

    def forbidden(*args, **kwargs):
        pytest.fail("weighted certification scored the batch")

    monkeypatch.setattr(detector, "compute_p_values", forbidden)
    with pytest.raises(ValueError, match="unweighted"):
        detector.fdp_bounds(batch)


@pytest.mark.parametrize("tie_break", ["classical", "randomized"])
@pytest.mark.parametrize("detached", [False, True])
def test_tie_mode_provenance_and_target_selection_across_native_workflows(
    tie_break, detached, monkeypatch
):
    reference = np.arange(30.0).reshape(-1, 1)
    batch = np.array([[40.0], [45.0], [50.0]])
    if detached:
        detector = _detached_score_detector(reference.ravel(), tie_break)
    else:
        detector = ConformalDetector(
            ScoreDetector(),
            strategy=Split(n_calib=15),
            estimation=Empirical(tie_break=tie_break),
            seed=4,
        ).fit(reference)
    direct = detector.fdp_bounds(batch, method="ks")
    result = detector.last_result
    expected_mode = (
        TieBreakMode.CLASSICAL if tie_break == "classical" else TieBreakMode.RANDOMIZED
    )
    assert result._provenance.empirical_tie_break is expected_mode
    assert result._provenance.calibration_mode is (
        CalibrationMode.DETACHED if detached else CalibrationMode.INTEGRATED
    )
    copied = result.copy()
    assert copied._provenance is result._provenance
    with pytest.raises(FrozenInstanceError):
        result._provenance.empirical_tie_break = None

    def forbidden(*args, **kwargs):
        pytest.fail("certificate reuse rescored the test family")

    monkeypatch.setattr(detector, "compute_p_values", forbidden)
    detector.estimation = Empirical(
        tie_break="randomized" if tie_break == "classical" else "classical"
    )
    assert copied._provenance.empirical_tie_break is expected_mode
    snapshot = copied.fdp_bounds(method="ks")
    expert = FDPCertificate.from_p_values(
        result.p_values, n_calibration=len(result.calib_scores), method="ks"
    )
    for certificate in [snapshot, expert]:
        np.testing.assert_array_equal(certificate.to_frame(), direct.to_frame())
        for target in [0, 0.1, 0.5, 1]:
            assert certificate.threshold_for(max_fdp=target) == direct.threshold_for(
                max_fdp=target
            )


def test_classical_certification_allows_cross_ties_and_absent_test_scores():
    detector = _detached_score_detector([0, 1, 1, 2], "classical")
    batch = np.array([[1], [1], [3]])
    expected_p_values = detector.compute_p_values(batch)
    certificate = detector.fdp_bounds(batch, method="ks")
    np.testing.assert_array_equal(certificate.p_values, expected_p_values)
    np.testing.assert_array_equal(expected_p_values, [0.8, 0.8, 0.2])
    result = detector.last_result
    result.test_scores = None
    np.testing.assert_array_equal(
        result.fdp_bounds(method="ks").to_frame(), certificate.to_frame()
    )


def test_randomized_cross_ties_rejected_even_when_p_values_are_distinct(monkeypatch):
    from nonconform._internal import fdp_bounds as core

    detector = _detached_score_detector([0, 1, 1, 2], "randomized")
    batch = np.array([[1], [1], [3]])
    p_values = detector.compute_p_values(batch)
    assert len(np.unique(p_values)) == len(batch)
    snapshot = detector.last_result

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported randomized cross-ties reached Monte Carlo sampling")

    monkeypatch.setattr(core, "_sample_conformal_null_p_values", forbidden)
    for certify in [snapshot.fdp_bounds, lambda: detector.fdp_bounds(batch)]:
        with pytest.raises(ValueError, match="tie"):
            certify()
    # Rejecting certification does not change the already supported p-value API.
    np.testing.assert_array_equal(detector.compute_p_values(batch), p_values)


@pytest.mark.parametrize(
    "calibration_scores,test_scores",
    [
        ([0, 0, 2, 2], [1, 3]),
        ([0, 2, 4], [1, 1, 3]),
        ([0, 0, 2, 2], [1, 1, 3]),
        ([0.0, 1.0], [np.nextafter(1.0, 2.0), 2.0]),
    ],
)
def test_randomized_certification_allows_within_set_duplicates_and_near_cross_ties(
    calibration_scores, test_scores
):
    detector = _detached_score_detector(calibration_scores, "randomized")
    batch = np.asarray(test_scores).reshape(-1, 1)
    certificate = detector.fdp_bounds(batch, method="ks")
    result = detector.last_result
    np.testing.assert_array_equal(certificate.p_values, result.p_values)
    np.testing.assert_array_equal(
        certificate.to_frame(), result.copy().fdp_bounds(method="ks").to_frame()
    )


def test_randomized_cross_tie_check_preserves_large_integer_score_precision():
    reference = (2**53 + np.arange(0, 80, 2, dtype=np.int64)).reshape(-1, 1)
    detector = ConformalDetector(
        ScoreDetector(),
        strategy=Split(n_calib=0.5),
        estimation=Empirical(tie_break="randomized"),
        aggregation="maximum",
        score_polarity="higher_is_anomalous",
        seed=1,
    ).fit(reference)
    batch = np.array([[detector.calibration_set[0] + 1]], dtype=np.int64)
    detector.compute_p_values(batch)
    result = detector.last_result
    assert result.calib_scores.dtype == np.int64
    assert result.test_scores.dtype == np.int64
    assert not np.any(result.calib_scores == result.test_scores[0])
    certificate = result.fdp_bounds(method="ks")
    np.testing.assert_array_equal(certificate.p_values, result.p_values)


@pytest.mark.parametrize(
    "test_scores,match",
    [
        (None, "test_scores"),
        ([1.0], "batch size"),
        ([[1.0, 3.0]], "1D"),
        ([np.nan, 3.0], "finite"),
        ([1.0, np.inf], "finite"),
    ],
)
def test_randomized_certification_requires_complete_valid_test_scores(
    monkeypatch, test_scores, match
):
    from nonconform._internal import fdp_bounds as core

    detector = _detached_score_detector([0, 2, 4], "randomized")
    detector.compute_p_values(np.array([[1], [3]]))
    result = detector.last_result
    result.test_scores = test_scores

    def forbidden(*args, **kwargs):
        pytest.fail("invalid randomized scores reached Monte Carlo sampling")

    monkeypatch.setattr(core, "_sample_conformal_null_p_values", forbidden)
    with pytest.raises(ValueError, match=match):
        result.fdp_bounds()
