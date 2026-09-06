"""Integration tests for post-hoc conformal FDP bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError

from nonconform import ConformalDetector, Split
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
from nonconform.scoring import ConditionalEmpirical, Probabilistic
from nonconform.structures import ConformalResult


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
