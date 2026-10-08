"""Unit tests for raw-score false-positive-rate certificates."""

import copy
import inspect
import json
import pickle
from functools import partial
from math import comb, sqrt

import numpy as np
import pytest
from scipy.stats import ksone

from nonconform import ConformalDetector
from nonconform._internal import fpr_bounds as core
from nonconform.fdr import FPRCertificate
from nonconform.structures import ConformalResult


def _certificate(calibration_scores=None, **kwargs) -> FPRCertificate:
    if calibration_scores is None:
        calibration_scores = np.array([-1.0, 0.0, 2.0, 2.0])
    params = {"confidence": 0.8}
    params.update(kwargs)
    return FPRCertificate.from_scores(calibration_scores, **params)


def _pickle_roundtrip(certificate, *, protocol):
    return pickle.loads(pickle.dumps(certificate, protocol=protocol))


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"calibration_scores": np.array([[0.1, 0.2]])}, ValueError, "1D"),
        ({"calibration_scores": np.array([0.1, np.nan])}, ValueError, "finite"),
        ({"calibration_scores": np.array([])}, ValueError, "at least one"),
        ({"confidence": 0.0}, ValueError, "confidence"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"confidence": np.nan}, ValueError, "confidence"),
    ],
)
def test_fpr_certificate_validates_inputs(kwargs, error_type, match):
    params = {"calibration_scores": np.array([0.1, 0.2, 0.3])}
    params.update(kwargs)

    with pytest.raises(error_type, match=match):
        FPRCertificate.from_scores(**params)


@pytest.mark.parametrize("confidence", [0.01, 0.5, 0.95, np.nextafter(1.0, 0.0)])
def test_one_score_ks_cutoff_matches_uniform_distribution(confidence):
    # With one uniform observation U, D+ = 1 - U is uniform too.
    certificate = _certificate(np.array([0.0]), confidence=confidence)

    assert certificate.critical_value == pytest.approx(confidence, rel=2e-14, abs=0.0)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.5, (sqrt(3.0) - 1.0) / 2.0),
        (0.95, 1.0 - sqrt(0.05)),
    ],
)
def test_two_score_ks_cutoff_matches_independent_closed_form(confidence, expected):
    # For n=2, P(D+ <= c) = c(1+c) below 1/2 and 1-(1-c)^2 above it.
    certificate = _certificate(np.array([-2.0, 4.0]), confidence=confidence)

    assert certificate.critical_value == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("n_calibration", "confidence"),
    [
        (900, 0.95),
        (1000, 0.95),
        (100_000, 0.95),
        (1, np.nextafter(0.0, 1.0)),
        (1, np.nextafter(1.0, 0.0)),
        (1000, np.nextafter(0.0, 1.0)),
        (1000, 1e-12),
        (1000, np.nextafter(1.0, 0.0)),
    ],
)
def test_cutoff_has_requested_finite_sample_coverage(n_calibration, confidence):
    certificate = _certificate(np.zeros(n_calibration), confidence=confidence)

    assert 0.0 < certificate.critical_value <= 1.0
    coverage = ksone.cdf(certificate.critical_value, n_calibration)
    assert coverage == pytest.approx(confidence, rel=1e-10, abs=5e-13)


def test_cutoff_corrects_previous_seeded_monte_carlo_undercoverage():
    certificate = _certificate(np.zeros(900), confidence=0.95)

    # The old 1,000-draw seed-42 cutoff had only 0.9360805372 coverage.
    old_cutoff = 0.03890119020789279
    assert certificate.critical_value > old_cutoff
    assert ksone.cdf(certificate.critical_value, 900) >= 0.95 - 5e-13


@pytest.mark.parametrize("invalid_cutoff", [np.nan, np.inf, -np.inf, -0.1, 1.1])
def test_invalid_quantile_fails_without_substituting_another_band(
    monkeypatch, invalid_cutoff
):
    monkeypatch.setattr(core.ksone, "ppf", lambda *_: invalid_cutoff)

    with pytest.raises(RuntimeError):
        _certificate()


@pytest.mark.parametrize("confidence", [0.5, 0.95])
@pytest.mark.parametrize("inlier_probability_one", [0.1, 0.5, 0.9])
def test_bernoulli_ties_have_conservative_simultaneous_coverage(
    confidence, inlier_probability_one
):
    n_calibration = 20
    thresholds = np.array([-np.inf, 0.0, np.nextafter(0.0, np.inf), 1.0, np.inf])
    true_fpr = np.array([1.0, 1.0, inlier_probability_one, inlier_probability_one, 0.0])
    coverage = 0.0
    # Enumerate all possible counts, weighted by their exact binomial law.
    for count_one in range(n_calibration + 1):
        scores = np.concatenate(
            [np.zeros(n_calibration - count_one), np.ones(count_one)]
        )
        certificate = _certificate(scores, confidence=confidence)
        if np.all(certificate.bound_at(thresholds) >= true_fpr):
            coverage += (
                comb(n_calibration, count_one)
                * inlier_probability_one**count_one
                * (1.0 - inlier_probability_one) ** (n_calibration - count_one)
            )

    assert coverage >= confidence - 1e-12


@pytest.mark.parametrize(
    ("entry_point", "expected_parameters"),
    [
        (FPRCertificate.from_scores, ["calibration_scores", "confidence"]),
        (ConformalDetector.fpr_bounds, ["self", "confidence"]),
        (ConformalResult.fpr_bounds, ["self", "confidence"]),
    ],
)
def test_fpr_entry_points_expose_only_confidence(entry_point, expected_parameters):
    parameters = inspect.signature(entry_point).parameters

    assert list(parameters) == expected_parameters
    assert parameters["confidence"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["confidence"].default == 0.95


def test_certificate_configuration_has_no_monte_carlo_controls():
    certificate = _certificate()

    assert certificate.method == "ks"
    assert not hasattr(certificate, "seed")
    assert not hasattr(certificate, "n_resamples")
    assert "n_resamples" not in repr(certificate)


def test_default_grid_covers_every_decision_and_inclusive_tail_count():
    certificate = _certificate()

    np.testing.assert_array_equal(
        certificate.thresholds,
        np.r_[-np.inf, np.nextafter([-1.0, 0.0, 2.0], np.inf), np.inf],
    )
    np.testing.assert_array_equal(certificate.alarm_counts, [4, 3, 2, 0, 0])
    np.testing.assert_allclose(certificate.empirical_fpr, [1.0, 0.75, 0.5, 0.0, 0.0])


def test_bound_queries_are_monotone_and_support_extreme_thresholds():
    certificate = _certificate()
    thresholds = np.array([-np.inf, -0.5, 0.0, 1.0, 2.0, np.inf])

    table = certificate.to_frame(thresholds)

    np.testing.assert_array_equal(table.alarm_counts, [4, 3, 3, 2, 2, 0])
    assert np.all(np.diff(table.fpr_upper_bound.to_numpy()) <= 0)
    assert np.all((table.fpr_upper_bound >= 0) & (table.fpr_upper_bound <= 1))
    assert isinstance(certificate.bound_at(0.0), float)
    np.testing.assert_allclose(
        certificate.bound_at(thresholds), table.fpr_upper_bound.to_numpy()
    )


@pytest.mark.parametrize(
    "calibration_scores",
    [
        np.array([1.0]),
        np.array([-1.0, 0.0, 2.0, 2.0]),
        np.array([np.nextafter(0.0, -np.inf), 0.0, np.nextafter(0.0, np.inf)]),
        np.array([-np.finfo(float).max, np.finfo(float).max]),
    ],
    ids=["one-score", "ties", "adjacent-floats", "finite-extremes"],
)
def test_queries_match_direct_inclusive_counts(calibration_scores):
    with np.errstate(over="ignore"):
        thresholds = np.r_[
            -np.inf,
            calibration_scores,
            np.nextafter(calibration_scores, np.inf),
            0.123,
            np.inf,
        ]
    with np.errstate(over="raise"):
        certificate = _certificate(calibration_scores)
    expected_counts = np.count_nonzero(
        calibration_scores[:, None] >= thresholds, axis=0
    )
    expected_bounds = np.minimum(
        1.0, expected_counts / len(calibration_scores) + certificate.critical_value
    )
    expected_bounds[np.isposinf(thresholds)] = 0.0

    table = certificate.to_frame(thresholds)

    np.testing.assert_array_equal(table.alarm_counts, expected_counts)
    np.testing.assert_allclose(table.fpr_upper_bound, expected_bounds)
    assert np.all(certificate.thresholds[1:] > certificate.thresholds[:-1])


def test_select_uses_anomalous_higher_inclusive_threshold_rule():
    certificate = _certificate()
    scores = np.array([2.0, 2.0, 1.999, -1.0])

    np.testing.assert_array_equal(
        certificate.select(scores, threshold=2.0),
        [True, True, False, False],
    )


def test_threshold_for_uses_nextafter_to_step_past_ties():
    certificate = _certificate(np.array([0.0, 0.0, 1.0]), confidence=0.5)
    target = float(certificate.bound_at(np.nextafter(0.0, np.inf)))

    threshold = certificate.threshold_for(target)

    assert threshold == np.nextafter(0.0, np.inf)
    assert certificate.bound_at(threshold) <= target


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_threshold_for_preserves_tie_exclusion_in_direct_score_comparisons(dtype):
    certificate = FPRCertificate.from_scores(np.ones(1000, dtype=dtype))
    scores = np.array([1.0, 2.0, 0.0, 1.0], dtype=dtype)

    threshold = certificate.threshold_for(0.05)

    assert certificate.bound_at(threshold) <= 0.05
    np.testing.assert_array_equal(scores >= threshold, [False, True, False, False])
    np.testing.assert_array_equal(
        scores >= threshold, certificate.select(scores, target_fpr=0.05)
    )


def test_threshold_for_returns_infinity_when_target_is_below_band_floor():
    certificate = _certificate()
    target = certificate.critical_value / 2.0

    threshold = certificate.threshold_for(target)

    assert np.isinf(threshold)
    assert certificate.bound_at(threshold) == 0.0
    np.testing.assert_array_equal(
        certificate.select(np.array([-10.0, 0.0, 10.0]), target_fpr=target),
        [False, False, False],
    )


def test_default_grid_exposes_feasible_thresholds_above_tied_scores():
    certificate = FPRCertificate.from_scores(np.ones(1000))
    table = certificate.to_frame()
    threshold = certificate.threshold_for(0.05)

    assert np.isfinite(threshold)
    assert threshold in table.threshold.to_numpy()
    assert np.any(table.fpr_upper_bound <= 0.05)
    assert table.iloc[-1].threshold == np.inf
    assert table.iloc[-1].fpr_upper_bound == 0.0
    np.testing.assert_array_equal(table.threshold, certificate.thresholds)
    np.testing.assert_array_equal(table.alarm_counts, certificate.alarm_counts)
    np.testing.assert_array_equal(table.empirical_fpr, certificate.empirical_fpr)
    np.testing.assert_array_equal(table.fpr_upper_bound, certificate.fpr_upper_bounds)


@pytest.mark.parametrize("target", [0.001, 0.05, 0.5, 0.99])
def test_threshold_inversion_matches_the_default_grid(target):
    certificate = FPRCertificate.from_scores(np.ones(1000))
    table = certificate.to_frame()
    eligible = table.loc[table.fpr_upper_bound <= target, "threshold"]

    threshold = certificate.threshold_for(target)

    assert threshold == eligible.iloc[0]
    assert certificate.bound_at(threshold) <= target


def test_select_preserves_tie_exclusion_after_json_threshold_roundtrip():
    certificate = FPRCertificate.from_scores(np.ones(1000, dtype=np.float32))
    threshold = json.loads(json.dumps(certificate.threshold_for(0.05)))
    scores = np.array([1.0, 2.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_array_equal(
        certificate.select(scores, threshold=threshold), [False, True, False, False]
    )


def test_select_requires_exactly_one_threshold_specification():
    certificate = _certificate()
    scores = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="exactly one"):
        certificate.select(scores)
    with pytest.raises(ValueError, match="exactly one"):
        certificate.select(scores, threshold=0.0, target_fpr=0.1)
    with pytest.raises(ValueError, match="scalar"):
        certificate.select(scores, threshold=np.array([0.0, 1.0]))


def test_custom_grid_preserves_order_duplicates_and_is_independent():
    certificate = _certificate()
    grid = np.array([2.0, -1.0, 2.0])
    expected = certificate.to_frame(grid).copy()

    table = certificate.to_frame(grid)
    table.iloc[:, :] = 0

    np.testing.assert_allclose(certificate.to_frame(grid), expected)


def test_certificate_is_immutable_and_isolates_source_arrays():
    calibration_scores = np.array([0.1, 0.2, 0.3])
    certificate = _certificate(calibration_scores)
    calibration_scores[:] = 99.0

    np.testing.assert_allclose(certificate.calibration_scores, [0.1, 0.2, 0.3])
    for name in [
        "calibration_scores",
        "thresholds",
        "alarm_counts",
        "empirical_fpr",
        "fpr_upper_bounds",
    ]:
        exposed = getattr(certificate, name)
        with pytest.raises(ValueError):
            exposed[:] = 0
        with pytest.raises(ValueError):
            exposed.flags.writeable = True
    with pytest.raises(AttributeError):
        certificate.confidence = 0.5
    with pytest.raises(AttributeError):
        certificate.calibration_scores = np.array([0.0])


@pytest.mark.parametrize(
    "transfer",
    [
        copy.copy,
        copy.deepcopy,
        *[
            partial(_pickle_roundtrip, protocol=protocol)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ],
    ],
    ids=[
        "copy",
        "deepcopy",
        *[f"pickle-{p}" for p in range(pickle.HIGHEST_PROTOCOL + 1)],
    ],
)
def test_certificate_transfer_preserves_evidence_band_and_immutability(
    transfer, monkeypatch
):
    certificate = _certificate(np.array([2.0, -1.0, 2.0, 0.0]))
    expected = certificate.to_frame()
    monkeypatch.setattr(
        core,
        "prepare_ks_band",
        lambda **_: pytest.fail("copying must preserve the prepared KS band"),
    )

    restored = transfer(certificate)

    assert isinstance(restored, FPRCertificate)
    assert restored.critical_value == certificate.critical_value
    assert restored.confidence == certificate.confidence
    np.testing.assert_array_equal(
        restored.calibration_scores, certificate.calibration_scores
    )
    np.testing.assert_array_equal(restored.to_frame(), expected)
    np.testing.assert_array_equal(
        restored.select(np.array([-1.0, 0.0, 3.0]), target_fpr=0.8),
        certificate.select(np.array([-1.0, 0.0, 3.0]), target_fpr=0.8),
    )
    for name in [
        "calibration_scores",
        "thresholds",
        "alarm_counts",
        "fpr_upper_bounds",
    ]:
        with pytest.raises(ValueError):
            getattr(restored, name).flags.writeable = True
    for name in ["_calibration_scores", "_thresholds", "_alarm_counts"]:
        with pytest.raises(ValueError):
            getattr(restored, name).flags.writeable = True
    with pytest.raises(AttributeError):
        restored.confidence = 0.5
    with pytest.raises(AttributeError):
        del restored._band


def test_construction_and_queries_use_no_randomness_or_repeated_preparation(
    monkeypatch,
):
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *_: pytest.fail("FPR certificates must not draw randomness"),
    )
    first = _certificate()
    second = _certificate()

    assert first.critical_value == second.critical_value
    monkeypatch.setattr(
        core,
        "prepare_ks_band",
        lambda **_: pytest.fail("queries must reuse the prepared band"),
    )
    first.bound_at([0.0, 1.0])
    first.threshold_for(0.9)
    first.to_frame()
    first.select(np.array([0.0, 2.0]), threshold=1.0)


def test_constructor_requires_factory():
    with pytest.raises(TypeError, match="from_scores"):
        FPRCertificate()


@pytest.mark.parametrize("threshold", [np.nan, [[0.1]], "invalid"])
def test_threshold_query_validation(threshold):
    certificate = _certificate()

    with pytest.raises(ValueError):
        certificate.bound_at(threshold)
