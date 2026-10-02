"""Unit tests for raw-score false-positive-rate certificates."""

import numpy as np
import pytest

from nonconform._internal import fpr_bounds as core
from nonconform.fdr import FPRCertificate


def _certificate(calibration_scores=None, **kwargs) -> FPRCertificate:
    if calibration_scores is None:
        calibration_scores = np.array([-1.0, 0.0, 2.0, 2.0])
    params = {
        "confidence": 0.8,
        "n_resamples": 25,
        "seed": 7,
    }
    params.update(kwargs)
    return FPRCertificate.from_scores(calibration_scores, **params)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"calibration_scores": np.array([[0.1, 0.2]])}, ValueError, "1D"),
        ({"calibration_scores": np.array([0.1, np.nan])}, ValueError, "finite"),
        ({"calibration_scores": np.array([])}, ValueError, "at least one"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"n_resamples": 0}, ValueError, "positive"),
        ({"seed": -1}, ValueError, "non-negative"),
        ({"seed": "1"}, TypeError, "non-negative"),
    ],
)
def test_fpr_certificate_validates_inputs(kwargs, error_type, match):
    params = {"calibration_scores": np.array([0.1, 0.2, 0.3])}
    params.update(kwargs)

    with pytest.raises(error_type, match=match):
        FPRCertificate.from_scores(**params)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.2, 0.10278619903042452),
        (0.5, 0.3665003817554412),
        (0.8, 0.5320650471562792),
        (0.95, np.inf),
    ],
)
def test_ks_cutoff_matches_independent_one_sided_reference(confidence, expected):
    # The four size-three uniform samples from seed 7 have D+ statistics
    # 0.1027861990, 0.3665003818, 0.3280680288, and 0.5320650472.
    # The corrected ranks are 1, 3, 4, and 5; rank 5 exceeds the four draws.
    certificate = _certificate(
        np.array([-2.0, 0.0, 4.0]),
        confidence=confidence,
        n_resamples=4,
        seed=7,
    )

    assert certificate.critical_value == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_insufficient_resamples_produce_a_unit_band_and_empty_selection():
    certificate = _certificate(confidence=0.95, n_resamples=1)

    assert certificate.critical_value == np.inf
    np.testing.assert_array_equal(
        certificate.bound_at(np.array([-np.inf, 0.0, 2.0, np.inf])),
        np.ones(4),
    )
    assert certificate.threshold_for(0.05) == np.inf
    np.testing.assert_array_equal(
        certificate.select(np.array([-10.0, 0.0, 10.0]), target_fpr=0.05),
        [False, False, False],
    )


def test_default_grid_uses_sorted_unique_scores_and_inclusive_tail_counts():
    certificate = _certificate()

    np.testing.assert_allclose(certificate.thresholds, [-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(certificate.alarm_counts, [4, 3, 2])
    np.testing.assert_allclose(certificate.empirical_fpr, [1.0, 0.75, 0.5])


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
    certificate = FPRCertificate.from_scores(np.ones(1000, dtype=dtype), seed=42)
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
    np.testing.assert_array_equal(
        certificate.select(np.array([-10.0, 0.0, 10.0]), target_fpr=target),
        [False, False, False],
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


def test_same_seed_reproduces_prepared_band_and_queries_do_not_resample(monkeypatch):
    first = _certificate(seed=11)
    second = _certificate(seed=11)

    assert first.critical_value == second.critical_value
    monkeypatch.setattr(
        core,
        "prepare_ks_band",
        lambda **_: pytest.fail("queries must not prepare or resample the band"),
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
