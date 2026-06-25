import numpy as np
import pytest

from nonconform.fdr import (
    EValueSelectionResult,
    conformal_e_value_selection,
    conformal_e_values,
    e_value_false_discovery_control,
)


def test_e_bh_selects_largest_valid_prefix():
    e_values = np.array([30.0, 12.0, 4.0, 1.0])

    selected = e_value_false_discovery_control(e_values, alpha=0.2)

    np.testing.assert_array_equal(selected, np.array([True, True, False, False]))


def test_e_bh_returns_no_discoveries_when_no_prefix_passes():
    selected = e_value_false_discovery_control(np.array([5.0, 4.0]), alpha=0.1)

    np.testing.assert_array_equal(selected, np.array([False, False]))


def test_e_bh_is_monotone_in_alpha():
    e_values = np.array([30.0, 12.0, 4.0, 1.0])

    low_alpha = e_value_false_discovery_control(e_values, alpha=0.05)
    high_alpha = e_value_false_discovery_control(e_values, alpha=0.2)

    assert np.count_nonzero(low_alpha) <= np.count_nonzero(high_alpha)


def test_e_bh_ties_are_deterministic():
    e_values = np.array([100.0, 10.0, 10.0, 10.0, 1.0])

    selected = e_value_false_discovery_control(e_values, alpha=0.2)

    np.testing.assert_array_equal(
        selected,
        np.array([True, True, True, True, False]),
    )


@pytest.mark.parametrize(
    ("e_values", "match"),
    [
        (np.array([]), "at least one"),
        (np.array([[1.0, 2.0]]), "1D"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, -0.1]), "non-negative"),
    ],
)
def test_e_bh_validates_e_values(e_values, match):
    with pytest.raises(ValueError, match=match):
        e_value_false_discovery_control(e_values, alpha=0.2)


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan])
def test_e_bh_validates_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        e_value_false_discovery_control(np.array([1.0]), alpha=alpha)


def test_conformal_e_values_match_hand_computed_single_split():
    test_scores = np.array([3.0, 0.5])
    calib_scores = np.array([0.0, 1.0, 2.0])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)

    np.testing.assert_allclose(e_values, np.array([4.0, 0.0]))


def test_conformal_e_values_average_repeated_splits():
    test_scores = np.array([[3.0, 0.5], [3.0, 1.0]])
    calib_scores = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)

    np.testing.assert_allclose(e_values, np.array([4.0, 2.0]))


def test_conformal_e_values_without_valid_threshold_are_zero():
    test_scores = np.array([1.0, 2.0])
    calib_scores = np.array([3.0, 4.0])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.1)

    np.testing.assert_allclose(e_values, np.zeros(2))


@pytest.mark.parametrize(
    ("test_scores", "calib_scores", "match"),
    [
        (np.array([]), np.array([1.0]), "test_scores"),
        (np.array([1.0]), np.array([]), "calib_scores"),
        (np.array([[1.0]]), np.array([1.0]), "same dimension"),
        (np.ones((2, 3)), np.ones((3, 4)), "same number of repetitions"),
        (np.array([1.0, np.inf]), np.array([1.0]), "finite"),
    ],
)
def test_conformal_e_values_validate_shapes_and_values(
    test_scores,
    calib_scores,
    match,
):
    with pytest.raises(ValueError, match=match):
        conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)


@pytest.mark.parametrize("alpha_bh", [0.0, 1.0, np.nan])
def test_conformal_e_values_validate_alpha_bh(alpha_bh):
    with pytest.raises(ValueError, match="alpha_bh"):
        conformal_e_values(np.array([1.0]), np.array([0.0]), alpha_bh=alpha_bh)


def test_conformal_e_value_selection_defaults_alpha_bh_and_copies_arrays():
    test_scores = np.array([1.0, 2.0])
    calib_scores = np.array([3.0, 4.0])

    result = conformal_e_value_selection(test_scores, calib_scores, alpha=0.2)

    assert isinstance(result, EValueSelectionResult)
    assert result.alpha == pytest.approx(0.2)
    assert result.alpha_bh == pytest.approx(0.02)
    assert result.e_threshold == float("inf")
    assert result.n_repetitions == 1
    np.testing.assert_array_equal(result.selected, np.array([False, False]))

    result.e_values[0] = 99.0
    fresh = conformal_e_value_selection(test_scores, calib_scores, alpha=0.2)
    np.testing.assert_allclose(fresh.e_values, np.zeros(2))
