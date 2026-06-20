import numpy as np
import pytest

from nonconform.cleaning import apply_label_trim, select_label_trim_candidates
from nonconform.scoring import Empirical


def test_select_label_trim_candidates_returns_top_scores_in_descending_priority():
    scores = np.array([0.1, 4.0, 2.5, 7.0, 3.0])

    candidates = select_label_trim_candidates(scores, label_budget=3)

    np.testing.assert_array_equal(candidates, np.array([3, 1, 4]))


def test_select_label_trim_candidates_breaks_ties_by_original_order():
    scores = np.array([0.1, 4.0, 4.0, 2.0])

    candidates = select_label_trim_candidates(scores, label_budget=3)

    np.testing.assert_array_equal(candidates, np.array([1, 2, 3]))


def test_select_label_trim_candidates_zero_budget_returns_empty_indices():
    scores = np.array([1.0, 2.0, 3.0])

    candidates = select_label_trim_candidates(scores, label_budget=0)

    assert candidates.dtype == int
    assert candidates.size == 0


def test_select_label_trim_candidates_caps_budget_to_score_count():
    scores = np.array([1.0, 3.0])

    candidates = select_label_trim_candidates(scores, label_budget=5)

    np.testing.assert_array_equal(candidates, np.array([1, 0]))


@pytest.mark.parametrize("label_budget", [-1, 1.5, True, "2"])
def test_select_label_trim_candidates_rejects_invalid_budgets(label_budget):
    scores = np.array([1.0, 2.0, 3.0])

    with pytest.raises((TypeError, ValueError)):
        select_label_trim_candidates(scores, label_budget=label_budget)


def test_select_label_trim_candidates_rejects_non_1d_scores():
    with pytest.raises(ValueError, match="1D array"):
        select_label_trim_candidates(np.array([[1.0], [2.0]]), label_budget=1)


def test_select_label_trim_candidates_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        select_label_trim_candidates(np.array([1.0, np.nan]), label_budget=1)


def test_select_label_trim_candidates_does_not_mutate_scores():
    scores = np.array([1.0, 3.0, 2.0])
    original = scores.copy()

    candidates = select_label_trim_candidates(scores, label_budget=2)
    candidates[0] = 0

    np.testing.assert_array_equal(scores, original)


def test_apply_label_trim_removes_only_labeled_outlier_candidates():
    scores = np.array([0.5, 8.0, 1.0, 7.0, 2.0])
    candidates = np.array([1, 3, 4])
    labels = np.array([1, 0, 1])

    result = apply_label_trim(scores, candidates, labels)

    np.testing.assert_array_equal(result.removed_indices, np.array([1, 4]))
    np.testing.assert_array_equal(
        result.keep_mask, np.array([True, False, True, True, False])
    )
    np.testing.assert_array_equal(result.trimmed_scores, np.array([0.5, 1.0, 7.0]))
    np.testing.assert_array_equal(result.candidate_indices, candidates)
    assert result.n_original == 5
    assert result.n_candidates == 3
    assert result.n_removed == 2
    assert result.n_kept == 3


def test_apply_label_trim_keeps_all_scores_when_no_candidates_are_outliers():
    scores = np.array([0.5, 8.0, 1.0])

    result = apply_label_trim(scores, np.array([1, 2]), np.array([0, 0]))

    np.testing.assert_array_equal(result.removed_indices, np.array([], dtype=int))
    np.testing.assert_array_equal(result.keep_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(result.trimmed_scores, scores)


def test_apply_label_trim_can_remove_all_candidates():
    scores = np.array([0.5, 8.0, 1.0, 7.0])

    result = apply_label_trim(scores, np.array([1, 3]), np.array([1, 1]))

    np.testing.assert_array_equal(result.removed_indices, np.array([1, 3]))
    np.testing.assert_array_equal(result.trimmed_scores, np.array([0.5, 1.0]))


def test_apply_label_trim_supports_custom_outlier_label():
    scores = np.array([0.5, 8.0, 1.0])

    result = apply_label_trim(
        scores,
        np.array([1, 2]),
        np.array(["outlier", "normal"]),
        outlier_label="outlier",
    )

    np.testing.assert_array_equal(result.removed_indices, np.array([1]))


def test_apply_label_trim_rejects_duplicate_candidate_indices():
    with pytest.raises(ValueError, match="duplicates"):
        apply_label_trim(np.array([0.5, 8.0]), np.array([1, 1]), np.array([1, 0]))


def test_apply_label_trim_rejects_out_of_bounds_candidate_indices():
    with pytest.raises(IndexError, match="bounds"):
        apply_label_trim(np.array([0.5, 8.0]), np.array([2]), np.array([1]))


def test_apply_label_trim_rejects_non_integer_candidate_indices():
    with pytest.raises(TypeError, match="integers"):
        apply_label_trim(np.array([0.5, 8.0]), np.array([1.0]), np.array([1]))


def test_apply_label_trim_rejects_candidate_label_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        apply_label_trim(np.array([0.5, 8.0]), np.array([1]), np.array([1, 0]))


def test_apply_label_trim_does_not_mutate_inputs_or_expose_mutable_input_views():
    scores = np.array([0.5, 8.0, 1.0])
    candidates = np.array([1])
    labels = np.array([1])

    result = apply_label_trim(scores, candidates, labels)
    result.trimmed_scores[0] = 100.0
    result.keep_mask[0] = False
    result.removed_indices[0] = 0
    result.candidate_indices[0] = 0

    np.testing.assert_array_equal(scores, np.array([0.5, 8.0, 1.0]))
    np.testing.assert_array_equal(candidates, np.array([1]))
    np.testing.assert_array_equal(labels, np.array([1]))


def test_label_trim_changes_empirical_p_values_only_when_used_explicitly():
    calibration_scores = np.array([0.0, 1.0, 2.0, 10.0])
    test_scores = np.array([9.0])
    candidates = select_label_trim_candidates(calibration_scores, label_budget=1)
    result = apply_label_trim(calibration_scores, candidates, np.array([1]))
    estimator = Empirical()

    standard = estimator.compute_p_values(test_scores, calibration_scores)
    trimmed = estimator.compute_p_values(test_scores, result.trimmed_scores)

    np.testing.assert_allclose(standard, np.array([0.4]))
    np.testing.assert_allclose(trimmed, np.array([0.25]))
    np.testing.assert_array_equal(calibration_scores, np.array([0.0, 1.0, 2.0, 10.0]))
