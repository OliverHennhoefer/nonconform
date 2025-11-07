import numpy as np

from nonconform.utils.stat.statistical import (
    calculate_p_val,
    calculate_weighted_p_val,
)


class TestBasicPValueCalculation:
    def test_basic_p_value_calculation(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=100)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 10

    def test_p_values_in_valid_range(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=100)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_known_scores(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 0) / (1 + 4)
        np.testing.assert_almost_equal(p_values[0], expected)

    def test_score_at_median(self):
        test_scores = np.array([3.0])
        calib_scores = np.array([1.0, 2.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 2) / (1 + 4)
        np.testing.assert_almost_equal(p_values[0], expected)

    def test_multiple_test_scores(self):
        test_scores = np.array([1.0, 3.0, 5.0])
        calib_scores = np.array([2.0, 4.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 3


class TestWeightedPValueCalculation:
    def test_weighted_p_value_calculation(self, sample_scores, sample_weights):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=100)
        test_weights, calib_weights = sample_weights(n_test=10, n_calib=100)
        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert len(p_values) == 10

    def test_weighted_p_values_in_valid_range(self, sample_scores, sample_weights):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=100)
        test_weights, calib_weights = sample_weights(n_test=20, n_calib=100)
        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_uniform_weights_match_standard(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 2.0, 4.0, 6.0])
        test_weights = np.ones(2)
        calib_weights = np.ones(4)

        p_vals_weighted = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        p_vals_standard = calculate_p_val(test_scores, calib_scores)

        np.testing.assert_array_almost_equal(p_vals_weighted, p_vals_standard)

    def test_known_weighted_scores(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 4.0, 6.0])
        test_weights = np.array([1.0])
        calib_weights = np.array([0.5, 0.5, 2.0])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        weighted_sum = 2.0 + 1.0
        total_weight = 3.0 + 1.0
        expected = weighted_sum / total_weight
        np.testing.assert_almost_equal(p_values[0], expected)


class TestPValueRange:
    def test_minimum_p_value_greater_than_zero(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=100)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert np.all(p_values > 0)

    def test_maximum_p_value_is_one(self):
        test_scores = np.array([-10.0])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert p_values[0] == 1.0

    def test_p_value_shape_matches_test_scores(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=50)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert p_values.shape == test_scores.shape


class TestMathematicalCorrectness:
    def test_conservative_adjustment(self):
        test_scores = np.array([10.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 0) / (1 + 3)
        np.testing.assert_almost_equal(p_values[0], expected)

    def test_all_calib_greater_than_test(self):
        test_scores = np.array([0.5])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 5) / (1 + 5)
        assert p_values[0] == expected

    def test_no_calib_greater_than_test(self):
        test_scores = np.array([10.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 0) / (1 + 3)
        np.testing.assert_almost_equal(p_values[0], expected)
