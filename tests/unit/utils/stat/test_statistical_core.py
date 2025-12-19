import numpy as np

from nonconform.scoring import Empirical, calculate_p_val, calculate_weighted_p_val


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
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
        expected = (1 + 0) / (1 + 4)
        np.testing.assert_almost_equal(p_values[0], expected)

    def test_score_at_median(self):
        test_scores = np.array([3.0])
        calib_scores = np.array([1.0, 2.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
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
            test_scores, calib_scores, test_weights, calib_weights, randomize=False
        )
        p_vals_standard = calculate_p_val(test_scores, calib_scores, randomize=False)

        np.testing.assert_array_almost_equal(p_vals_weighted, p_vals_standard)

    def test_known_weighted_scores(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 4.0, 6.0])
        test_weights = np.array([1.0])
        calib_weights = np.array([0.5, 0.5, 2.0])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights, randomize=False
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
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
        assert p_values[0] == 1.0

    def test_p_value_shape_matches_test_scores(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=50)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert p_values.shape == test_scores.shape


class TestMathematicalCorrectness:
    def test_conservative_adjustment(self):
        test_scores = np.array([10.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
        expected = (1 + 0) / (1 + 3)
        np.testing.assert_almost_equal(p_values[0], expected)

    def test_all_calib_greater_than_test(self):
        test_scores = np.array([0.5])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
        expected = (1 + 5) / (1 + 5)
        assert p_values[0] == expected

    def test_no_calib_greater_than_test(self):
        test_scores = np.array([10.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        p_values = calculate_p_val(test_scores, calib_scores, randomize=False)
        expected = (1 + 0) / (1 + 3)
        np.testing.assert_almost_equal(p_values[0], expected)


class TestRandomizedPValues:
    """Tests for randomized p-value computation (Jin & Candes 2023)."""

    def test_randomized_p_values_in_valid_range(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=100)
        p_values = calculate_p_val(test_scores, calib_scores, randomize=True)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_randomized_weighted_p_values_in_valid_range(
        self, sample_scores, sample_weights
    ):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=100)
        test_weights, calib_weights = sample_weights(n_test=20, n_calib=100)
        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights, randomize=True
        )
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_reproducibility_with_seed(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 3.0, 5.0, 7.0])  # has ties

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        p_values1 = calculate_p_val(test_scores, calib_scores, randomize=True, rng=rng1)
        p_values2 = calculate_p_val(test_scores, calib_scores, randomize=True, rng=rng2)

        np.testing.assert_array_equal(p_values1, p_values2)

    def test_weighted_reproducibility_with_seed(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 3.0, 5.0, 7.0])
        test_weights = np.array([1.0, 2.0])
        calib_weights = np.array([1.0, 1.0, 1.0, 1.0])

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        p_values1 = calculate_weighted_p_val(
            test_scores,
            calib_scores,
            test_weights,
            calib_weights,
            randomize=True,
            rng=rng1,
        )
        p_values2 = calculate_weighted_p_val(
            test_scores,
            calib_scores,
            test_weights,
            calib_weights,
            randomize=True,
            rng=rng2,
        )

        np.testing.assert_array_equal(p_values1, p_values2)

    def test_randomized_differs_from_non_randomized_with_ties(self):
        """When ties exist, randomized output should differ from non-randomized."""
        test_scores = np.array([3.0])
        calib_scores = np.array([3.0, 3.0, 3.0, 3.0])  # all ties

        rng = np.random.default_rng(42)
        p_random = calculate_p_val(test_scores, calib_scores, randomize=True, rng=rng)
        p_deterministic = calculate_p_val(test_scores, calib_scores, randomize=False)

        # With all ties, randomized p-value should differ (unless U=1 exactly)
        assert p_random[0] != p_deterministic[0]
        # Both should be in valid range
        assert 0.0 <= p_random[0] <= 1.0
        assert 0.0 <= p_deterministic[0] <= 1.0

    def test_no_ties_gives_similar_results(self):
        """When no ties exist, randomized and non-randomized should give same result."""
        test_scores = np.array([2.5])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0])  # no ties with test

        rng = np.random.default_rng(42)
        p_random = calculate_p_val(test_scores, calib_scores, randomize=True, rng=rng)
        p_deterministic = calculate_p_val(test_scores, calib_scores, randomize=False)

        # With no ties, the only difference is the test point's own weight randomization
        # p_deterministic = (1 + n_greater) / (1 + n_cal)
        # p_random = (n_greater + (0 + 1) * U) / (1 + n_cal)
        # They differ only by U term, but should be close
        assert abs(p_random[0] - p_deterministic[0]) < 1.0 / (1 + len(calib_scores))


class TestEmpiricalClassRandomization:
    """Tests for Empirical class with randomize parameter."""

    def test_empirical_default_is_not_randomized(self):
        estimation = Empirical()
        assert estimation._randomize is False

    def test_empirical_randomize_false(self):
        estimation = Empirical(randomize=False)
        assert estimation._randomize is False

    def test_empirical_reproducibility_with_seed(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 3.0, 5.0, 7.0])

        estimation1 = Empirical(randomize=True)
        estimation1.set_seed(42)

        estimation2 = Empirical(randomize=True)
        estimation2.set_seed(42)

        p_values1 = estimation1.compute_p_values(test_scores, calib_scores)
        p_values2 = estimation2.compute_p_values(test_scores, calib_scores)

        np.testing.assert_array_equal(p_values1, p_values2)

    def test_empirical_weighted_reproducibility_with_seed(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 3.0, 5.0, 7.0])
        weights = (np.array([1.0, 1.0, 1.0, 1.0]), np.array([1.0, 2.0]))

        estimation1 = Empirical(randomize=True)
        estimation1.set_seed(42)

        estimation2 = Empirical(randomize=True)
        estimation2.set_seed(42)

        p_values1 = estimation1.compute_p_values(test_scores, calib_scores, weights)
        p_values2 = estimation2.compute_p_values(test_scores, calib_scores, weights)

        np.testing.assert_array_equal(p_values1, p_values2)

    def test_empirical_deterministic_mode(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 2.0, 3.0, 4.0])

        estimation = Empirical(randomize=False)
        p_values = estimation.compute_p_values(test_scores, calib_scores)

        expected = (1 + 0) / (1 + 4)
        np.testing.assert_almost_equal(p_values[0], expected)
