import numpy as np

from nonconform.utils.stat.statistical import (
    calculate_p_val,
    calculate_weighted_p_val,
)


class TestEmptyInputs:
    def test_single_calibration_score(self):
        test_scores = np.array([3.0])
        calib_scores = np.array([2.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 0) / (1 + 1)
        assert p_values[0] == expected

    def test_single_test_score(self):
        test_scores = np.array([3.0])
        calib_scores = np.array([1.0, 2.0, 4.0, 5.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 1


class TestEqualScores:
    def test_all_scores_equal(self):
        test_scores = np.array([3.0, 3.0, 3.0])
        calib_scores = np.array([3.0, 3.0, 3.0, 3.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 4) / (1 + 4)
        assert np.all(p_values == expected)

    def test_test_equals_all_calib(self):
        test_scores = np.array([2.0])
        calib_scores = np.array([2.0, 2.0, 2.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        expected = (1 + 3) / (1 + 3)
        assert p_values[0] == expected


class TestExtremeWeights:
    def test_zero_test_weight(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        test_weights = np.array([0.0])
        calib_weights = np.array([1.0, 1.0, 1.0])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert p_values[0] >= 0.0
        assert p_values[0] <= 1.0

    def test_zero_calib_weights(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 6.0, 7.0])
        test_weights = np.array([1.0])
        calib_weights = np.array([0.0, 0.0, 0.0])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert len(p_values) == 1

    def test_very_large_weights(self):
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 6.0])
        test_weights = np.array([1e6])
        calib_weights = np.array([1e6, 1e6])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert p_values[0] >= 0.0
        assert p_values[0] <= 1.0

    def test_mixed_zero_and_nonzero_weights(self):
        test_scores = np.array([3.0, 5.0])
        calib_scores = np.array([1.0, 2.0, 4.0, 6.0])
        test_weights = np.array([0.0, 1.0])
        calib_weights = np.array([1.0, 0.0, 1.0, 0.0])

        p_values = calculate_weighted_p_val(
            test_scores, calib_scores, test_weights, calib_weights
        )
        assert len(p_values) == 2
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)


class TestNumericalStability:
    def test_large_arrays(self):
        rng = np.random.default_rng(42)
        test_scores = rng.standard_normal(1000)
        calib_scores = rng.standard_normal(5000)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 1000
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_extreme_score_values(self):
        test_scores = np.array([1e10, -1e10])
        calib_scores = np.array([0.0, 1.0, -1.0])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)

    def test_very_close_scores(self):
        test_scores = np.array([1.0000001])
        calib_scores = np.array([1.0, 1.0000002, 1.0000003])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert p_values[0] >= 0.0
        assert p_values[0] <= 1.0


class TestDataTypes:
    def test_float32_scores(self):
        test_scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        calib_scores = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 3

    def test_float64_scores(self):
        test_scores = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        calib_scores = np.array([0.5, 1.5, 2.5], dtype=np.float64)
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 3

    def test_integer_scores(self):
        test_scores = np.array([1, 2, 3])
        calib_scores = np.array([0, 1, 2, 3, 4])
        p_values = calculate_p_val(test_scores, calib_scores)
        assert len(p_values) == 3
