import numpy as np
import pytest

from nonconform.metrics import aggregate


class TestMeanAggregation:
    def test_mean_aggregation(self, sample_2d_scores):
        scores = sample_2d_scores()
        result = aggregate("mean", scores)
        expected = np.mean(scores, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mean_aggregation_shape(self, sample_2d_scores):
        scores = sample_2d_scores(n_models=5, n_samples=10)
        result = aggregate("mean", scores)
        assert result.shape == (10,)

    def test_mean_with_different_values(self):
        scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = aggregate("mean", scores)
        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestMedianAggregation:
    def test_median_aggregation(self, sample_2d_scores):
        scores = sample_2d_scores()
        result = aggregate("median", scores)
        expected = np.median(scores, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_median_aggregation_shape(self, sample_2d_scores):
        scores = sample_2d_scores(n_models=5, n_samples=10)
        result = aggregate("median", scores)
        assert result.shape == (10,)

    def test_median_with_odd_number_of_models(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = aggregate("median", scores)
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestMinimumAggregation:
    def test_minimum_aggregation(self, sample_2d_scores):
        scores = sample_2d_scores()
        result = aggregate("minimum", scores)
        expected = np.min(scores, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_minimum_aggregation_shape(self, sample_2d_scores):
        scores = sample_2d_scores(n_models=5, n_samples=10)
        result = aggregate("minimum", scores)
        assert result.shape == (10,)

    def test_minimum_with_known_values(self):
        scores = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        result = aggregate("minimum", scores)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestMaximumAggregation:
    def test_maximum_aggregation(self, sample_2d_scores):
        scores = sample_2d_scores()
        result = aggregate("maximum", scores)
        expected = np.max(scores, axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_maximum_aggregation_shape(self, sample_2d_scores):
        scores = sample_2d_scores(n_models=5, n_samples=10)
        result = aggregate("maximum", scores)
        assert result.shape == (10,)

    def test_maximum_with_known_values(self):
        scores = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        result = aggregate("maximum", scores)
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestEdgeCases:
    def test_single_row_input(self):
        scores = np.array([[1.0, 2.0, 3.0]])
        result = aggregate("mean", scores)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_column_input(self):
        scores = np.array([[1.0], [2.0], [3.0]])
        result = aggregate("mean", scores)
        assert result.shape == (1,)
        assert result[0] == 2.0

    def test_all_same_values(self):
        scores = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        result = aggregate("mean", scores)
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_integer_array_conversion(self):
        scores = np.array([[1, 2, 3], [4, 5, 6]])
        result = aggregate("mean", scores)
        assert isinstance(result[0], np.floating | float)

    def test_invalid_method_raises(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Unsupported aggregation method"):
            aggregate("invalid", scores)

    def test_non_string_method_raises(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(TypeError, match="aggregation method must be a string"):
            aggregate(1, scores)  # type: ignore[arg-type]


class TestDataTypes:
    def test_float32_input(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = aggregate("mean", scores)
        assert result.dtype == np.float32

    def test_float64_input(self):
        scores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        result = aggregate("mean", scores)
        assert result.dtype == np.float64
