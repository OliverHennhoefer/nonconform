import numpy as np

from nonconform.utils.stat.results import ConformalResult


class TestInstantiation:
    def test_default_instantiation(self):
        result = ConformalResult()
        assert result.p_values is None
        assert result.test_scores is None
        assert result.calib_scores is None
        assert result.test_weights is None
        assert result.calib_weights is None
        assert result.metadata == {}

    def test_with_p_values(self):
        p_vals = np.array([0.1, 0.2, 0.3])
        result = ConformalResult(p_values=p_vals)
        np.testing.assert_array_equal(result.p_values, p_vals)

    def test_with_scores(self):
        test_scores = np.array([1.0, 2.0, 3.0])
        calib_scores = np.array([0.5, 1.5, 2.5])
        result = ConformalResult(test_scores=test_scores, calib_scores=calib_scores)
        np.testing.assert_array_equal(result.test_scores, test_scores)
        np.testing.assert_array_equal(result.calib_scores, calib_scores)

    def test_with_weights(self):
        test_weights = np.array([1.0, 1.0, 1.0])
        calib_weights = np.array([0.8, 1.2, 1.0])
        result = ConformalResult(test_weights=test_weights, calib_weights=calib_weights)
        np.testing.assert_array_equal(result.test_weights, test_weights)
        np.testing.assert_array_equal(result.calib_weights, calib_weights)

    def test_with_metadata(self):
        metadata = {"key": "value", "number": 42}
        result = ConformalResult(metadata=metadata)
        assert result.metadata == metadata


class TestCopyFunctionality:
    def test_copy_creates_new_instance(self):
        result = ConformalResult()
        copied = result.copy()
        assert result is not copied

    def test_copy_with_none_values(self):
        result = ConformalResult()
        copied = result.copy()
        assert copied.p_values is None
        assert copied.test_scores is None

    def test_copy_arrays_are_independent(self):
        p_vals = np.array([0.1, 0.2, 0.3])
        result = ConformalResult(p_values=p_vals)
        copied = result.copy()

        copied.p_values[0] = 0.9
        assert result.p_values[0] == 0.1

    def test_copy_all_attributes(self, conformal_result):
        result = conformal_result(include_p_values=True, include_weights=True)
        copied = result.copy()

        np.testing.assert_array_equal(copied.p_values, result.p_values)
        np.testing.assert_array_equal(copied.test_scores, result.test_scores)
        np.testing.assert_array_equal(copied.calib_scores, result.calib_scores)
        np.testing.assert_array_equal(copied.test_weights, result.test_weights)
        np.testing.assert_array_equal(copied.calib_weights, result.calib_weights)

    def test_metadata_shallow_copy(self):
        metadata = {"key": [1, 2, 3]}
        result = ConformalResult(metadata=metadata)
        copied = result.copy()

        copied.metadata["key"].append(4)
        assert len(result.metadata["key"]) == 4


class TestEdgeCases:
    def test_empty_arrays(self):
        result = ConformalResult(
            p_values=np.array([]),
            test_scores=np.array([]),
            calib_scores=np.array([]),
        )
        assert len(result.p_values) == 0
        assert len(result.test_scores) == 0
        assert len(result.calib_scores) == 0

    def test_empty_metadata(self):
        result = ConformalResult(metadata={})
        assert result.metadata == {}

    def test_large_arrays(self):
        rng = np.random.default_rng(42)
        large_array = rng.standard_normal(10000)
        result = ConformalResult(test_scores=large_array)
        assert len(result.test_scores) == 10000
