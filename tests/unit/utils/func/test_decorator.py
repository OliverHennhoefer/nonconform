import numpy as np
import pandas as pd

from nonconform.utils.func.decorator import _ensure_numpy_array


class TestDataFrameConversion:
    def test_dataframe_converts_to_numpy(self, sample_dataframe):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = sample_dataframe()
        result = dummy_method(None, df)
        assert isinstance(result, np.ndarray)

    def test_conversion_preserves_shape(self, sample_dataframe):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = sample_dataframe(n_rows=20, n_cols=5)
        result = dummy_method(None, df)
        assert result.shape == (20, 5)

    def test_conversion_preserves_values(self, sample_dataframe):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = sample_dataframe()
        result = dummy_method(None, df)
        np.testing.assert_array_equal(result, df.to_numpy())

    def test_single_column_dataframe(self, sample_dataframe):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = sample_dataframe(n_cols=1)
        result = dummy_method(None, df)
        assert result.shape == (10, 1)


class TestSeriesConversion:
    def test_series_converts_to_numpy(self, sample_series):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        series = sample_series()
        result = dummy_method(None, series)
        assert isinstance(result, np.ndarray)

    def test_series_preserves_length(self, sample_series):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        series = sample_series(n_rows=25)
        result = dummy_method(None, series)
        assert len(result) == 25

    def test_series_preserves_values(self, sample_series):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        series = sample_series()
        result = dummy_method(None, series)
        np.testing.assert_array_equal(result, series.to_numpy())


class TestPassThrough:
    def test_numpy_array_unchanged(self, sample_array):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        arr = sample_array()
        result = dummy_method(None, arr)
        assert isinstance(result, np.ndarray)
        assert result is arr

    def test_different_shapes_pass_through(self, sample_array):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        arr1d = sample_array(shape=(15,))
        arr2d = sample_array(shape=(10, 5))
        arr3d = sample_array(shape=(5, 4, 3))

        assert dummy_method(None, arr1d).shape == (15,)
        assert dummy_method(None, arr2d).shape == (10, 5)
        assert dummy_method(None, arr3d).shape == (5, 4, 3)


class TestEdgeCases:
    def test_empty_dataframe(self):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = pd.DataFrame()
        result = dummy_method(None, df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)

    def test_empty_series(self):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        series = pd.Series([], dtype=np.float32)
        result = dummy_method(None, series)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_empty_array(self):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        arr = np.array([])
        result = dummy_method(None, arr)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_integer_dtype_dataframe(self):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        df = pd.DataFrame([[1, 2], [3, 4]], dtype=np.int32)
        result = dummy_method(None, df)
        assert result.dtype == np.int32

    def test_float64_dtype_series(self):
        @_ensure_numpy_array
        def dummy_method(self, x):
            return x

        series = pd.Series([1.0, 2.0, 3.0], dtype=np.float64)
        result = dummy_method(None, series)
        assert result.dtype == np.float64


class TestMetadataPreservation:
    def test_function_name_preserved(self):
        @_ensure_numpy_array
        def test_function(self, x):
            return x

        assert test_function.__name__ == "test_function"

    def test_docstring_preserved(self):
        @_ensure_numpy_array
        def documented_function(self, x):
            """This is a test docstring."""
            return x

        assert documented_function.__doc__ == "This is a test docstring."

    def test_decorated_function_callable(self):
        @_ensure_numpy_array
        def test_function(self, x):
            return x

        assert callable(test_function)


class TestAdditionalArguments:
    def test_decorator_with_args(self, sample_dataframe):
        @_ensure_numpy_array
        def method_with_args(self, x, arg1, arg2):
            return x, arg1, arg2

        df = sample_dataframe()
        result_x, result_arg1, result_arg2 = method_with_args(None, df, "test", 42)
        assert isinstance(result_x, np.ndarray)
        assert result_arg1 == "test"
        assert result_arg2 == 42

    def test_decorator_with_kwargs(self, sample_dataframe):
        @_ensure_numpy_array
        def method_with_kwargs(self, x, **kwargs):
            return x, kwargs

        df = sample_dataframe()
        result_x, result_kwargs = method_with_kwargs(None, df, key1="value1", key2=123)
        assert isinstance(result_x, np.ndarray)
        assert result_kwargs == {"key1": "value1", "key2": 123}
