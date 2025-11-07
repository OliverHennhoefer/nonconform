import pandas as pd
import pytest

from nonconform.utils.data import Dataset, get_info, list_available, load


class TestDatasetLoading:
    def test_load_returns_dataframe(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_class_column(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert "Class" in df.columns

    def test_dataframe_has_correct_column_names(
        self, temp_manager, mock_urlopen, mock_npz_data
    ):
        npz_data = mock_npz_data(n_samples=50, n_features=7)
        with mock_urlopen(npz_data):
            df = temp_manager.load(Dataset.BREAST, setup=False)
            expected_cols = [f"V{i + 1}" for i in range(7)] + ["Class"]
            assert list(df.columns) == expected_cols

    def test_load_with_enum_value(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert len(df) > 0

    def test_labels_are_binary(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert set(df["Class"].unique()).issubset({0, 1})

    def test_load_different_datasets(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df1 = temp_manager.load(Dataset.BREAST, setup=False)
            df2 = temp_manager.load(Dataset.FRAUD, setup=False)
            assert isinstance(df1, pd.DataFrame)
            assert isinstance(df2, pd.DataFrame)


class TestSetupMode:
    def test_setup_returns_three_elements(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            result = temp_manager.load(Dataset.BREAST, setup=True, seed=42)
            assert len(result) == 3

    def test_setup_returns_correct_types(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert isinstance(x_train, pd.DataFrame)
            assert isinstance(x_test, pd.DataFrame)
            assert isinstance(y_test, pd.Series)

    def test_training_set_has_no_class_column(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert "Class" not in x_train.columns
            assert "Class" not in x_test.columns

    def test_test_labels_are_series(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert len(y_test) == len(x_test)

    def test_setup_with_seed_reproducible(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train1, x_test1, y_test1 = temp_manager.load(
                Dataset.BREAST, setup=True, seed=42
            )
        temp_manager._memory_cache.clear()
        with mock_urlopen():
            x_train2, x_test2, y_test2 = temp_manager.load(
                Dataset.BREAST, setup=True, seed=42
            )

        assert len(x_train1) == len(x_train2)
        assert len(x_test1) == len(x_test2)

    def test_setup_splits_data_correctly(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert len(x_train) > 0
            assert len(x_test) > 0
            assert len(y_test) > 0

    def test_test_set_contains_anomalies(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert 1 in y_test.values

    def test_test_set_contains_normal(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            x_train, x_test, y_test = temp_manager.load(Dataset.BREAST, setup=True)
            assert 0 in y_test.values


class TestMetadataAccess:
    def test_get_info_returns_dataset_info(self, temp_manager):
        info = temp_manager.get_info(Dataset.BREAST)
        assert info.name == "breast"
        assert info.filename == "breast_w.npz"

    def test_get_info_has_required_fields(self, temp_manager):
        info = temp_manager.get_info(Dataset.FRAUD)
        assert hasattr(info, "name")
        assert hasattr(info, "description")
        assert hasattr(info, "filename")
        assert hasattr(info, "samples")
        assert hasattr(info, "features")
        assert hasattr(info, "anomaly_rate")

    def test_list_available_returns_list(self, temp_manager):
        datasets = temp_manager.list_available()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_list_available_is_sorted(self, temp_manager):
        datasets = temp_manager.list_available()
        assert datasets == sorted(datasets)

    def test_list_available_contains_known_datasets(self, temp_manager):
        datasets = temp_manager.list_available()
        assert "breast" in datasets
        assert "fraud" in datasets


class TestPublicAPI:
    def test_load_function_works(self, mock_urlopen):
        with mock_urlopen():
            df = load(Dataset.BREAST, setup=False)
            assert isinstance(df, pd.DataFrame)

    def test_list_available_function(self):
        datasets = list_available()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_get_info_function(self):
        info = get_info(Dataset.BREAST)
        assert info.name == "breast"


class TestDataTypeConversion:
    def test_int_to_float32_conversion(
        self, temp_manager, mock_urlopen, mock_npz_data_int
    ):
        import numpy as np

        int_data = mock_npz_data_int(dtype=np.int32)
        with mock_urlopen(int_data):
            df = temp_manager.load(Dataset.BREAST, setup=False)
            feature_cols = [col for col in df.columns if col != "Class"]
            assert df[feature_cols].dtypes.iloc[0] == np.float32

    def test_various_int_types_converted(self, temp_manager, mock_urlopen):
        import io

        import numpy as np

        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            rng = np.random.default_rng(42)
            x_data = rng.integers(0, 100, size=(50, 3), dtype=dtype)
            y = np.array([0] * 40 + [1] * 10)

            buffer = io.BytesIO()
            np.savez_compressed(buffer, X=x_data, y=y)
            buffer.seek(0)
            int_data = buffer.read()

            with mock_urlopen(int_data):
                df = temp_manager.load(Dataset.BREAST, setup=False)
                feature_cols = [col for col in df.columns if col != "Class"]
                assert df[feature_cols].dtypes.iloc[0] == np.float32


class TestDatasetValidation:
    def test_invalid_dataset_raises_error(self, temp_manager):
        class FakeDataset:
            value = "nonexistent_dataset"

        with pytest.raises(ValueError, match="not found"):
            temp_manager.load(FakeDataset(), setup=False)

    def test_error_message_lists_available_datasets(self, temp_manager):
        class FakeDataset:
            value = "invalid"

        with pytest.raises(ValueError, match="Available datasets"):
            temp_manager.load(FakeDataset(), setup=False)
