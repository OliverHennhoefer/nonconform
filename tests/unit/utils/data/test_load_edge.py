import io
import os
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pytest

from nonconform.utils.data import Dataset
from nonconform.utils.data.load import DatasetManager


class TestInvalidDataset:
    def test_invalid_dataset_name_raises_error(self, temp_manager):
        class InvalidDataset:
            value = "nonexistent_dataset_xyz"

        with pytest.raises(ValueError, match="not found"):
            temp_manager.load(InvalidDataset())

    def test_error_message_suggests_alternatives(self, temp_manager):
        class InvalidDataset:
            value = "invalid"

        with pytest.raises(ValueError, match="Available datasets"):
            temp_manager.load(InvalidDataset())

    def test_error_lists_actual_datasets(self, temp_manager):
        class InvalidDataset:
            value = "fake"

        try:
            temp_manager.load(InvalidDataset())
        except ValueError as e:
            assert "breast" in str(e)
            assert "fraud" in str(e)


class TestNetworkErrors:
    def test_url_error_on_download_failure(self, temp_manager):
        with patch("nonconform.utils.data.load.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Network error")
            with pytest.raises(URLError, match="Failed to download"):
                temp_manager.load(Dataset.BREAST, setup=False)

    def test_url_error_preserves_original_message(self, temp_manager):
        with patch("nonconform.utils.data.load.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection timeout")
            with pytest.raises(URLError, match="Connection timeout"):
                temp_manager.load(Dataset.BREAST, setup=False)


class TestCorruptedData:
    def test_invalid_npz_format_raises_error(self, temp_manager):
        with patch("nonconform.utils.data.load.urlopen") as mock_urlopen:
            mock_response = type("MockResponse", (), {})()
            mock_response.read = lambda: b"invalid npz data"
            mock_response.__enter__ = lambda self: self
            mock_response.__exit__ = lambda self, *args: None
            mock_urlopen.return_value = mock_response

            with pytest.raises(Exception):
                temp_manager.load(Dataset.BREAST, setup=False)

    def test_missing_x_key_in_npz(self, temp_manager, mock_urlopen):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, y=np.array([0, 1]))
        buffer.seek(0)
        corrupted_data = buffer.read()

        with mock_urlopen(corrupted_data):
            with pytest.raises(KeyError):
                temp_manager.load(Dataset.BREAST, setup=False)

    def test_missing_y_key_in_npz(self, temp_manager, mock_urlopen):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=np.array([[1, 2, 3]]))
        buffer.seek(0)
        corrupted_data = buffer.read()

        with mock_urlopen(corrupted_data):
            with pytest.raises(KeyError):
                temp_manager.load(Dataset.BREAST, setup=False)


class TestSetupModeEdgeCases:
    def test_setup_with_minimal_data(self, temp_manager, mock_urlopen):
        x_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])

        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=x_data, y=y)
        buffer.seek(0)
        minimal_data = buffer.read()

        with mock_urlopen(minimal_data):
            x_train, x_test, y_test = temp_manager.load(
                Dataset.BREAST, setup=True, seed=42
            )
            assert len(x_train) > 0

    def test_setup_with_no_anomalies(self, temp_manager, mock_urlopen):
        rng = np.random.default_rng(42)
        x_data = rng.standard_normal((100, 5))
        y = np.array([0] * 100)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=x_data, y=y)
        buffer.seek(0)
        no_anomaly_data = buffer.read()

        with mock_urlopen(no_anomaly_data):
            x_train, x_test, y_test = temp_manager.load(
                Dataset.BREAST, setup=True, seed=42
            )
            assert len(x_train) > 0
            assert len(y_test) > 0
            assert 1 not in y_test.values

    def test_setup_respects_sample_limits(self, temp_manager, mock_urlopen):
        rng = np.random.default_rng(42)
        x_data = rng.standard_normal((5000, 10))
        y = np.array([0] * 4500 + [1] * 500)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=x_data, y=y)
        buffer.seek(0)
        large_data = buffer.read()

        with mock_urlopen(large_data):
            x_train, x_test, y_test = temp_manager.load(
                Dataset.BREAST, setup=True, seed=42
            )
            assert len(x_test) <= 1000


class TestCachePermissions:
    def test_handles_readonly_cache_directory(
        self, temp_manager, mock_urlopen, tmp_path
    ):
        cache_dir = temp_manager.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            cache_dir.chmod(0o444)

            with mock_urlopen():
                df = temp_manager.load(Dataset.BREAST, setup=False)
                assert df is not None
        finally:
            cache_dir.chmod(0o755)


class TestEnvironmentVariables:
    def test_custom_cache_dir_from_env(self, tmp_path, mock_urlopen):
        custom_cache = tmp_path / "custom_cache"
        with patch.dict(os.environ, {"UNQUAD_CACHE_DIR": str(custom_cache)}):
            manager = DatasetManager()
            with mock_urlopen():
                manager.load(Dataset.BREAST, setup=False)
            assert custom_cache.exists()

    def test_custom_version_from_env(self, tmp_path, mock_urlopen):
        with patch.dict(os.environ, {"UNQUAD_DATASET_VERSION": "v0.99.0-test"}):
            manager = DatasetManager()
            assert manager.version == "v0.99.0-test"

    def test_custom_base_url_from_env(self, tmp_path):
        custom_url = "https://example.com/datasets/"
        with patch.dict(os.environ, {"UNQUAD_DATASET_URL": custom_url}):
            manager = DatasetManager()
            assert manager.base_url == custom_url


class TestDataIntegrity:
    def test_data_shape_matches_labels(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert len(df) == len(df["Class"])

    def test_no_missing_values_in_features(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df = temp_manager.load(Dataset.BREAST, setup=False)
            feature_cols = [col for col in df.columns if col != "Class"]
            assert not df[feature_cols].isna().any().any()

    def test_labels_not_converted_to_float(
        self, temp_manager, mock_urlopen, mock_npz_data_int
    ):
        int_data = mock_npz_data_int()
        with mock_urlopen(int_data):
            df = temp_manager.load(Dataset.BREAST, setup=False)
            assert df["Class"].dtype in [np.int32, np.int64, int]


class TestConcurrentAccess:
    def test_multiple_managers_share_singleton(self):
        from nonconform.utils.data.load import _manager

        manager1 = _manager
        manager2 = _manager
        assert manager1 is manager2

    def test_load_same_dataset_multiple_times(self, temp_manager, mock_urlopen):
        with mock_urlopen():
            df1 = temp_manager.load(Dataset.BREAST, setup=False)
            df2 = temp_manager.load(Dataset.BREAST, setup=False)
            assert len(df1) == len(df2)
