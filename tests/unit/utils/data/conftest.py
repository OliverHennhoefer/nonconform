import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from nonconform.utils.data.load import DatasetManager


@pytest.fixture
def mock_npz_data():
    def _create(n_samples=100, n_features=5, anomaly_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        n_anomalies = int(n_samples * anomaly_rate)
        n_normal = n_samples - n_anomalies

        x_data = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = np.array([0] * n_normal + [1] * n_anomalies)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=x_data, y=y)
        buffer.seek(0)
        return buffer.read()

    return _create


@pytest.fixture
def mock_npz_data_int():
    def _create(n_samples=50, n_features=3, dtype=np.int32):
        rng = np.random.default_rng(42)
        x_data = rng.integers(0, 100, size=(n_samples, n_features), dtype=dtype)
        y = np.array([0] * 40 + [1] * 10)

        buffer = io.BytesIO()
        np.savez_compressed(buffer, X=x_data, y=y)
        buffer.seek(0)
        return buffer.read()

    return _create


@pytest.fixture
def temp_manager(tmp_path):
    manager = DatasetManager()
    cache_root = tmp_path / "cache"
    manager._cache_dir = cache_root / manager.version
    manager._cache_dir.mkdir(parents=True, exist_ok=True)
    manager._memory_cache.clear()
    return manager


@pytest.fixture
def mock_urlopen(mock_npz_data):
    def _mock(return_data=None):
        if return_data is None:
            return_data = mock_npz_data()

        mock_response = MagicMock()
        mock_response.read.return_value = return_data
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        return patch("nonconform.utils.data.load.urlopen", return_value=mock_response)

    return _mock


@pytest.fixture
def synthetic_dataset():
    def _make(n_normal=200, n_anomaly=50, n_features=5, seed=42):
        rng = np.random.default_rng(seed)

        normal = pd.DataFrame(
            rng.standard_normal((n_normal, n_features)),
            columns=[f"V{i + 1}" for i in range(n_features)],
        )
        normal["Class"] = 0

        anomaly = pd.DataFrame(
            rng.standard_normal((n_anomaly, n_features)) + 3,
            columns=[f"V{i + 1}" for i in range(n_features)],
        )
        anomaly["Class"] = 1

        return pd.concat([normal, anomaly], ignore_index=True)

    return _make
