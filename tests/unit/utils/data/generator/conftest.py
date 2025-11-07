import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_dataset():
    def _make(n_normal=300, n_anomaly=100, n_features=5, seed=42):
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


@pytest.fixture
def tiny_dataset():
    def _make(n_normal=10, n_anomaly=5, n_features=3, seed=42):
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


@pytest.fixture
def large_dataset():
    def _make(n_normal=2000, n_anomaly=500, n_features=10, seed=42):
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


@pytest.fixture
def imbalanced_dataset():
    def _make(n_normal=990, n_anomaly=10, n_features=5, seed=42):
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


def assert_batch_valid(x_batch, y_batch, expected_size):
    assert len(x_batch) == expected_size
    assert len(y_batch) == expected_size
    assert isinstance(x_batch, pd.DataFrame)
    assert isinstance(y_batch, pd.Series)
    assert y_batch.dtype == int
    assert set(y_batch.unique()).issubset({0, 1})


def assert_exact_proportion(batches, expected_total_anomalies):
    total_anomalies = sum(y_batch.sum() for _, y_batch in batches)
    assert total_anomalies == expected_total_anomalies
