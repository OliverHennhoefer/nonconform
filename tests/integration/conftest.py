"""Shared fixtures for integration tests."""

import numpy as np
import pytest

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF


@pytest.fixture
def simple_dataset():
    """Create a simple synthetic dataset for basic integration testing."""

    def _create(
        n_train=200,
        n_test=100,
        n_features=5,
        anomaly_rate_test=0.1,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        # Training data: all normal
        x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)

        # Test data: normal + anomalies
        n_anomalies = int(n_test * anomaly_rate_test)
        n_normal = n_test - n_anomalies

        x_test_normal = rng.standard_normal((n_normal, n_features)).astype(np.float32)
        x_test_anomaly = (rng.standard_normal((n_anomalies, n_features)) + 3).astype(
            np.float32
        )

        x_test = np.vstack([x_test_normal, x_test_anomaly])
        y_test = np.array([0] * n_normal + [1] * n_anomalies)

        # Shuffle test data
        shuffle_idx = rng.permutation(n_test)
        x_test = x_test[shuffle_idx]
        y_test = y_test[shuffle_idx]

        return x_train, x_test, y_test

    return _create


@pytest.fixture
def shifted_dataset():
    """Create dataset with covariate shift between train and test.

    Returns:
        Factory function that generates (X_train, X_test, y_test) tuple
        where test distribution is shifted.
    """

    def _create(
        n_train=200,
        n_test=100,
        n_features=5,
        shift_mean=1.0,
        shift_scale=0.5,
        anomaly_rate_test=0.1,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        # Training data: standard normal
        x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)

        # Test data: shifted normal + anomalies
        n_anomalies = int(n_test * anomaly_rate_test)
        n_normal = n_test - n_anomalies

        # Normal test data with covariate shift
        x_test_normal = (
            rng.standard_normal((n_normal, n_features)) * shift_scale + shift_mean
        ).astype(np.float32)

        # Anomalies (even more shifted)
        x_test_anomaly = (
            rng.standard_normal((n_anomalies, n_features)) + shift_mean + 3
        ).astype(np.float32)

        x_test = np.vstack([x_test_normal, x_test_anomaly])
        y_test = np.array([0] * n_normal + [1] * n_anomalies)

        # Shuffle test data
        shuffle_idx = rng.permutation(n_test)
        x_test = x_test[shuffle_idx]
        y_test = y_test[shuffle_idx]

        return x_train, x_test, y_test

    return _create


@pytest.fixture
def multimodal_dataset():
    """Create dataset with multiple clusters for complex scenarios.

    Returns:
        Factory function that generates (X_train, X_test, y_test) tuple
        with multimodal normal distribution.
    """

    def _create(
        n_train=200,
        n_test=100,
        n_features=5,
        n_clusters=3,
        anomaly_rate_test=0.1,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        # Training data: mixture of Gaussians
        cluster_size_train = n_train // n_clusters
        x_train_list = []
        for _ in range(n_clusters):
            center = rng.standard_normal(n_features) * 3
            cluster = rng.standard_normal((cluster_size_train, n_features)) + center
            x_train_list.append(cluster)
        x_train = np.vstack(x_train_list).astype(np.float32)

        # Test data: similar mixture + anomalies
        n_anomalies = int(n_test * anomaly_rate_test)
        n_normal = n_test - n_anomalies

        cluster_size_test = n_normal // n_clusters
        x_test_normal_list = []
        for _ in range(n_clusters):
            center = rng.standard_normal(n_features) * 3
            cluster = rng.standard_normal((cluster_size_test, n_features)) + center
            x_test_normal_list.append(cluster)
        x_test_normal = np.vstack(x_test_normal_list).astype(np.float32)

        # Anomalies: far from all clusters
        x_test_anomaly = (
            rng.standard_normal((n_anomalies, n_features)) * 0.1 + 10
        ).astype(np.float32)

        x_test = np.vstack([x_test_normal, x_test_anomaly])
        y_test = np.array([0] * n_normal + [1] * n_anomalies)

        # Shuffle test data
        shuffle_idx = rng.permutation(n_test)
        x_test = x_test[shuffle_idx]
        y_test = y_test[shuffle_idx]

        return x_train, x_test, y_test

    return _create


@pytest.fixture
def pyod_detectors():
    """Provide common PyOD detector instances.

    Returns:
        Dict of detector name -> detector instance.
    """

    def _create(seed=42):
        return {
            "iforest": IForest(n_estimators=50, random_state=seed),
            "knn": KNN(n_neighbors=5),
            "lof": LOF(n_neighbors=5),
            "ecod": ECOD(),
        }

    return _create


@pytest.fixture
def pyod_detector():
    """Create default PyOD detector for testing."""
    return IForest(n_estimators=50, random_state=42)
