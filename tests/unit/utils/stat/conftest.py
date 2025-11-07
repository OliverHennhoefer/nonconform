import numpy as np
import pytest

from nonconform.utils.stat.results import ConformalResult


@pytest.fixture
def sample_scores():
    """Create sample test and calibration scores."""

    def _create(n_test=20, n_calib=100, seed=42):
        rng = np.random.default_rng(seed)
        test_scores = rng.standard_normal(n_test).astype(np.float32)
        calib_scores = rng.standard_normal(n_calib).astype(np.float32)
        return test_scores, calib_scores

    return _create


@pytest.fixture
def sample_weights():
    """Create sample importance weights."""

    def _create(n_test=20, n_calib=100, seed=42):
        rng = np.random.default_rng(seed)
        test_weights = rng.uniform(0.5, 1.5, n_test).astype(np.float32)
        calib_weights = rng.uniform(0.5, 1.5, n_calib).astype(np.float32)
        return test_weights, calib_weights

    return _create


@pytest.fixture
def sample_p_values():
    """Create sample p-values."""

    def _create(n=20, seed=42):
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, n).astype(np.float32)

    return _create


@pytest.fixture
def sample_binary_labels():
    """Create sample binary labels for metrics testing."""

    def _create(n=100, anomaly_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        n_anomalies = int(n * anomaly_rate)
        labels = np.array([0] * (n - n_anomalies) + [1] * n_anomalies)
        rng.shuffle(labels)
        return labels

    return _create


@pytest.fixture
def conformal_result():
    """Create ConformalResult instances for testing."""

    def _create(
        n_test=20,
        n_calib=100,
        include_p_values=True,
        include_weights=True,
        include_metadata=False,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        test_scores = rng.standard_normal(n_test).astype(np.float32)
        calib_scores = rng.standard_normal(n_calib).astype(np.float32)

        p_values = None
        if include_p_values:
            p_values = rng.uniform(0.0, 1.0, n_test).astype(np.float32)

        test_weights = None
        calib_weights = None
        if include_weights:
            test_weights = rng.uniform(0.5, 1.5, n_test).astype(np.float32)
            calib_weights = rng.uniform(0.5, 1.5, n_calib).astype(np.float32)

        metadata = {}
        if include_metadata:
            metadata = {"kde_support": np.array([0.0, 1.0, 2.0])}

        return ConformalResult(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            metadata=metadata,
        )

    return _create


@pytest.fixture
def sample_2d_scores():
    """Create 2D scores for aggregation testing."""

    def _create(n_models=5, n_samples=10, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_models, n_samples)).astype(np.float32)

    return _create
