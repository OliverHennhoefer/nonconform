import numpy as np
import pytest

from nonconform.utils.func.enums import Kernel


@pytest.fixture
def sample_calibration_data():
    """Create sample calibration data with various distributions."""

    def _create(distribution="normal", n_samples=100, seed=42):
        rng = np.random.default_rng(seed)

        if distribution == "normal":
            return rng.standard_normal(n_samples).astype(np.float64)
        elif distribution == "uniform":
            return rng.uniform(-3, 3, n_samples).astype(np.float64)
        elif distribution == "bimodal":
            mode1 = rng.standard_normal(n_samples // 2) - 2
            mode2 = rng.standard_normal(n_samples // 2) + 2
            return np.concatenate([mode1, mode2]).astype(np.float64)
        elif distribution == "skewed":
            return rng.exponential(2.0, n_samples).astype(np.float64)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    return _create


@pytest.fixture
def sample_weights():
    """Create sample weight arrays for weighted KDE."""

    def _create(n_samples=100, pattern="uniform", seed=42):
        rng = np.random.default_rng(seed)

        if pattern == "uniform":
            return np.ones(n_samples, dtype=np.float64)
        elif pattern == "random":
            return rng.uniform(0.5, 1.5, n_samples).astype(np.float64)
        elif pattern == "sparse":
            weights = rng.uniform(0.0, 1.0, n_samples)
            weights[weights < 0.7] = 0.0
            return weights.astype(np.float64)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    return _create


@pytest.fixture
def kernel_options():
    """Create kernel option lists for testing."""

    def _create(kernel_type="single"):
        if kernel_type == "single":
            return [Kernel.GAUSSIAN]
        elif kernel_type == "multiple":
            return [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV, Kernel.TRIANGULAR]
        elif kernel_type == "all":
            return [
                Kernel.GAUSSIAN,
                Kernel.EXPONENTIAL,
                Kernel.BOX,
                Kernel.TRIANGULAR,
                Kernel.EPANECHNIKOV,
            ]
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    return _create


@pytest.fixture
def constant_data():
    """Create constant/edge case data for testing."""

    def _create(case="constant", n_samples=100):
        if case == "constant":
            return np.ones(n_samples, dtype=np.float64) * 5.0
        elif case == "two_values":
            return np.array([1.0, 2.0], dtype=np.float64)
        elif case == "single":
            return np.array([3.14], dtype=np.float64)
        elif case == "zeros":
            return np.zeros(n_samples, dtype=np.float64)
        else:
            raise ValueError(f"Unknown case: {case}")

    return _create


@pytest.fixture
def extreme_data():
    """Create data with extreme values for testing."""

    def _create(case="large", n_samples=50, seed=42):
        rng = np.random.default_rng(seed)

        if case == "large":
            return rng.standard_normal(n_samples) * 1e10
        elif case == "small":
            return rng.standard_normal(n_samples) * 1e-10
        elif case == "mixed":
            data = rng.standard_normal(n_samples)
            data[: n_samples // 2] *= 1e10
            data[n_samples // 2 :] *= 1e-10
            return data
        else:
            raise ValueError(f"Unknown case: {case}")

    return _create
