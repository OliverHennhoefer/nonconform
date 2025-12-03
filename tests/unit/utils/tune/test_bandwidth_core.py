import numpy as np

from nonconform.utils.tune.bandwidth import (
    IQR_TO_STD_FACTOR,
    SCOTT_CONSTANT,
    SILVERMAN_CONSTANT,
    _scott_bandwidth,
    _sheather_jones_bandwidth,
    _silverman_bandwidth,
    compute_bandwidth_range,
)


class TestScottBandwidth:
    def test_scott_formula(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(data)
        sigma = np.std(data, ddof=1)
        expected = SCOTT_CONSTANT * sigma * n ** (-0.2)
        result = _scott_bandwidth(data)
        assert np.isclose(result, expected)

    def test_scott_returns_positive(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _scott_bandwidth(data)
        assert result > 0

    def test_scott_scales_with_std(self):
        rng = np.random.default_rng(42)
        data1 = rng.standard_normal(100)
        data2 = rng.standard_normal(100) * 10
        bw1 = _scott_bandwidth(data1)
        bw2 = _scott_bandwidth(data2)
        assert bw2 > bw1


class TestSilvermanBandwidth:
    def test_silverman_formula(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(data)
        sigma = np.std(data, ddof=1)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        sigma_hat = min(sigma, iqr / IQR_TO_STD_FACTOR)
        expected = SILVERMAN_CONSTANT * sigma_hat * n ** (-0.2)
        result = _silverman_bandwidth(data)
        assert np.isclose(result, expected)

    def test_silverman_returns_positive(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _silverman_bandwidth(data)
        assert result > 0

    def test_silverman_uses_robust_estimator(self):
        data = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        result = _silverman_bandwidth(data)
        assert result > 0


class TestSheatherJonesBandwidth:
    def test_sheather_jones_returns_positive(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _sheather_jones_bandwidth(data)
        assert result > 0

    def test_sheather_jones_fallback(self):
        data = np.array([1.0, 2.0])
        result = _sheather_jones_bandwidth(data)
        assert result > 0


class TestBandwidthRange:
    def test_returns_tuple(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = compute_bandwidth_range(data)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_min_less_than_max(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = compute_bandwidth_range(data)
        assert bw_min < bw_max

    def test_both_positive(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = compute_bandwidth_range(data)
        assert bw_min > 0
        assert bw_max > 0

    def test_range_computation(self):
        data = np.array([0.0, 10.0])
        bw_min, bw_max = compute_bandwidth_range(data)

        # Current implementation uses robust percentile-based statistics
        q1, q99 = np.percentile(data, [1, 99])
        robust_range = q99 - q1
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        robust_std = iqr / IQR_TO_STD_FACTOR if iqr > 0 else float(np.std(data))

        expected_min = min(robust_range * 0.001, robust_std * 0.01)
        expected_max = max(robust_range * 0.5, robust_std * 2)

        assert np.isclose(bw_min, expected_min)
        assert np.isclose(bw_max, expected_max)


class TestMathematicalCorrectness:
    def test_scott_decreases_with_sample_size(self):
        rng = np.random.default_rng(42)
        data_small = rng.standard_normal(10)
        data_large = rng.standard_normal(1000)
        bw_small = _scott_bandwidth(data_small)
        bw_large = _scott_bandwidth(data_large)
        assert bw_small > bw_large

    def test_silverman_decreases_with_sample_size(self):
        rng = np.random.default_rng(123)
        data_small = rng.standard_normal(10)
        data_large = rng.standard_normal(1000)
        bw_small = _silverman_bandwidth(data_small)
        bw_large = _silverman_bandwidth(data_large)
        assert bw_small > bw_large

    def test_bandwidth_range_scales(self):
        data1 = np.array([0.0, 1.0])
        data2 = np.array([0.0, 100.0])
        _, max1 = compute_bandwidth_range(data1)
        _, max2 = compute_bandwidth_range(data2)
        assert max2 > max1
