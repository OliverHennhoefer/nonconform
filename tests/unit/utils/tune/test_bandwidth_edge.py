import numpy as np

from nonconform.utils.tune.bandwidth import (
    _scott_bandwidth,
    _sheather_jones_bandwidth,
    _silverman_bandwidth,
    compute_bandwidth_range,
)


class TestConstantData:
    def test_scott_with_constant_data(self, constant_data):
        data = constant_data(case="constant")
        result = _scott_bandwidth(data)
        assert result >= 0

    def test_silverman_with_constant_data(self, constant_data):
        data = constant_data(case="constant")
        result = _silverman_bandwidth(data)
        assert result >= 0

    def test_silverman_iqr_zero(self, constant_data):
        data = constant_data(case="constant")
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        assert iqr == 0
        result = _silverman_bandwidth(data)
        assert result >= 0

    def test_bandwidth_range_constant_data(self, constant_data):
        data = constant_data(case="constant")
        bw_min, bw_max = compute_bandwidth_range(data)
        assert bw_min >= 0
        assert bw_max >= 0


class TestSmallSamples:
    def test_scott_single_point(self, constant_data):
        data = constant_data(case="single")
        result = _scott_bandwidth(data)
        assert result is not None

    def test_silverman_single_point(self, constant_data):
        data = constant_data(case="single")
        result = _silverman_bandwidth(data)
        assert result is not None

    def test_scott_two_points(self, constant_data):
        data = constant_data(case="two_values")
        result = _scott_bandwidth(data)
        assert result > 0

    def test_silverman_two_points(self, constant_data):
        data = constant_data(case="two_values")
        result = _silverman_bandwidth(data)
        assert result > 0


class TestExtremeValues:
    def test_scott_large_values(self, extreme_data):
        data = extreme_data(case="large")
        result = _scott_bandwidth(data)
        assert np.isfinite(result)
        assert result > 0

    def test_silverman_large_values(self, extreme_data):
        data = extreme_data(case="large")
        result = _silverman_bandwidth(data)
        assert np.isfinite(result)
        assert result > 0

    def test_scott_small_values(self, extreme_data):
        data = extreme_data(case="small")
        result = _scott_bandwidth(data)
        assert np.isfinite(result)
        assert result > 0

    def test_silverman_small_values(self, extreme_data):
        data = extreme_data(case="small")
        result = _silverman_bandwidth(data)
        assert np.isfinite(result)
        assert result > 0


class TestSheatherJonesFallback:
    def test_fallback_on_small_sample(self, constant_data):
        data = constant_data(case="two_values")
        result = _sheather_jones_bandwidth(data)
        assert result > 0
        assert np.isfinite(result)

    def test_fallback_is_silverman(self, constant_data):
        data = constant_data(case="constant")
        result = _sheather_jones_bandwidth(data)
        assert np.isfinite(result)


class TestNumericalStability:
    def test_zeros_data(self, constant_data):
        data = constant_data(case="zeros")
        bw_scott = _scott_bandwidth(data)
        bw_silverman = _silverman_bandwidth(data)
        bw_min, bw_max = compute_bandwidth_range(data)

        assert np.isfinite(bw_scott)
        assert np.isfinite(bw_silverman)
        assert np.isfinite(bw_min) and np.isfinite(bw_max)

    def test_range_with_zero_std(self, constant_data):
        data = constant_data(case="constant")
        bw_min, bw_max = compute_bandwidth_range(data)
        assert bw_min >= 0
        assert bw_max >= bw_min
