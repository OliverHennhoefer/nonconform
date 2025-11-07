import numpy as np

from nonconform.utils.tune.tuning import (
    _collect_heuristic_bandwidths,
    tune_kde_hyperparameters,
)


class TestHeuristicCollection:
    def test_collects_bandwidths(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert len(heuristics) > 0

    def test_all_positive(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(bw > 0 for bw in heuristics)

    def test_all_finite(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(np.isfinite(bw) for bw in heuristics)


class TestBandwidthDeduplication:
    def test_removes_duplicates(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.001, 0.002
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert len(heuristics) == len(set(heuristics))

    def test_close_values_removed(self):
        data = np.array([1.0, 1.0, 1.0, 2.0])
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert len(heuristics) >= 1


class TestBandwidthClipping:
    def test_clips_to_min(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 100.0, 200.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(bw >= bw_min for bw in heuristics)

    def test_clips_to_max(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.0001, 0.001
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(bw <= bw_max for bw in heuristics)

    def test_within_range(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(bw_min <= bw <= bw_max for bw in heuristics)


class TestWarmupTrials:
    def test_study_has_trials(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=10)
        assert result["study"] is not None
        assert len(result["study"].trials) > 0

    def test_warmup_enqueued(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=10)
        assert result["study"].trials[0] is not None


class TestFilteringInvalid:
    def test_filters_invalid_bandwidths(self, constant_data):
        data = constant_data(case="zeros")
        bw_min, bw_max = 0.01, 10.0
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert all(np.isfinite(bw) for bw in heuristics)
        assert all(bw > 0 for bw in heuristics)

    def test_fallback_when_all_invalid(self):
        data = np.array([1.0, 1.0])
        bw_min, bw_max = 1e10, 1e11
        heuristics = _collect_heuristic_bandwidths(data, bw_min, bw_max)
        assert len(heuristics) == 1
        assert heuristics[0] >= bw_min
