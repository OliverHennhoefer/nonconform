import numpy as np

from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.tuning import tune_kde_hyperparameters


class TestEmptyKernelList:
    def test_empty_list_defaults_to_gaussian(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = tune_kde_hyperparameters(data, [], n_trials=5)
        assert result["kernel"] == Kernel.GAUSSIAN

    def test_empty_list_works(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = tune_kde_hyperparameters(data, [], n_trials=5)
        assert result["bandwidth"] > 0


class TestSingleCalibrationPoint:
    def test_single_point(self, constant_data, kernel_options):
        data = constant_data(case="two_values")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, cv_folds=2)
        assert result["bandwidth"] > 0

    def test_single_point_heuristic_mode(self, constant_data, kernel_options):
        data = constant_data(case="single")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=0)
        assert result["bandwidth"] > 0


class TestIdenticalScores:
    def test_constant_data(self, constant_data, kernel_options):
        data = constant_data(case="constant")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] >= 0

    def test_constant_heuristic_mode(self, constant_data, kernel_options):
        data = constant_data(case="constant")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=0)
        assert result["bandwidth"] >= 0


class TestExtremeBandwidthRanges:
    def test_very_small_data_range(self, constant_data, kernel_options):
        data = constant_data(case="two_values")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] > 0

    def test_data_sorting(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100, seed=42)
        unsorted_data = data.copy()
        rng = np.random.default_rng(42)
        rng.shuffle(unsorted_data)

        kernels = kernel_options(kernel_type="single")
        result1 = tune_kde_hyperparameters(data, kernels, n_trials=5, seed=42)
        result2 = tune_kde_hyperparameters(unsorted_data, kernels, n_trials=5, seed=42)

        assert result1["bandwidth"] == result2["bandwidth"]


class TestDataWeightCorrespondence:
    def test_maintains_correspondence(
        self, sample_calibration_data, sample_weights, kernel_options
    ):
        data = sample_calibration_data(n_samples=100, seed=42)
        weights = sample_weights(n_samples=100, pattern="random", seed=42)
        unsorted_data = data.copy()
        unsorted_weights = weights.copy()

        rng = np.random.default_rng(42)
        idx = rng.permutation(100)
        unsorted_data = unsorted_data[idx]
        unsorted_weights = unsorted_weights[idx]

        kernels = kernel_options(kernel_type="single")
        result1 = tune_kde_hyperparameters(
            data, kernels, n_trials=5, weights=weights, seed=42
        )
        result2 = tune_kde_hyperparameters(
            unsorted_data, kernels, n_trials=5, weights=unsorted_weights, seed=42
        )

        assert np.isclose(result1["bandwidth"], result2["bandwidth"], rtol=0.1)


class TestExtremeValues:
    def test_large_values(self, extreme_data, kernel_options):
        data = extreme_data(case="large")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert np.isfinite(result["bandwidth"])
        assert result["bandwidth"] > 0

    def test_small_values(self, extreme_data, kernel_options):
        data = extreme_data(case="small")
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert np.isfinite(result["bandwidth"])
        assert result["bandwidth"] > 0


class TestDifferentDistributions:
    def test_uniform_distribution(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(distribution="uniform", n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] > 0

    def test_bimodal_distribution(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(distribution="bimodal", n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] > 0

    def test_skewed_distribution(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(distribution="skewed", n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] > 0
