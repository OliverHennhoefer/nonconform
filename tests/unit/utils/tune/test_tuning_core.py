from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.tuning import tune_kde_hyperparameters


class TestBasicOptimization:
    def test_returns_dict(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert isinstance(result, dict)

    def test_has_required_keys(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert "bandwidth" in result
        assert "kernel" in result
        assert "best_score" in result
        assert "study" in result

    def test_bandwidth_is_positive(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert result["bandwidth"] > 0

    def test_kernel_is_enum(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert isinstance(result["kernel"], Kernel)

    def test_best_score_is_number(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5)
        assert isinstance(result["best_score"], int | float)


class TestMultiKernelOptimization:
    def test_multiple_kernels(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="multiple")
        result = tune_kde_hyperparameters(data, kernels, n_trials=10)
        assert result["kernel"] in kernels

    def test_kernel_in_options(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        kernels = [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV]
        result = tune_kde_hyperparameters(data, kernels, n_trials=10)
        assert result["kernel"] in kernels

    def test_single_kernel_enum(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = tune_kde_hyperparameters(data, Kernel.GAUSSIAN, n_trials=5)
        assert result["kernel"] == Kernel.GAUSSIAN


class TestHeuristicMode:
    def test_heuristic_mode_zero_trials(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=0)
        assert result["best_score"] is None
        assert result["study"] is None

    def test_heuristic_mode_negative_trials(
        self, sample_calibration_data, kernel_options
    ):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=-1)
        assert result["best_score"] is None
        assert result["study"] is None

    def test_heuristic_returns_bandwidth(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=0)
        assert result["bandwidth"] > 0


class TestCrossValidation:
    def test_leave_one_out(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=50)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, cv_folds=-1)
        assert result["bandwidth"] > 0

    def test_k_fold(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, cv_folds=5)
        assert result["bandwidth"] > 0


class TestWeightedKDE:
    def test_with_weights(
        self, sample_calibration_data, kernel_options, sample_weights
    ):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        weights = sample_weights(n_samples=100, pattern="uniform")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, weights=weights)
        assert result["bandwidth"] > 0

    def test_without_weights(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, weights=None)
        assert result["bandwidth"] > 0

    def test_random_weights(
        self, sample_calibration_data, kernel_options, sample_weights
    ):
        data = sample_calibration_data(n_samples=100)
        kernels = kernel_options(kernel_type="single")
        weights = sample_weights(n_samples=100, pattern="random")
        result = tune_kde_hyperparameters(data, kernels, n_trials=5, weights=weights)
        assert result["bandwidth"] > 0


class TestSeedReproducibility:
    def test_same_seed_same_results(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100, seed=42)
        kernels = kernel_options(kernel_type="single")

        result1 = tune_kde_hyperparameters(data, kernels, n_trials=10, seed=42)
        result2 = tune_kde_hyperparameters(data, kernels, n_trials=10, seed=42)

        assert result1["bandwidth"] == result2["bandwidth"]
        assert result1["kernel"] == result2["kernel"]

    def test_different_seeds_may_differ(self, sample_calibration_data, kernel_options):
        data = sample_calibration_data(n_samples=100, seed=42)
        kernels = kernel_options(kernel_type="multiple")

        result1 = tune_kde_hyperparameters(data, kernels, n_trials=10, seed=42)
        result2 = tune_kde_hyperparameters(data, kernels, n_trials=10, seed=123)

        assert isinstance(result1["bandwidth"], float)
        assert isinstance(result2["bandwidth"], float)
