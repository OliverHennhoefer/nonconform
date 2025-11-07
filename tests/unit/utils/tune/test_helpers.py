import numpy as np

from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.tuning import (
    _compute_cv_log_likelihood,
    _fit_kde,
    _normalise_kernels,
)


class TestNormaliseKernels:
    def test_single_kernel_enum(self):
        result = _normalise_kernels(Kernel.GAUSSIAN)
        assert result == [Kernel.GAUSSIAN]

    def test_kernel_list(self):
        kernels = [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV]
        result = _normalise_kernels(kernels)
        assert result == kernels

    def test_empty_list_defaults(self):
        result = _normalise_kernels([])
        assert result == [Kernel.GAUSSIAN]

    def test_returns_list(self):
        result = _normalise_kernels(Kernel.TRIANGULAR)
        assert isinstance(result, list)

    def test_preserves_order(self):
        kernels = [Kernel.BOX, Kernel.GAUSSIAN, Kernel.TRIANGULAR]
        result = _normalise_kernels(kernels)
        assert result == kernels


class TestFitKDE:
    def test_returns_kde_object(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        kde = _fit_kde(data, bandwidth=1.0, kernel=Kernel.GAUSSIAN, weights=None)
        assert kde is not None

    def test_with_weights(self, sample_calibration_data, sample_weights):
        data = sample_calibration_data(n_samples=100)
        weights = sample_weights(n_samples=100, pattern="uniform")
        kde = _fit_kde(data, bandwidth=1.0, kernel=Kernel.GAUSSIAN, weights=weights)
        assert kde is not None

    def test_without_weights(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        kde = _fit_kde(data, bandwidth=1.0, kernel=Kernel.GAUSSIAN, weights=None)
        assert kde is not None

    def test_different_kernels(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        for kernel in [Kernel.GAUSSIAN, Kernel.EPANECHNIKOV, Kernel.TRIANGULAR]:
            kde = _fit_kde(data, bandwidth=1.0, kernel=kernel, weights=None)
            assert kde is not None

    def test_data_sorting(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        unsorted_data = data.copy()
        rng = np.random.default_rng(42)
        rng.shuffle(unsorted_data)

        kde1 = _fit_kde(data, bandwidth=1.0, kernel=Kernel.GAUSSIAN, weights=None)
        kde2 = _fit_kde(
            unsorted_data, bandwidth=1.0, kernel=Kernel.GAUSSIAN, weights=None
        )

        grid1, pdf1 = kde1.evaluate()
        grid2, pdf2 = kde2.evaluate()

        assert np.allclose(grid1, grid2)
        assert np.allclose(pdf1, pdf2)


class TestComputeCVLogLikelihood:
    def test_returns_number(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=5, weights=None, seed=42
        )
        assert isinstance(result, int | float)

    def test_returns_finite(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=5, weights=None, seed=42
        )
        assert np.isfinite(result)

    def test_leave_one_out(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=50)
        result = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=-1, weights=None, seed=42
        )
        assert np.isfinite(result)

    def test_k_fold(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100)
        result = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=10, weights=None, seed=42
        )
        assert np.isfinite(result)

    def test_with_weights(self, sample_calibration_data, sample_weights):
        data = sample_calibration_data(n_samples=100)
        weights = sample_weights(n_samples=100, pattern="uniform")
        result = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=5, weights=weights, seed=42
        )
        assert np.isfinite(result)

    def test_reproducible_with_seed(self, sample_calibration_data):
        data = sample_calibration_data(n_samples=100, seed=42)
        result1 = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=5, weights=None, seed=42
        )
        result2 = _compute_cv_log_likelihood(
            data, Kernel.GAUSSIAN, bandwidth=1.0, cv_folds=5, weights=None, seed=42
        )
        assert result1 == result2
