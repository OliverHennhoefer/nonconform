"""Unit tests for strategy/calibration/jackknife_bootstrap.py."""

import numpy as np
import pytest

from nonconform import JackknifeBootstrap
from tests.conftest import MockDetector


class TestJackknifeBootstrapFitCalibrate:
    """Tests for JackknifeBootstrap.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(101)
        return rng.standard_normal((30, 5))

    @pytest.fixture
    def detector(self):
        """Deterministic mock detector."""
        return MockDetector(scores=np.array([0.1, 0.5, 0.9]))

    def test_plus_mode_returns_n_bootstrap_detectors(self, sample_data, detector):
        """mode='plus' returns n_bootstraps trained detectors."""
        strategy = JackknifeBootstrap(n_bootstraps=5, mode="plus")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=123
        )
        assert len(detector_set) == 5
        assert len(calib_scores) == len(sample_data)
        assert not np.any(np.isnan(calib_scores))

    def test_single_model_mode_returns_single_detector(self, sample_data, detector):
        """mode='single_model' returns a single trained detector."""
        strategy = JackknifeBootstrap(n_bootstraps=5, mode="single_model")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=123
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_calibration_ids_cover_all_samples(self, sample_data, detector):
        """Calibration IDs cover all samples in JaB+."""
        strategy = JackknifeBootstrap(n_bootstraps=5, mode="plus")
        strategy.fit_calibrate(x=sample_data, detector=detector, seed=123)
        assert strategy.calibration_ids == list(range(len(sample_data)))

    def test_n_jobs_minus_one_runs_without_error(self, sample_data, detector):
        """n_jobs=-1 runs without error and returns expected results."""
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="plus")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=detector, seed=123, n_jobs=-1
        )
        assert len(detector_set) == 3
        assert len(calib_scores) == len(sample_data)

    def test_n_jobs_zero_raises(self, sample_data, detector):
        """n_jobs must be None, -1, or positive integer."""
        strategy = JackknifeBootstrap(n_bootstraps=3, mode="plus")
        with pytest.raises(ValueError, match="n_jobs"):
            strategy.fit_calibrate(x=sample_data, detector=detector, seed=123, n_jobs=0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            JackknifeBootstrap(n_bootstraps=3, mode="invalid")  # type: ignore[arg-type]
