"""Unit tests for strategy/calibration/cross_validation.py."""

import numpy as np
import pytest

from nonconform import CrossValidation
from tests.conftest import MockDetector


class TestCrossValidationFitCalibrate:
    """Tests for CrossValidation.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((50, 4))

    def test_plus_true_returns_k_detectors(self, sample_data):
        """plus=True returns k trained detectors."""
        strategy = CrossValidation(k=5, plus=True, shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == 5
        assert len(calib_scores) == len(sample_data)

    def test_plus_false_returns_single_detector(self, sample_data):
        """plus=False returns a single trained detector."""
        strategy = CrossValidation(k=5, plus=False, shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_calibration_ids_cover_all_samples(self, sample_data):
        """Calibration IDs cover all samples exactly once."""
        strategy = CrossValidation(k=5, plus=True, shuffle=True)
        strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123, weighted=True
        )
        calib_ids = strategy.calibration_ids
        assert len(calib_ids) == len(sample_data)
        assert set(calib_ids) == set(range(len(sample_data)))

    def test_jackknife_factory_uses_leave_one_out(self):
        """Jackknife factory uses k=n (leave-one-out)."""
        rng = np.random.default_rng(7)
        sample_data = rng.standard_normal((12, 3))
        strategy = CrossValidation.jackknife(plus=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=7
        )
        assert len(detector_set) == len(sample_data)
        assert len(calib_scores) == len(sample_data)
