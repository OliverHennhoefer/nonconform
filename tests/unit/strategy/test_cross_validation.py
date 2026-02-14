"""Unit tests for strategy/calibration/cross_validation.py."""

import numpy as np
import pytest

from nonconform import CrossValidation
from nonconform.enums import ConformalMode
from tests.conftest import MockDetector


class TestCrossValidationFitCalibrate:
    """Tests for CrossValidation.fit_calibrate()."""

    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((50, 4))

    def test_plus_mode_returns_k_detectors(self, sample_data):
        """mode='plus' returns k trained detectors."""
        strategy = CrossValidation(k=5, mode="plus", shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == 5
        assert len(calib_scores) == len(sample_data)

    def test_single_model_mode_returns_single_detector(self, sample_data):
        """mode='single_model' returns a single trained detector."""
        strategy = CrossValidation(k=5, mode="single_model", shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_calibration_ids_cover_all_samples(self, sample_data):
        """Calibration IDs cover all samples exactly once."""
        strategy = CrossValidation(k=5, mode="plus", shuffle=True)
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
        strategy = CrossValidation.jackknife(mode="plus")
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=7
        )
        assert len(detector_set) == len(sample_data)
        assert len(calib_scores) == len(sample_data)

    def test_accepts_enum_mode(self, sample_data):
        strategy = CrossValidation(k=5, mode=ConformalMode.PLUS, shuffle=True)
        detector_set, _ = strategy.fit_calibrate(
            x=sample_data, detector=MockDetector(), seed=123
        )
        assert len(detector_set) == 5

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            CrossValidation(k=5, mode="invalid")  # type: ignore[arg-type]

    def test_non_boolean_shuffle_raises(self):
        with pytest.raises(TypeError, match="shuffle must be a boolean value"):
            CrossValidation(k=5, mode="plus", shuffle=1)  # type: ignore[arg-type]
