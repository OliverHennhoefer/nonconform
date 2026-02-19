"""Unit tests for strategy/calibration/cross_validation.py."""

import numpy as np
import pytest

from nonconform import CrossValidation
from nonconform.enums import ConformalMode
from tests.conftest import MockDetector


class SetParamsTypeErrorDetector:
    """Detector whose set_params raises TypeError (unsupported random_state)."""

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> "SetParamsTypeErrorDetector":
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X), dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> "SetParamsTypeErrorDetector":
        raise TypeError("set_params is not supported")

    def __copy__(self) -> "SetParamsTypeErrorDetector":
        return SetParamsTypeErrorDetector()

    def __deepcopy__(self, memo: dict) -> "SetParamsTypeErrorDetector":
        return SetParamsTypeErrorDetector()


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

    def test_k_below_two_raises_with_diagnostic_notes(self, sample_data):
        strategy = CrossValidation(k=1, mode="plus", shuffle=True)
        with pytest.raises(ValueError, match="k must be at least 2"):
            strategy.fit_calibrate(
                x=sample_data,
                detector=MockDetector(),
                seed=123,
            )

    def test_k_above_sample_size_raises_with_diagnostic_notes(self, sample_data):
        strategy = CrossValidation(k=len(sample_data) + 1, mode="plus", shuffle=True)
        with pytest.raises(ValueError, match="Not enough samples"):
            strategy.fit_calibrate(
                x=sample_data,
                detector=MockDetector(),
                seed=123,
            )

    def test_set_params_type_error_is_ignored(self, sample_data):
        strategy = CrossValidation(k=5, mode="single_model", shuffle=True)
        detector_set, calib_scores = strategy.fit_calibrate(
            x=sample_data,
            detector=SetParamsTypeErrorDetector(),
            seed=123,
        )
        assert len(detector_set) == 1
        assert len(calib_scores) == len(sample_data)

    def test_public_properties_reflect_configuration(self):
        plus = CrossValidation(k=7, mode="plus", shuffle=True)
        assert plus.k == 7
        assert plus.mode == "plus"
        jackknife = CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL)
        assert jackknife.k is None
        assert jackknife.mode == "single_model"
