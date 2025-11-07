import sys

from nonconform.utils.func.params import _set_params


class TestContaminationSetting:
    def test_sets_contamination_to_min_float(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["contamination"] == sys.float_info.min

    def test_contamination_value_is_positive(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["contamination"] > 0

    def test_contamination_value_is_very_small(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["contamination"] < 1e-300


class TestRandomStateSetting:
    def test_sets_random_state_with_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["random_state"] == 42

    def test_different_seeds_set_different_random_states(self, mock_detector):
        detector1 = mock_detector()
        detector2 = mock_detector()

        _set_params(detector1, seed=42)
        _set_params(detector2, seed=123)

        assert (
            detector1.get_params()["random_state"]
            != detector2.get_params()["random_state"]
        )

    def test_same_seed_sets_same_random_state(self, mock_detector):
        detector1 = mock_detector()
        detector2 = mock_detector()

        _set_params(detector1, seed=42)
        _set_params(detector2, seed=42)

        assert (
            detector1.get_params()["random_state"]
            == detector2.get_params()["random_state"]
        )


class TestNJobsSetting:
    def test_sets_n_jobs_to_minus_one(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["n_jobs"] == -1

    def test_n_jobs_enables_parallel_processing(self, mock_detector):
        detector = mock_detector()
        initial_n_jobs = detector.get_params()["n_jobs"]
        _set_params(detector, seed=42)

        assert detector.get_params()["n_jobs"] != initial_n_jobs


class TestParameterExistence:
    def test_handles_detector_without_contamination(self, mock_detector):
        detector = mock_detector(has_contamination=False)
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert "contamination" not in params

    def test_handles_detector_without_n_jobs(self, mock_detector):
        detector = mock_detector(has_n_jobs=False)
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert "n_jobs" not in params

    def test_handles_detector_without_random_state(self, mock_detector):
        detector = mock_detector(has_random_state=False)
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert "random_state" not in params

    def test_sets_only_available_parameters(self, mock_detector):
        detector = mock_detector(
            has_contamination=True, has_n_jobs=False, has_random_state=True
        )
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["contamination"] == sys.float_info.min
        assert params["random_state"] == 42
        assert "n_jobs" not in params


class TestReturnValue:
    def test_returns_detector_instance(self, mock_detector):
        detector = mock_detector()
        result = _set_params(detector, seed=42)

        assert result is detector

    def test_returned_detector_has_updated_params(self, mock_detector):
        detector = mock_detector()
        result = _set_params(detector, seed=42)

        params = result.get_params()
        assert params["contamination"] == sys.float_info.min
        assert params["random_state"] == 42


class TestMultipleParameterSetting:
    def test_all_parameters_set_together(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=99)

        params = detector.get_params()
        assert params["contamination"] == sys.float_info.min
        assert params["n_jobs"] == -1
        assert params["random_state"] == 99

    def test_parameters_not_overwritten_by_defaults(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42)

        params = detector.get_params()
        assert params["random_state"] == 42
        assert params["contamination"] != 0.1
