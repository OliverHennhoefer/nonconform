from unittest.mock import MagicMock

import pytest

from nonconform.utils.func.params import _set_params, forbidden_model_list


class TestForbiddenModels:
    def test_cblof_raises_error(self):
        from pyod.models.cblof import CBLOF

        detector = MagicMock(spec=CBLOF)
        detector.__class__ = CBLOF
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="not supported"):
            _set_params(detector, seed=42)

    def test_cof_raises_error(self):
        from pyod.models.cof import COF

        detector = MagicMock(spec=COF)
        detector.__class__ = COF
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="not supported"):
            _set_params(detector, seed=42)

    def test_rgraph_raises_error(self):
        from pyod.models.rgraph import RGraph

        detector = MagicMock(spec=RGraph)
        detector.__class__ = RGraph
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="not supported"):
            _set_params(detector, seed=42)

    def test_sampling_raises_error(self):
        from pyod.models.sampling import Sampling

        detector = MagicMock(spec=Sampling)
        detector.__class__ = Sampling
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="not supported"):
            _set_params(detector, seed=42)

    def test_sos_raises_error(self):
        from pyod.models.sos import SOS

        detector = MagicMock(spec=SOS)
        detector.__class__ = SOS
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="not supported"):
            _set_params(detector, seed=42)

    def test_error_message_includes_model_name(self):
        from pyod.models.cblof import CBLOF

        detector = MagicMock(spec=CBLOF)
        detector.__class__ = CBLOF
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="CBLOF"):
            _set_params(detector, seed=42)

    def test_error_message_lists_forbidden_models(self):
        from pyod.models.cblof import CBLOF

        detector = MagicMock(spec=CBLOF)
        detector.__class__ = CBLOF
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="Forbidden detectors"):
            _set_params(detector, seed=42)

    def test_error_message_suggests_alternatives(self):
        from pyod.models.cblof import CBLOF

        detector = MagicMock(spec=CBLOF)
        detector.__class__ = CBLOF
        detector.get_params.return_value = {}

        with pytest.raises(ValueError, match="IForest|HBOS|ECOD"):
            _set_params(detector, seed=42)

    def test_forbidden_list_has_five_models(self):
        assert len(forbidden_model_list) == 5


class TestRandomIteration:
    def test_random_iteration_creates_different_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=0)
        seed0 = detector.get_params()["random_state"]

        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=1)
        seed1 = detector.get_params()["random_state"]

        assert seed0 != seed1

    def test_same_iteration_and_seed_produces_same_random_state(self, mock_detector):
        detector1 = mock_detector()
        _set_params(detector1, seed=42, random_iteration=True, iteration=5)

        detector2 = mock_detector()
        _set_params(detector2, seed=42, random_iteration=True, iteration=5)

        assert (
            detector1.get_params()["random_state"]
            == detector2.get_params()["random_state"]
        )

    def test_different_base_seeds_with_same_iteration(self, mock_detector):
        detector1 = mock_detector()
        _set_params(detector1, seed=42, random_iteration=True, iteration=0)

        detector2 = mock_detector()
        _set_params(detector2, seed=99, random_iteration=True, iteration=0)

        assert (
            detector1.get_params()["random_state"]
            != detector2.get_params()["random_state"]
        )

    def test_random_state_within_32bit_range(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=999999)

        random_state = detector.get_params()["random_state"]
        assert 0 <= random_state < 2**32

    def test_random_iteration_false_uses_base_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=False, iteration=10)

        assert detector.get_params()["random_state"] == 42

    def test_random_iteration_without_iteration_uses_base_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=None)

        assert detector.get_params()["random_state"] == 42


class TestEdgeCases:
    def test_iteration_zero(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=0)

        assert detector.get_params()["random_state"] is not None

    def test_large_iteration_number(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=1000000)

        random_state = detector.get_params()["random_state"]
        assert 0 <= random_state < 2**32

    def test_negative_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=-42, random_iteration=False)

        assert detector.get_params()["random_state"] == -42

    def test_very_large_seed(self, mock_detector):
        large_seed = 2**31 - 1
        detector = mock_detector()
        _set_params(detector, seed=large_seed, random_iteration=False)

        assert detector.get_params()["random_state"] == large_seed

    def test_zero_seed(self, mock_detector):
        detector = mock_detector()
        _set_params(detector, seed=0, random_iteration=False)

        assert detector.get_params()["random_state"] == 0


class TestReproducibility:
    def test_multiple_calls_same_iteration_same_result(self, mock_detector):
        results = []
        for _ in range(5):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=7)
            results.append(detector.get_params()["random_state"])

        assert len(set(results)) == 1

    def test_sequential_iterations_differ(self, mock_detector):
        seeds = []
        for i in range(10):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=i)
            seeds.append(detector.get_params()["random_state"])

        assert len(set(seeds)) == 10

    def test_hash_collision_unlikely(self, mock_detector):
        seeds = set()
        for i in range(100):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=i)
            seeds.add(detector.get_params()["random_state"])

        assert len(seeds) >= 99


class TestParameterOverride:
    def test_existing_contamination_gets_overwritten(self, mock_detector):
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["contamination"] = 0.5

        _set_params(detector, seed=42)

        import sys

        assert detector.get_params()["contamination"] == sys.float_info.min

    def test_existing_n_jobs_gets_overwritten(self, mock_detector):
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["n_jobs"] = 1

        _set_params(detector, seed=42)

        assert detector.get_params()["n_jobs"] == -1

    def test_existing_random_state_gets_overwritten(self, mock_detector):
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["random_state"] = 999

        _set_params(detector, seed=42)

        assert detector.get_params()["random_state"] == 42
