"""Edge case tests for parameter configuration with alias support."""

from unittest.mock import MagicMock

from nonconform.utils.func.params import _set_params


class TestParameterAliases:
    """Test parameter alias support for cross-library compatibility."""

    def test_random_state_alias_seed(self):
        """Test 'seed' alias works for random_state."""
        detector = MagicMock()
        detector.get_params.return_value = {"seed": None}
        params_dict = {"seed": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=42)

        detector.set_params.assert_called()
        assert params_dict["seed"] == 42

    def test_random_state_alias_random_seed(self):
        """Test 'random_seed' alias works."""
        detector = MagicMock()
        detector.get_params.return_value = {"random_seed": None}
        params_dict = {"random_seed": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=99)

        detector.set_params.assert_called()
        assert params_dict["random_seed"] == 99

    def test_n_jobs_alias_n_threads(self):
        """Test 'n_threads' alias works for n_jobs."""
        detector = MagicMock()
        detector.get_params.return_value = {"n_threads": 1}
        params_dict = {"n_threads": 1}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=None)

        assert params_dict["n_threads"] == -1

    def test_n_jobs_alias_num_workers(self):
        """Test 'num_workers' alias works."""
        detector = MagicMock()
        detector.get_params.return_value = {"num_workers": 1, "random_state": None}
        params_dict = {"num_workers": 1, "random_state": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=42)

        assert params_dict["num_workers"] == -1

    def test_prefers_random_state_over_aliases(self):
        """Test random_state is preferred when multiple aliases exist."""
        detector = MagicMock()
        detector.get_params.return_value = {"random_state": None, "seed": None}
        params_dict = {"random_state": None, "seed": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=42)

        assert params_dict["random_state"] == 42
        assert params_dict["seed"] is None


class TestMissingRandomState:
    """Test behavior when random_state parameter is missing."""

    def test_warns_when_seed_provided_but_no_alias(self, caplog, capfd):
        """Test warning when seed provided but no random_state alias."""
        import logging

        # Ensure logger propagates to caplog by temporarily enabling it
        logger = logging.getLogger("nonconform")
        original_propagate = logger.propagate
        logger.propagate = True

        detector = MagicMock()
        detector.get_params.return_value = {}

        try:
            with caplog.at_level(logging.WARNING, logger="nonconform"):
                _set_params(detector, seed=42)

            # Check caplog records, stderr, and caplog.text for warning
            captured = capfd.readouterr()
            all_output = caplog.text + captured.err
            record_messages = " ".join(r.message for r in caplog.records)
            all_output += record_messages

            assert "random_state" in all_output
            assert "Reproducibility cannot be guaranteed" in all_output
        finally:
            logger.propagate = original_propagate

    def test_no_warning_when_seed_none_and_no_alias(self, caplog):
        """Test no warning when seed=None even without random_state."""
        import logging

        detector = MagicMock()
        detector.get_params.return_value = {}

        with caplog.at_level(logging.WARNING):
            _set_params(detector, seed=None)

        assert "random_state" not in caplog.text


class TestOptionalParameters:
    """Test optional parameter handling."""

    def test_contamination_optional(self):
        """Test contamination parameter is optional (no error if missing)."""
        detector = MagicMock()
        detector.get_params.return_value = {"random_state": None}
        params_dict = {"random_state": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=42)

    def test_n_jobs_optional(self):
        """Test n_jobs parameter is optional."""
        detector = MagicMock()
        detector.get_params.return_value = {"random_state": None}
        params_dict = {"random_state": None}
        detector.set_params.side_effect = lambda **kwargs: params_dict.update(kwargs)

        _set_params(detector, seed=42)


class TestRandomIteration:
    """Test dynamic seed generation for iterative strategies."""

    def test_random_iteration_creates_different_seed(self, mock_detector):
        """Test different iterations produce different seeds."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=0)
        seed0 = detector.get_params()["random_state"]

        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=1)
        seed1 = detector.get_params()["random_state"]

        assert seed0 != seed1

    def test_same_iteration_and_seed_produces_same_random_state(self, mock_detector):
        """Test reproducibility with same seed and iteration."""
        detector1 = mock_detector()
        _set_params(detector1, seed=42, random_iteration=True, iteration=5)

        detector2 = mock_detector()
        _set_params(detector2, seed=42, random_iteration=True, iteration=5)

        assert (
            detector1.get_params()["random_state"]
            == detector2.get_params()["random_state"]
        )

    def test_different_base_seeds_with_same_iteration(self, mock_detector):
        """Test different base seeds produce different results."""
        detector1 = mock_detector()
        _set_params(detector1, seed=42, random_iteration=True, iteration=0)

        detector2 = mock_detector()
        _set_params(detector2, seed=99, random_iteration=True, iteration=0)

        assert (
            detector1.get_params()["random_state"]
            != detector2.get_params()["random_state"]
        )

    def test_random_state_within_32bit_range(self, mock_detector):
        """Test generated random_state fits in 32-bit range."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=999999)

        random_state = detector.get_params()["random_state"]
        assert 0 <= random_state < 2**32

    def test_random_iteration_false_uses_base_seed(self, mock_detector):
        """Test random_iteration=False uses base seed directly."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=False, iteration=10)

        assert detector.get_params()["random_state"] == 42

    def test_random_iteration_without_iteration_uses_base_seed(self, mock_detector):
        """Test missing iteration parameter uses base seed."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=None)

        assert detector.get_params()["random_state"] == 42


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_iteration_zero(self, mock_detector):
        """Test iteration=0 is valid."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=0)

        assert detector.get_params()["random_state"] is not None

    def test_large_iteration_number(self, mock_detector):
        """Test very large iteration numbers work correctly."""
        detector = mock_detector()
        _set_params(detector, seed=42, random_iteration=True, iteration=1000000)

        random_state = detector.get_params()["random_state"]
        assert 0 <= random_state < 2**32

    def test_negative_seed(self, mock_detector):
        """Test negative seeds are accepted."""
        detector = mock_detector()
        _set_params(detector, seed=-42, random_iteration=False)

        assert detector.get_params()["random_state"] == -42

    def test_very_large_seed(self, mock_detector):
        """Test very large seeds work correctly."""
        large_seed = 2**31 - 1
        detector = mock_detector()
        _set_params(detector, seed=large_seed, random_iteration=False)

        assert detector.get_params()["random_state"] == large_seed

    def test_zero_seed(self, mock_detector):
        """Test seed=0 is valid."""
        detector = mock_detector()
        _set_params(detector, seed=0, random_iteration=False)

        assert detector.get_params()["random_state"] == 0


class TestReproducibility:
    """Test deterministic behavior and reproducibility."""

    def test_multiple_calls_same_iteration_same_result(self, mock_detector):
        """Test reproducibility across multiple calls."""
        results = []
        for _ in range(5):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=7)
            results.append(detector.get_params()["random_state"])

        assert len(set(results)) == 1

    def test_sequential_iterations_differ(self, mock_detector):
        """Test sequential iterations produce different seeds."""
        seeds = []
        for i in range(10):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=i)
            seeds.append(detector.get_params()["random_state"])

        assert len(set(seeds)) == 10

    def test_hash_collision_unlikely(self, mock_detector):
        """Test hash collisions are unlikely."""
        seeds = set()
        for i in range(100):
            detector = mock_detector()
            _set_params(detector, seed=42, random_iteration=True, iteration=i)
            seeds.add(detector.get_params()["random_state"])

        assert len(seeds) >= 99


class TestParameterOverride:
    """Test parameter override behavior."""

    def test_existing_contamination_gets_overwritten(self, mock_detector):
        """Test contamination is set to minimum value."""
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["contamination"] = 0.5

        _set_params(detector, seed=42)

        import sys

        assert detector.get_params()["contamination"] == sys.float_info.min

    def test_existing_n_jobs_gets_overwritten(self, mock_detector):
        """Test n_jobs is set to -1."""
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["n_jobs"] = 1

        _set_params(detector, seed=42)

        assert detector.get_params()["n_jobs"] == -1

    def test_existing_random_state_gets_overwritten(self, mock_detector):
        """Test random_state is overwritten with provided seed."""
        detector = mock_detector()
        initial_params = detector.get_params().copy()
        initial_params["random_state"] = 999

        _set_params(detector, seed=42)

        assert detector.get_params()["random_state"] == 42
