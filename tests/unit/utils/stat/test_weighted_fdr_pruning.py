import numpy as np

from nonconform.utils.func.enums import Pruning
from nonconform.utils.stat.weighted_fdr import weighted_false_discovery_control


class TestHeterogeneousPruning:
    def test_heterogeneous_pruning(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=42
        )
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 20

    def test_heterogeneous_with_seed(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=42
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=42
        )
        np.testing.assert_array_equal(discoveries1, discoveries2)

    def test_heterogeneous_different_seeds(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=42
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=123
        )
        if np.sum(discoveries1) > 0 and np.sum(discoveries2) > 0:
            assert not np.array_equal(discoveries1, discoveries2)


class TestHomogeneousPruning:
    def test_homogeneous_pruning(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=42
        )
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 20

    def test_homogeneous_with_seed(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=42
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=42
        )
        np.testing.assert_array_equal(discoveries1, discoveries2)

    def test_homogeneous_different_seeds(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=42
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=123
        )
        if np.sum(discoveries1) > 0 and np.sum(discoveries2) > 0:
            assert not np.array_equal(discoveries1, discoveries2)


class TestDeterministicPruning:
    def test_deterministic_pruning(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=42
        )
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 20

    def test_deterministic_is_reproducible(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=42
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=99
        )
        np.testing.assert_array_equal(discoveries1, discoveries2)

    def test_deterministic_ignores_seed(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries1 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=1
        )
        discoveries2 = weighted_false_discovery_control(
            result=result, alpha=0.1, pruning=Pruning.DETERMINISTIC, seed=1000
        )
        np.testing.assert_array_equal(discoveries1, discoveries2)


class TestSeedReproducibility:
    def test_same_seed_same_results_heterogeneous(self, conformal_result):
        result = conformal_result(n_test=30, n_calib=100, seed=123)

        discoveries = []
        for _ in range(3):
            disc = weighted_false_discovery_control(
                result=result, alpha=0.1, pruning=Pruning.HETEROGENEOUS, seed=777
            )
            discoveries.append(disc)

        for i in range(1, len(discoveries)):
            np.testing.assert_array_equal(discoveries[0], discoveries[i])

    def test_same_seed_same_results_homogeneous(self, conformal_result):
        result = conformal_result(n_test=30, n_calib=100, seed=123)

        discoveries = []
        for _ in range(3):
            disc = weighted_false_discovery_control(
                result=result, alpha=0.1, pruning=Pruning.HOMOGENEOUS, seed=777
            )
            discoveries.append(disc)

        for i in range(1, len(discoveries)):
            np.testing.assert_array_equal(discoveries[0], discoveries[i])


class TestPruningComparison:
    def test_all_pruning_methods_return_valid_output(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)

        for pruning in [
            Pruning.HETEROGENEOUS,
            Pruning.HOMOGENEOUS,
            Pruning.DETERMINISTIC,
        ]:
            discoveries = weighted_false_discovery_control(
                result=result, alpha=0.1, pruning=pruning, seed=42
            )
            assert isinstance(discoveries, np.ndarray)
            assert len(discoveries) == 20
            assert discoveries.dtype == bool
