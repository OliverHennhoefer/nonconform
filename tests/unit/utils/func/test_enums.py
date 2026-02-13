import pytest

from nonconform.enums import Distribution, Kernel, Pruning, ScorePolarity


class TestDistribution:
    def test_has_expected_members(self):
        members = [Distribution.BETA_BINOMIAL, Distribution.UNIFORM, Distribution.GRID]
        assert len(members) == 3
        assert {member.name for member in Distribution} == {
            "BETA_BINOMIAL",
            "UNIFORM",
            "GRID",
        }

    def test_member_values_unique(self):
        values = [member.value for member in Distribution]
        assert len(values) == len(set(values))


class TestDataset:
    def test_dataset_enum_removed(self):
        import importlib

        constants = importlib.import_module("nonconform._internal.constants")
        with pytest.raises(AttributeError):
            getattr(constants, "Dataset")


class TestAggregation:
    def test_aggregation_enum_removed_from_public_api(self):
        import importlib

        enums = importlib.import_module("nonconform.enums")
        assert not hasattr(enums, "Aggregation")


class TestPruning:
    def test_has_expected_members(self):
        members = [
            Pruning.HETEROGENEOUS,
            Pruning.HOMOGENEOUS,
            Pruning.DETERMINISTIC,
        ]
        assert len(members) == 3
        assert {member.name for member in Pruning} == {
            "HETEROGENEOUS",
            "HOMOGENEOUS",
            "DETERMINISTIC",
        }

    def test_member_values_unique(self):
        values = [member.value for member in Pruning]
        assert len(values) == len(set(values))


class TestKernel:
    def test_has_expected_members(self):
        kernels = [
            Kernel.GAUSSIAN,
            Kernel.EXPONENTIAL,
            Kernel.BOX,
            Kernel.TRIANGULAR,
            Kernel.EPANECHNIKOV,
            Kernel.BIWEIGHT,
            Kernel.TRIWEIGHT,
            Kernel.TRICUBE,
            Kernel.COSINE,
        ]
        assert len(kernels) == 9
        assert {kernel.name for kernel in Kernel} == {
            "GAUSSIAN",
            "EXPONENTIAL",
            "BOX",
            "TRIANGULAR",
            "EPANECHNIKOV",
            "BIWEIGHT",
            "TRIWEIGHT",
            "TRICUBE",
            "COSINE",
        }

    def test_member_values_unique(self):
        values = [member.value for member in Kernel]
        assert len(values) == len(set(values))


class TestScorePolarity:
    def test_has_expected_members(self):
        members = [
            ScorePolarity.AUTO,
            ScorePolarity.HIGHER_IS_ANOMALOUS,
            ScorePolarity.HIGHER_IS_NORMAL,
        ]
        assert len(members) == 3
        assert {member.name for member in ScorePolarity} == {
            "AUTO",
            "HIGHER_IS_ANOMALOUS",
            "HIGHER_IS_NORMAL",
        }

    def test_member_values_unique(self):
        values = [member.value for member in ScorePolarity]
        assert len(values) == len(set(values))
