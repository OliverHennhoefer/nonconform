import pytest

from nonconform.utils.func.enums import (
    Aggregation,
    Distribution,
    Kernel,
    Pruning,
)


class TestDistribution:
    def test_has_beta_binomial_member(self):
        assert hasattr(Distribution, "BETA_BINOMIAL")

    def test_has_uniform_member(self):
        assert hasattr(Distribution, "UNIFORM")

    def test_has_grid_member(self):
        assert hasattr(Distribution, "GRID")

    def test_all_members_accessible(self):
        members = [Distribution.BETA_BINOMIAL, Distribution.UNIFORM, Distribution.GRID]
        assert len(members) == 3

    def test_member_values_unique(self):
        values = [m.value for m in Distribution]
        assert len(values) == len(set(values))


class TestAggregation:
    def test_has_mean_member(self):
        assert hasattr(Aggregation, "MEAN")

    def test_has_median_member(self):
        assert hasattr(Aggregation, "MEDIAN")

    def test_has_minimum_member(self):
        assert hasattr(Aggregation, "MINIMUM")

    def test_has_maximum_member(self):
        assert hasattr(Aggregation, "MAXIMUM")

    def test_all_members_accessible(self):
        members = [
            Aggregation.MEAN,
            Aggregation.MEDIAN,
            Aggregation.MINIMUM,
            Aggregation.MAXIMUM,
        ]
        assert len(members) == 4

    def test_member_values_unique(self):
        values = [m.value for m in Aggregation]
        assert len(values) == len(set(values))


class TestDataset:
    def test_dataset_enum_removed(self):
        import importlib

        enums = importlib.import_module("nonconform.utils.func.enums")
        with pytest.raises(AttributeError):
            getattr(enums, "Dataset")


class TestPruning:
    def test_has_heterogeneous_member(self):
        assert hasattr(Pruning, "HETEROGENEOUS")

    def test_has_homogeneous_member(self):
        assert hasattr(Pruning, "HOMOGENEOUS")

    def test_has_deterministic_member(self):
        assert hasattr(Pruning, "DETERMINISTIC")

    def test_all_members_accessible(self):
        members = [
            Pruning.HETEROGENEOUS,
            Pruning.HOMOGENEOUS,
            Pruning.DETERMINISTIC,
        ]
        assert len(members) == 3

    def test_member_values_unique(self):
        values = [m.value for m in Pruning]
        assert len(values) == len(set(values))


class TestKernel:
    def test_has_9_kernels(self):
        assert len(list(Kernel)) == 9

    def test_has_gaussian_kernel(self):
        assert Kernel.GAUSSIAN.value == "gaussian"

    def test_has_exponential_kernel(self):
        assert Kernel.EXPONENTIAL.value == "exponential"

    def test_triangular_has_short_name(self):
        assert Kernel.TRIANGULAR.value == "tri"

    def test_epanechnikov_has_short_name(self):
        assert Kernel.EPANECHNIKOV.value == "epa"

    def test_all_kernels_accessible(self):
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


class TestEnumProperties:
    def test_distribution_iteration(self):
        members = list(Distribution)
        assert len(members) == 3

    def test_aggregation_iteration(self):
        members = list(Aggregation)
        assert len(members) == 4

    def test_pruning_iteration(self):
        members = list(Pruning)
        assert len(members) == 3

    def test_enum_member_equality(self):
        assert Aggregation.MEAN == Aggregation.MEAN
        assert Aggregation.MEAN != Aggregation.MEDIAN

    def test_enum_member_identity(self):
        assert Aggregation.MEAN is Aggregation.MEAN

    def test_enum_string_representation(self):
        assert "MEAN" in str(Aggregation.MEAN)


class TestEnumValidation:
    def test_invalid_member_access_raises_error(self):
        with pytest.raises(AttributeError):
            _ = Aggregation.INVALID


class TestEnumUsage:
    def test_can_use_in_comparisons(self):
        agg = Aggregation.MEAN
        assert agg == Aggregation.MEAN
        assert agg != Aggregation.MEDIAN

    def test_can_use_in_sets(self):
        agg_set = {Aggregation.MEAN, Aggregation.MEDIAN, Aggregation.MEAN}
        assert len(agg_set) == 2

    def test_can_use_in_dicts(self):
        agg_dict = {Aggregation.MEAN: "mean_value", Aggregation.MEDIAN: "median_value"}
        assert agg_dict[Aggregation.MEAN] == "mean_value"

    def test_can_iterate_over_enum(self):
        count = 0
        for _ in Aggregation:
            count += 1
        assert count == 4
