from nonconform.utils.data.generator import OnlineGenerator


class TestEdgeProportions:
    def test_zero_proportion(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.0,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 0

    def test_fifty_percent_proportion(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.5,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 50

    def test_ninety_nine_percent_proportion(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=0.99,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 99

    def test_one_hundred_percent_proportion(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=1.0,
            n_instances=50,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=50))
        assert total_anomalies == 50

    def test_near_zero_proportion(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=0.001,
            n_instances=1000,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=1000))
        assert total_anomalies == 1


class TestSmallNInstances:
    def test_single_instance(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.5,
            n_instances=1,
            seed=42,
        )
        instances = list(gen.generate(n_instances=1))
        assert len(instances) == 1

    def test_five_instances(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.2,
            n_instances=5,
            seed=42,
        )
        instances = list(gen.generate(n_instances=5))
        assert len(instances) == 5
        total_anomalies = sum(label for _, label in instances)
        assert total_anomalies == 1

    def test_ten_instances(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        instances = list(gen.generate(n_instances=10))
        assert len(instances) == 10
        total_anomalies = sum(label for _, label in instances)
        assert total_anomalies == 1


class TestLargeNInstances:
    def test_ten_thousand_instances(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=0.01,
            n_instances=10000,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=10000))
        assert total_anomalies == 100

    def test_large_with_small_proportion(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=0.0001,
            n_instances=10000,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=10000))
        assert total_anomalies == 1


class TestSmallDatasets:
    def test_tiny_dataset(self, tiny_dataset):
        gen = OnlineGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=10, n_anomaly=5),
            anomaly_proportion=0.2,
            n_instances=25,
            seed=42,
        )
        instances = list(gen.generate(n_instances=25))
        assert len(instances) == 25
        total_anomalies = sum(label for _, label in instances)
        assert total_anomalies == 5

    def test_very_small_dataset(self, tiny_dataset):
        gen = OnlineGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=5, n_anomaly=5),
            anomaly_proportion=0.5,
            n_instances=10,
            seed=42,
        )
        instances = list(gen.generate(n_instances=10))
        assert len(instances) == 10


class TestImbalancedDatasets:
    def test_highly_imbalanced_normal_heavy(self, imbalanced_dataset):
        gen = OnlineGenerator(
            load_data_func=lambda: imbalanced_dataset(n_normal=990, n_anomaly=10),
            anomaly_proportion=0.01,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 1

    def test_imbalanced_with_low_proportion(self, imbalanced_dataset):
        gen = OnlineGenerator(
            load_data_func=lambda: imbalanced_dataset(n_normal=500, n_anomaly=50),
            anomaly_proportion=0.05,
            n_instances=200,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=200))
        assert total_anomalies == 10


class TestFeatureIntegrity:
    def test_feature_columns_preserved(self, small_dataset):
        df = small_dataset(n_features=9)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.1,
            n_instances=5,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=5):
            expected_cols = [f"V{i + 1}" for i in range(9)]
            assert list(x_instance.columns) == expected_cols

    def test_no_nan_values(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=100):
            assert not x_instance.isna().any().any()

    def test_class_column_removed(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=10):
            assert "Class" not in x_instance.columns


class TestInstanceConsistency:
    def test_all_instances_size_one(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=50):
            assert len(x_instance) == 1

    def test_instance_index_reset(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=20,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=20):
            assert x_instance.index.tolist() == [0]


class TestSamplingWithReplacement:
    def test_can_generate_more_than_dataset_size(self, tiny_dataset):
        gen = OnlineGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=10, n_anomaly=5),
            anomaly_proportion=0.2,
            n_instances=500,
            seed=42,
        )
        instances = list(gen.generate(n_instances=500))
        assert len(instances) == 500
        total_anomalies = sum(label for _, label in instances)
        assert total_anomalies == 100


class TestVariableLabelDistribution:
    def test_labels_vary_across_sequence(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.5,
            n_instances=20,
            seed=42,
        )
        labels = [label for _, label in gen.generate(n_instances=20)]
        first_half = labels[:10]
        assert 0 in first_half
        assert 1 in first_half

    def test_not_all_normals_first(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.2,
            n_instances=50,
            seed=42,
        )
        labels = [label for _, label in gen.generate(n_instances=50)]
        first_anomaly_index = labels.index(1) if 1 in labels else -1
        assert first_anomaly_index < 45


class TestPrecisionEdgeCases:
    def test_fractional_anomalies_rounds_down(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.015,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 1

    def test_exact_fraction(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.25,
            n_instances=8,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=8))
        assert total_anomalies == 2


class TestDataTypeConsistency:
    def test_label_is_int(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=20,
            seed=42,
        )
        for _, label in gen.generate(n_instances=20):
            assert isinstance(label, int)
            assert not isinstance(label, bool)
