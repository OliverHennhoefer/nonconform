from nonconform.utils.data.generator import BatchGenerator

from .conftest import assert_batch_valid


class TestTruncationWarning:
    def test_warning_for_small_proportion(self, small_dataset, caplog):
        with caplog.at_level("WARNING"):
            gen = BatchGenerator(
                load_data_func=small_dataset,
                batch_size=100,
                anomaly_proportion=0.005,
                anomaly_mode="proportional",
                n_batches=1,
                seed=42,
            )
            assert gen.n_anomaly_per_batch == 0

    def test_warning_suggests_minimum_batch_size(self, small_dataset, caplog):
        with caplog.at_level("WARNING"):
            gen = BatchGenerator(
                load_data_func=small_dataset,
                batch_size=100,
                anomaly_proportion=0.005,
                anomaly_mode="proportional",
                seed=42,
            )
            assert gen.n_anomaly_per_batch == 0

    def test_warning_suggests_probabilistic_mode(self, small_dataset, caplog):
        with caplog.at_level("WARNING"):
            gen = BatchGenerator(
                load_data_func=small_dataset,
                batch_size=100,
                anomaly_proportion=0.005,
                anomaly_mode="proportional",
                seed=42,
            )
            assert gen.n_anomaly_per_batch == 0

    def test_no_warning_for_valid_proportion(self, small_dataset, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.01,
            n_batches=1,
            seed=42,
        )
        assert "results in 0 anomalies" not in caplog.text

    def test_truncated_batch_has_zero_anomalies(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.005,
            n_batches=1,
            seed=42,
        )
        x_batch, y_batch = next(gen.generate())
        assert y_batch.sum() == 0


class TestEdgeProportions:
    def test_zero_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.0,
            n_batches=3,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 0
            assert (y_batch == 0).sum() == 100

    def test_fifty_percent_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.5,
            n_batches=3,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 50
            assert (y_batch == 0).sum() == 50

    def test_near_zero_proportion(self, large_dataset):
        gen = BatchGenerator(
            load_data_func=large_dataset,
            batch_size=1000,
            anomaly_proportion=0.001,
            n_batches=2,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 1

    def test_no_warning_for_zero_proportion(self, small_dataset, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.0,
            seed=42,
        )
        assert "results in 0 anomalies" not in caplog.text


class TestExtremeBatchSizes:
    def test_batch_size_one_with_half_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=1,
            anomaly_proportion=0.5,
            n_batches=10,
            seed=42,
        )
        batches = list(gen.generate())
        assert all(len(y) == 1 for _, y in batches)

    def test_batch_size_one_with_zero_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=1,
            anomaly_proportion=0.0,
            n_batches=10,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert len(y_batch) == 1
            assert y_batch.sum() == 0

    def test_batch_size_one_with_full_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=1,
            anomaly_proportion=1.0,
            n_batches=10,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert len(y_batch) == 1
            assert y_batch.sum() == 1

    def test_very_large_batch_size(self, large_dataset):
        gen = BatchGenerator(
            load_data_func=large_dataset,
            batch_size=500,
            anomaly_proportion=0.1,
            n_batches=2,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert_batch_valid(x_batch, y_batch, 500)
            assert y_batch.sum() == 50


class TestSmallDatasets:
    def test_tiny_dataset_proportional(self, tiny_dataset):
        gen = BatchGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=10, n_anomaly=5),
            batch_size=5,
            anomaly_proportion=0.2,
            n_batches=3,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert_batch_valid(x_batch, y_batch, 5)
            assert y_batch.sum() == 1

    def test_tiny_dataset_probabilistic(self, tiny_dataset):
        gen = BatchGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=10, n_anomaly=5),
            batch_size=5,
            anomaly_proportion=0.2,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )
        batches = list(gen.generate())
        total_anomalies = sum(y.sum() for _, y in batches)
        assert total_anomalies == int(5 * 5 * 0.2)


class TestImbalancedDatasets:
    def test_highly_imbalanced_normal_heavy(self, imbalanced_dataset):
        gen = BatchGenerator(
            load_data_func=lambda: imbalanced_dataset(n_normal=990, n_anomaly=10),
            batch_size=100,
            anomaly_proportion=0.01,
            n_batches=5,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 1

    def test_imbalanced_with_low_proportion(self, imbalanced_dataset):
        gen = BatchGenerator(
            load_data_func=lambda: imbalanced_dataset(n_normal=500, n_anomaly=50),
            batch_size=100,
            anomaly_proportion=0.05,
            n_batches=3,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 5


class TestFeatureIntegrity:
    def test_feature_columns_preserved(self, small_dataset):
        df = small_dataset(n_features=7)
        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=1,
            seed=42,
        )
        x_batch, _ = next(gen.generate())
        expected_cols = [f"V{i + 1}" for i in range(7)]
        assert list(x_batch.columns) == expected_cols

    def test_no_nan_values(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        for x_batch, _ in gen.generate():
            assert not x_batch.isna().any().any()

    def test_class_column_removed(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=1,
            seed=42,
        )
        x_batch, _ = next(gen.generate())
        assert "Class" not in x_batch.columns


class TestSamplingWithReplacement:
    def test_can_generate_more_than_dataset_size(self, tiny_dataset):
        gen = BatchGenerator(
            load_data_func=lambda: tiny_dataset(n_normal=10, n_anomaly=5),
            batch_size=5,
            anomaly_proportion=0.2,
            n_batches=100,
            seed=42,
        )
        batches = list(gen.generate())
        assert len(batches) == 100
        total_instances = sum(len(x) for x, _ in batches)
        assert total_instances == 500


class TestBatchConsistency:
    def test_all_batches_same_size(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=75,
            anomaly_proportion=0.1,
            n_batches=10,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert len(x_batch) == 75
            assert len(y_batch) == 75

    def test_proportional_mode_exact_count_every_batch(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.2,
            n_batches=20,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 10
