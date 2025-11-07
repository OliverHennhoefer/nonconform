import pytest

from nonconform.utils.data.generator import BatchGenerator

from .conftest import assert_batch_valid, assert_exact_proportion


class TestBatchGeneratorInitialization:
    def test_valid_initialization_proportional(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=20,
            anomaly_proportion=0.1,
            anomaly_mode="proportional",
            seed=42,
        )
        assert gen.batch_size == 20
        assert gen.anomaly_proportion == 0.1
        assert gen.n_anomaly_per_batch == 2
        assert gen.n_normal_per_batch == 18

    def test_valid_initialization_probabilistic(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=20,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        assert gen.batch_size == 20
        assert gen.anomaly_proportion == 0.1
        assert gen.n_batches == 10

    def test_calculated_attributes(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.25,
            seed=42,
        )
        assert gen.n_anomaly_per_batch == 25
        assert gen.n_normal_per_batch == 75
        assert gen.n_anomaly_per_batch + gen.n_normal_per_batch == 100

    @pytest.mark.parametrize(
        "batch_size,proportion,expected_anomalies",
        [
            (100, 0.1, 10),
            (100, 0.5, 50),
            (100, 0.01, 1),
            (150, 0.01, 1),
        ],
    )
    def test_anomaly_calculation(
        self, small_dataset, batch_size, proportion, expected_anomalies
    ):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=batch_size,
            anomaly_proportion=proportion,
            seed=42,
        )
        assert gen.n_anomaly_per_batch == expected_anomalies


class TestBatchSizeValidation:
    def test_batch_size_zero(self, small_dataset):
        with pytest.raises(ValueError, match="must be positive"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=0,
                anomaly_proportion=0.1,
                seed=42,
            )

    def test_batch_size_negative(self, small_dataset):
        with pytest.raises(ValueError, match="must be positive"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=-10,
                anomaly_proportion=0.1,
                seed=42,
            )

    def test_batch_size_one(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=1,
            anomaly_proportion=0.5,
            n_batches=5,
            seed=42,
        )
        assert gen.batch_size == 1


class TestProportionalModeGeneration:
    def test_exact_anomalies_per_batch(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert_batch_valid(x_batch, y_batch, 100)
            assert y_batch.sum() == 10
            assert (y_batch == 0).sum() == 90

    def test_small_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.01,
            n_batches=3,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 1

    def test_half_percent_proportion(self, large_dataset):
        gen = BatchGenerator(
            load_data_func=large_dataset,
            batch_size=200,
            anomaly_proportion=0.005,
            n_batches=2,
            seed=42,
        )
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == 1

    def test_with_n_batches_limit(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        batches = list(gen.generate())
        assert len(batches) == 5

    def test_infinite_generation(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=None,
            seed=42,
        )
        count = 0
        for x_batch, y_batch in gen.generate():
            assert_batch_valid(x_batch, y_batch, 50)
            count += 1
            if count >= 100:
                break
        assert count == 100

    @pytest.mark.parametrize("proportion", [0.0, 0.01, 0.1, 0.25, 0.5])
    def test_different_proportions(self, small_dataset, proportion):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=proportion,
            n_batches=3,
            seed=42,
        )
        expected_anomalies = int(100 * proportion)
        for x_batch, y_batch in gen.generate():
            assert y_batch.sum() == expected_anomalies

    def test_batch_shuffling(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.1,
            n_batches=10,
            seed=42,
        )
        first_positions = []
        for x_batch, y_batch in gen.generate():
            first_positions.append(y_batch.iloc[0])
        assert len(set(first_positions)) > 1


class TestProbabilisticModeGeneration:
    def test_exact_global_proportion(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.05,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        batches = list(gen.generate())
        expected_total = int(50 * 10 * 0.05)
        assert_exact_proportion(batches, expected_total)
        assert len(batches) == 10

    def test_small_proportion_probabilistic(self, large_dataset):
        gen = BatchGenerator(
            load_data_func=large_dataset,
            batch_size=100,
            anomaly_proportion=0.005,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        batches = list(gen.generate())
        assert_exact_proportion(batches, 5)

    def test_variable_batch_counts(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        batch_counts = [y_batch.sum() for _, y_batch in gen.generate()]
        assert len(set(batch_counts)) > 1

    def test_tracking_state_updates(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )
        expected_total = int(50 * 5 * 0.1)
        for _ in gen.generate():
            pass
        assert gen._current_anomalies == expected_total
        assert gen._items_generated == 250


class TestInsufficientDataValidation:
    def test_insufficient_normal_instances(self, tiny_dataset):
        with pytest.raises(ValueError, match="Not enough normal"):
            BatchGenerator(
                load_data_func=lambda: tiny_dataset(n_normal=5, n_anomaly=10),
                batch_size=20,
                anomaly_proportion=0.1,
                seed=42,
            )

    def test_insufficient_anomaly_instances(self, tiny_dataset):
        with pytest.raises(ValueError, match="Not enough anomaly"):
            BatchGenerator(
                load_data_func=lambda: tiny_dataset(n_normal=400, n_anomaly=10),
                batch_size=100,
                anomaly_proportion=0.2,
                seed=42,
            )


class TestBaseGeneratorValidation:
    def test_invalid_anomaly_proportion_negative(self, small_dataset):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=50,
                anomaly_proportion=-0.1,
                seed=42,
            )

    def test_invalid_anomaly_proportion_above_one(self, small_dataset):
        with pytest.raises(ValueError, match="between 0 and 1"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=50,
                anomaly_proportion=1.5,
                seed=42,
            )

    def test_invalid_train_size_zero(self, small_dataset):
        with pytest.raises(ValueError, match="train_size"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=50,
                anomaly_proportion=0.1,
                train_size=0.0,
                seed=42,
            )

    def test_invalid_train_size_one(self, small_dataset):
        with pytest.raises(ValueError, match="train_size"):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=50,
                anomaly_proportion=0.1,
                train_size=1.0,
                seed=42,
            )

    def test_probabilistic_mode_missing_n_batches(self, small_dataset):
        with pytest.raises(ValueError):
            BatchGenerator(
                load_data_func=small_dataset,
                batch_size=50,
                anomaly_proportion=0.1,
                anomaly_mode="probabilistic",
                seed=42,
            )
