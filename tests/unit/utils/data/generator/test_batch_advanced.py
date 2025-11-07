import numpy as np
import pandas as pd

from nonconform.utils.data.generator import BatchGenerator


class TestResetFunctionality:
    def test_reset_proportional_mode(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        batches1 = list(gen.generate())
        gen.reset()
        batches2 = list(gen.generate())

        assert len(batches1) == len(batches2)
        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            assert x1.shape == x2.shape
            assert y1.sum() == y2.sum()
            assert len(y1) == len(y2)

    def test_reset_probabilistic_mode(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        total1 = sum(y.sum() for _, y in gen.generate())
        gen.reset()
        total2 = sum(y.sum() for _, y in gen.generate())

        assert total1 == total2
        assert total1 == int(50 * 10 * 0.1)

    def test_reset_clears_tracking_state(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )
        list(gen.generate())
        assert gen._current_anomalies > 0
        assert gen._items_generated > 0

        gen.reset()
        assert gen._current_anomalies == 0
        assert gen._items_generated == 0

    def test_multiple_resets(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.2,
            n_batches=3,
            seed=42,
        )
        results = []
        for _ in range(3):
            batches = list(gen.generate())
            total_anomalies = sum(y.sum() for _, y in batches)
            results.append(total_anomalies)
            gen.reset()

        assert all(r == results[0] for r in results)

    def test_reset_mid_generation(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=10,
            seed=42,
        )
        partial = []
        for i, batch in enumerate(gen.generate()):
            partial.append(batch)
            if i == 4:
                break

        gen.reset()
        full = list(gen.generate())
        assert len(full) == 10


class TestReproducibility:
    def test_same_seed_same_results(self, small_dataset):
        gen1 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        gen2 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )

        batches1 = list(gen1.generate())
        batches2 = list(gen2.generate())

        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            pd.testing.assert_frame_equal(x1, x2)
            pd.testing.assert_series_equal(y1, y2)

    def test_same_seed_probabilistic_mode(self, small_dataset):
        gen1 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        gen2 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )

        batches1 = list(gen1.generate())
        batches2 = list(gen2.generate())

        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            pd.testing.assert_frame_equal(x1, x2)
            pd.testing.assert_series_equal(y1, y2)

    def test_different_seeds_different_results(self, small_dataset):
        gen1 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        gen2 = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=123,
        )

        batches1 = list(gen1.generate())
        batches2 = list(gen2.generate())

        differences = 0
        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            if not x1.equals(x2) or not y1.equals(y2):
                differences += 1

        assert differences > 0

    def test_reset_maintains_reproducibility(self, small_dataset):
        df = small_dataset(seed=999)
        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.2,
            anomaly_mode="proportional",
            n_batches=3,
            seed=999,
        )

        batches1 = list(gen.generate())
        total1 = sum(y.sum() for _, y in batches1)

        gen.reset()
        batches2 = list(gen.generate())
        total2 = sum(y.sum() for _, y in batches2)

        gen.reset()
        batches3 = list(gen.generate())
        total3 = sum(y.sum() for _, y in batches3)

        assert total1 == total2 == total3 == 30
        assert len(batches1) == len(batches2) == len(batches3) == 3

        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            assert y1.sum() == y2.sum()
            assert len(y1) == len(y2) == 50


class TestTrainingDataAccess:
    def test_get_training_data(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            seed=42,
        )
        x_train = gen.get_training_data()
        assert isinstance(x_train, pd.DataFrame)
        assert len(x_train) > 0
        assert "Class" not in x_train.columns

    def test_training_data_size(self, small_dataset):
        df = small_dataset(n_normal=100, n_anomaly=20)
        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            train_size=0.5,
            seed=42,
        )
        x_train = gen.get_training_data()
        assert len(x_train) == 50

    def test_training_data_only_normal(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            seed=42,
        )
        x_train = gen.get_training_data()
        assert len(x_train) > 0


class TestProbabilisticTracking:
    def test_tracking_increments_correctly(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )
        assert gen._current_anomalies == 0
        assert gen._items_generated == 0

        for i, (_, y_batch) in enumerate(gen.generate(), 1):
            expected_items = i * 50
            assert gen._items_generated == expected_items

    def test_tracking_matches_target(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=10,
            seed=42,
        )
        expected_target = int(50 * 10 * 0.1)
        assert gen._target_anomalies == expected_target

        list(gen.generate())
        assert gen._current_anomalies == expected_target

    def test_should_generate_anomaly_logic(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=10,
            anomaly_proportion=0.5,
            anomaly_mode="probabilistic",
            n_batches=2,
            seed=42,
        )
        list(gen.generate())
        assert gen._current_anomalies == 10
        assert gen._items_generated == 20

    def test_last_batch_adjustment(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=100,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=999,
        )
        batches = list(gen.generate())
        total_anomalies = sum(y.sum() for _, y in batches)
        assert total_anomalies == int(100 * 5 * 0.1)


class TestDataPreparationIntegration:
    def test_class_column_removed_from_batches(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=3,
            seed=42,
        )
        for x_batch, _ in gen.generate():
            assert "Class" not in x_batch.columns

    def test_labels_binary(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=5,
            seed=42,
        )
        for _, y_batch in gen.generate():
            assert set(y_batch.unique()).issubset({0, 1})

    def test_feature_values_from_dataset(self, small_dataset):
        df = small_dataset()
        original_values = df.drop(columns=["Class"]).values.flatten()

        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=1,
            seed=42,
        )
        x_batch, _ = next(gen.generate())
        batch_values = x_batch.values.flatten()

        for val in batch_values:
            assert np.any(np.isclose(original_values, val))


class TestStateConsistency:
    def test_n_normal_n_anomaly_counts(self, small_dataset):
        df = small_dataset(n_normal=100, n_anomaly=20, seed=123)
        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            train_size=0.5,
            seed=123,
        )
        assert gen.n_normal == 50
        assert gen.n_anomaly == 20

    def test_rng_state_differs_after_generation(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=1,
            seed=42,
        )
        initial_state = gen.rng.bit_generator.state
        next(gen.generate())
        final_state = gen.rng.bit_generator.state
        assert initial_state != final_state

    def test_rng_state_resets(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.1,
            n_batches=1,
            seed=42,
        )
        gen.reset()
        state1 = gen.rng.bit_generator.state
        gen.reset()
        state2 = gen.rng.bit_generator.state
        assert state1 == state2


class TestEdgeCaseIntegration:
    def test_zero_anomalies_with_probabilistic(self, small_dataset):
        gen = BatchGenerator(
            load_data_func=small_dataset,
            batch_size=50,
            anomaly_proportion=0.0,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )
        total = sum(y.sum() for _, y in gen.generate())
        assert total == 0

    def test_all_anomalies_if_data_supports(self, small_dataset):
        df = small_dataset(n_normal=100, n_anomaly=100)
        gen = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=1.0,
            n_batches=3,
            seed=42,
        )
        for _, y_batch in gen.generate():
            assert y_batch.sum() == 50

    def test_mixed_modes_same_dataset(self, small_dataset):
        df = small_dataset()

        gen_prop = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="proportional",
            n_batches=5,
            seed=42,
        )
        gen_prob = BatchGenerator(
            load_data_func=lambda: df,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="probabilistic",
            n_batches=5,
            seed=42,
        )

        total_prop = sum(y.sum() for _, y in gen_prop.generate())
        total_prob = sum(y.sum() for _, y in gen_prob.generate())

        assert total_prop == total_prob == int(50 * 5 * 0.1)
