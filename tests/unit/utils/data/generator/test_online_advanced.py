import numpy as np
import pandas as pd

from nonconform.utils.data.generator import OnlineGenerator


class TestResetFunctionality:
    def test_reset_clears_tracking_state(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        list(gen.generate(n_instances=50))
        assert gen._current_anomalies > 0
        assert gen._items_generated > 0

        gen.reset()
        assert gen._current_anomalies == 0
        assert gen._items_generated == 0

    def test_reset_allows_regeneration(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        total1 = sum(label for _, label in gen.generate(n_instances=100))
        gen.reset()
        total2 = sum(label for _, label in gen.generate(n_instances=100))

        assert total1 == total2 == 10

    def test_multiple_resets(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.2,
            n_instances=50,
            seed=42,
        )
        results = []
        for _ in range(3):
            total = sum(label for _, label in gen.generate(n_instances=50))
            results.append(total)
            gen.reset()

        assert all(r == results[0] == 10 for r in results)

    def test_reset_mid_generation(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        partial = []
        for i, item in enumerate(gen.generate(n_instances=100)):
            partial.append(item)
            if i == 24:
                break

        gen.reset()
        full = list(gen.generate(n_instances=100))
        assert len(full) == 100


class TestReproducibility:
    def test_same_seed_same_results(self, small_dataset):
        gen1 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        gen2 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )

        instances1 = list(gen1.generate(n_instances=50))
        instances2 = list(gen2.generate(n_instances=50))

        for (x1, y1), (x2, y2) in zip(instances1, instances2):
            pd.testing.assert_frame_equal(x1, x2)
            assert y1 == y2

    def test_different_seeds_different_results(self, small_dataset):
        gen1 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        gen2 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=123,
        )

        instances1 = list(gen1.generate(n_instances=50))
        instances2 = list(gen2.generate(n_instances=50))

        differences = 0
        for (x1, y1), (x2, y2) in zip(instances1, instances2):
            if not x1.equals(x2) or y1 != y2:
                differences += 1

        assert differences > 0

    def test_reset_maintains_reproducibility(self, small_dataset):
        df = small_dataset(seed=999)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.2,
            n_instances=30,
            seed=999,
        )

        instances1 = list(gen.generate(n_instances=30))
        total1 = sum(label for _, label in instances1)

        gen.reset()
        instances2 = list(gen.generate(n_instances=30))
        total2 = sum(label for _, label in instances2)

        gen.reset()
        instances3 = list(gen.generate(n_instances=30))
        total3 = sum(label for _, label in instances3)

        assert total1 == total2 == total3 == int(30 * 0.2)
        assert len(instances1) == len(instances2) == len(instances3) == 30


class TestProbabilisticTracking:
    def test_tracking_increments_correctly(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        assert gen._current_anomalies == 0
        assert gen._items_generated == 0

        for i, (_, label) in enumerate(gen.generate(n_instances=50), 1):
            assert gen._items_generated == i

    def test_tracking_matches_target(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        expected_target = int(100 * 0.1)
        assert gen._target_anomalies == expected_target

        list(gen.generate(n_instances=100))
        assert gen._current_anomalies == expected_target

    def test_tracking_with_partial_generation(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.2,
            n_instances=100,
            seed=42,
        )
        for i, _ in enumerate(gen.generate(n_instances=50), 1):
            pass

        assert gen._items_generated == 50
        assert gen._current_anomalies <= 20

    def test_last_instances_adjustment(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=999,
        )
        instances = list(gen.generate(n_instances=50))
        total_anomalies = sum(label for _, label in instances)
        assert total_anomalies == int(50 * 0.1)


class TestDataPreparationIntegration:
    def test_class_column_removed_from_instances(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=20,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=20):
            assert "Class" not in x_instance.columns

    def test_labels_binary(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.5,
            n_instances=50,
            seed=42,
        )
        for _, label in gen.generate(n_instances=50):
            assert label in [0, 1]

    def test_feature_values_from_dataset(self, small_dataset):
        df = small_dataset()
        original_values = df.drop(columns=["Class"]).values.flatten()

        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=10):
            instance_values = x_instance.values.flatten()
            for val in instance_values:
                assert np.any(np.isclose(original_values, val))


class TestStateConsistency:
    def test_n_normal_n_anomaly_counts(self, small_dataset):
        df = small_dataset(n_normal=100, n_anomaly=20, seed=123)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.1,
            n_instances=50,
            train_size=0.5,
            seed=123,
        )
        assert gen.n_normal == 50
        assert gen.n_anomaly == 20

    def test_rng_state_differs_after_generation(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        initial_state = gen.rng.bit_generator.state
        next(gen.generate(n_instances=10))
        final_state = gen.rng.bit_generator.state
        assert initial_state != final_state

    def test_rng_state_resets(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        gen.reset()
        state1 = gen.rng.bit_generator.state
        gen.reset()
        state2 = gen.rng.bit_generator.state
        assert state1 == state2


class TestPartialGenerationScenarios:
    def test_partial_then_continue_not_tracked_separately(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        for i, _ in enumerate(gen.generate(n_instances=50)):
            if i == 24:
                break

        instances = list(gen.generate(n_instances=50))
        assert len(instances) == 50

    def test_partial_then_reset_works(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        for i, _ in enumerate(gen.generate(n_instances=50)):
            if i == 24:
                break

        gen.reset()
        instances = list(gen.generate(n_instances=100))
        assert len(instances) == 100


class TestSequentialGenerations:
    def test_multiple_full_generations(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.2,
            n_instances=50,
            seed=42,
        )

        total1 = sum(label for _, label in gen.generate(n_instances=50))
        gen.reset()
        total2 = sum(label for _, label in gen.generate(n_instances=50))
        gen.reset()
        total3 = sum(label for _, label in gen.generate(n_instances=50))

        assert total1 == total2 == total3 == int(50 * 0.2)

    def test_different_n_instances_after_reset(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )

        instances1 = list(gen.generate(n_instances=50))
        gen.reset()
        instances2 = list(gen.generate(n_instances=25))

        assert len(instances1) == 50
        assert len(instances2) == 25


class TestEdgeCaseIntegration:
    def test_zero_anomalies_tracking(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.0,
            n_instances=50,
            seed=42,
        )
        list(gen.generate(n_instances=50))
        assert gen._current_anomalies == 0
        assert gen._items_generated == 50

    def test_all_anomalies_if_data_supports(self, large_dataset):
        df = large_dataset(n_normal=100, n_anomaly=100)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=1.0,
            n_instances=30,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=30))
        assert total_anomalies == 30

    def test_very_small_proportion(self, large_dataset):
        gen = OnlineGenerator(
            load_data_func=large_dataset,
            anomaly_proportion=0.0001,
            n_instances=10000,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=10000))
        assert total_anomalies == 1


class TestAlwaysProbabilisticMode:
    def test_mode_is_probabilistic(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        assert gen.anomaly_mode == "probabilistic"

    def test_has_target_anomalies(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.15,
            n_instances=200,
            seed=42,
        )
        assert hasattr(gen, "_target_anomalies")
        assert gen._target_anomalies == int(200 * 0.15)
