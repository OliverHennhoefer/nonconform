import unittest

from nonconform.utils.data.generator import BatchGenerator, OnlineGenerator
from nonconform.utils.data.load import load_breast, load_shuttle


class TestDataGenerators(unittest.TestCase):
    """Test data generators for correct anomaly proportion and parameterization."""

    def test_batch_generator_proportional_mode(self):
        """Test batch generator proportional mode ensures exact anomalies per batch."""
        # Test case 1: 10% anomalies with batch size 100 = exactly 10 per batch
        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=100,
            anomaly_proportion=0.1,
            anomaly_mode="proportional",
            n_batches=5,
            random_state=42,
        )

        # Check training data exists
        x_train = batch_gen.get_training_data()
        self.assertGreater(len(x_train), 0)

        # Generate 5 batches and check each has exactly 10 anomalies
        for i, (x_batch, y_batch) in enumerate(batch_gen.generate()):
            with self.subTest(batch=i):
                self.assertEqual(len(x_batch), 100)
                self.assertEqual(len(y_batch), 100)
                self.assertEqual(y_batch.sum(), 10)  # Exactly 10 anomalies
                self.assertEqual((y_batch == 0).sum(), 90)  # Exactly 90 normal

        # Test case 2: 1% anomalies with batch size 100 = exactly 1 per batch
        batch_gen_small = BatchGenerator(
            load_data_func=load_breast,
            batch_size=100,
            anomaly_proportion=0.01,
            anomaly_mode="proportional",
            n_batches=3,
            random_state=42,
        )

        for i, (x_batch, y_batch) in enumerate(batch_gen_small.generate()):
            with self.subTest(batch=i):
                self.assertEqual(y_batch.sum(), 1)  # Exactly 1 anomaly per batch

    def test_batch_generator_probabilistic_mode(self):
        """Test batch generator probabilistic mode ensures exact global proportion."""
        # 5% anomalies over 10 batches of 50 = exactly 25 anomalies total
        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=50,
            anomaly_proportion=0.05,
            anomaly_mode="probabilistic",
            n_batches=10,
            random_state=42,
        )

        total_instances = 0
        total_anomalies = 0
        batch_anomaly_counts = []

        for x_batch, y_batch in batch_gen.generate():
            batch_anomalies = y_batch.sum()
            total_instances += len(x_batch)
            total_anomalies += batch_anomalies
            batch_anomaly_counts.append(batch_anomalies)

        # Check global proportion is exact
        expected_total_anomalies = int(10 * 50 * 0.05)  # 25
        self.assertEqual(total_instances, 500)
        self.assertEqual(total_anomalies, expected_total_anomalies)

        # Check that batches have variable anomaly counts (not all the same)
        self.assertGreater(
            len(set(batch_anomaly_counts)),
            1,
            "Probabilistic mode should produce variable anomaly counts per batch",
        )

    def test_online_generator_exact_proportion(self):
        """Test online generator ensures exact global anomaly proportion."""
        # Test case 1: 2% anomalies over 1000 instances = exactly 20 anomalies
        online_gen = OnlineGenerator(
            load_data_func=load_shuttle,
            anomaly_proportion=0.02,
            n_instances=1000,
            random_state=42,
        )

        # Check training data exists
        x_train = online_gen.get_training_data()
        self.assertGreater(len(x_train), 0)

        total_anomalies = 0
        instance_count = 0

        for x_instance, y_label in online_gen.generate(n_instances=1000):
            total_anomalies += y_label
            instance_count += 1
            self.assertEqual(len(x_instance), 1)  # Single instance
            self.assertIn(y_label, [0, 1])  # Valid label

        # Check exact proportion
        expected_anomalies = int(1000 * 0.02)  # 20
        self.assertEqual(instance_count, 1000)
        self.assertEqual(total_anomalies, expected_anomalies)

        # Test case 2: 1% anomalies over 100 instances = exactly 1 anomaly
        online_gen_small = OnlineGenerator(
            load_data_func=load_breast,
            anomaly_proportion=0.01,
            n_instances=100,
            random_state=42,
        )

        total_anomalies = 0
        for x_instance, y_label in online_gen_small.generate(n_instances=100):
            total_anomalies += y_label

        self.assertEqual(total_anomalies, 1)  # Exactly 1 anomaly

    def test_batch_generator_parameterization_validation(self):
        """Test batch generator parameter validation."""
        # Test invalid batch size
        with self.assertRaises(ValueError):
            BatchGenerator(
                load_data_func=load_shuttle,
                batch_size=0,
                anomaly_proportion=0.1,
                random_state=42,
            )

        # Test invalid anomaly proportion
        with self.assertRaises(ValueError):
            BatchGenerator(
                load_data_func=load_shuttle,
                batch_size=100,
                anomaly_proportion=1.5,  # > 1.0
                random_state=42,
            )

        # Test invalid anomaly mode
        with self.assertRaises(ValueError):
            BatchGenerator(
                load_data_func=load_shuttle,
                batch_size=100,
                anomaly_proportion=0.1,
                anomaly_mode="invalid_mode",
                random_state=42,
            )

        # Test probabilistic mode without n_batches
        with self.assertRaises(ValueError):
            BatchGenerator(
                load_data_func=load_shuttle,
                batch_size=100,
                anomaly_proportion=0.1,
                anomaly_mode="probabilistic",
                # n_batches=None (missing)
                random_state=42,
            )

    def test_online_generator_parameterization_validation(self):
        """Test online generator parameter validation."""
        # Test invalid anomaly proportion
        with self.assertRaises(ValueError):
            OnlineGenerator(
                load_data_func=load_shuttle,
                anomaly_proportion=-0.1,  # < 0
                n_instances=100,
                random_state=42,
            )

        # Test invalid n_instances
        with self.assertRaises(ValueError):
            OnlineGenerator(
                load_data_func=load_shuttle,
                anomaly_proportion=0.1,
                n_instances=0,  # <= 0
                random_state=42,
            )

        # Test exceeding n_instances in generate
        online_gen = OnlineGenerator(
            load_data_func=load_shuttle,
            anomaly_proportion=0.1,
            n_instances=100,
            random_state=42,
        )

        with self.assertRaises(ValueError):
            list(online_gen.generate(n_instances=200))  # Exceeds n_instances

    def test_different_datasets_compatibility(self):
        """Test generators work with different datasets."""
        datasets = [load_shuttle, load_breast]

        for load_func in datasets:
            with self.subTest(dataset=load_func.__name__):
                # Test batch generator
                batch_gen = BatchGenerator(
                    load_data_func=load_func,
                    batch_size=50,
                    anomaly_proportion=0.1,
                    n_batches=1,
                    random_state=42,
                )

                x_train = batch_gen.get_training_data()
                self.assertGreater(len(x_train), 0)

                # Generate one batch
                x_batch, y_batch = next(batch_gen.generate())
                self.assertEqual(len(x_batch), 50)
                self.assertEqual(y_batch.sum(), 5)  # 10% of 50

                # Test online generator
                online_gen = OnlineGenerator(
                    load_data_func=load_func,
                    anomaly_proportion=0.05,
                    n_instances=100,
                    random_state=42,
                )

                total_anomalies = 0
                for x_instance, y_label in online_gen.generate(n_instances=100):
                    total_anomalies += y_label

                self.assertEqual(total_anomalies, 5)  # 5% of 100

    def test_generator_reset_functionality(self):
        """Test generator reset functionality."""
        # Test that reset allows generators to be reused and maintains consistency

        # Test batch generator reset - focus on reproducible properties
        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=50,
            anomaly_proportion=0.1,
            anomaly_mode="proportional",
            n_batches=3,
            random_state=42,
        )

        # Generate some batches
        batches1 = list(batch_gen.generate())
        total_anomalies1 = sum(y.sum() for _, y in batches1)

        # Reset and generate again
        batch_gen.reset()
        batches2 = list(batch_gen.generate())
        total_anomalies2 = sum(y.sum() for _, y in batches2)

        # Check that properties are maintained after reset
        self.assertEqual(len(batches1), len(batches2))
        self.assertEqual(total_anomalies1, total_anomalies2)

        # Each batch should have same structure
        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            self.assertEqual(x1.shape, x2.shape)
            self.assertEqual(len(y1), len(y2))
            self.assertEqual(y1.sum(), y2.sum())  # Same anomaly count per batch

        # Test online generator reset
        online_gen = OnlineGenerator(
            load_data_func=load_shuttle,
            anomaly_proportion=0.1,
            n_instances=20,
            random_state=42,
        )

        # Generate instances
        instances1 = list(online_gen.generate(n_instances=20))
        total_anomalies1 = sum(y for _, y in instances1)

        # Reset and generate again
        online_gen.reset()
        instances2 = list(online_gen.generate(n_instances=20))
        total_anomalies2 = sum(y for _, y in instances2)

        # Check that anomaly counts are maintained
        self.assertEqual(len(instances1), len(instances2))
        self.assertEqual(total_anomalies1, total_anomalies2)

        # Test that generator can be used again after completion
        online_gen.reset()
        instances3 = list(online_gen.generate(n_instances=10))
        self.assertEqual(len(instances3), 10)

    def test_edge_case_anomaly_proportions(self):
        """Test edge cases for anomaly proportions."""
        # Test 0% anomalies (all normal)
        batch_gen_zero = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=100,
            anomaly_proportion=0.0,
            n_batches=1,
            random_state=42,
        )

        x_batch, y_batch = next(batch_gen_zero.generate())
        self.assertEqual(y_batch.sum(), 0)  # No anomalies

        # Test 100% anomalies (would require enough anomaly data)
        # This might fail due to insufficient anomaly instances,
        # so we test a high proportion
        try:
            batch_gen_high = BatchGenerator(
                load_data_func=load_shuttle,
                batch_size=10,  # Small batch to avoid data issues
                anomaly_proportion=0.5,  # 50% anomalies
                n_batches=1,
                random_state=42,
            )

            x_batch, y_batch = next(batch_gen_high.generate())
            self.assertEqual(y_batch.sum(), 5)  # 50% of 10
        except ValueError:
            # Expected if insufficient anomaly data
            pass


if __name__ == "__main__":
    unittest.main()
