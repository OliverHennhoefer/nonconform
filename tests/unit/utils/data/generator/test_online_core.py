import pytest

from nonconform.utils.data.generator import OnlineGenerator


class TestOnlineGeneratorInitialization:
    def test_valid_initialization(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        assert gen.anomaly_proportion == 0.1
        assert gen.n_batches == 100
        assert gen.anomaly_mode == "probabilistic"

    def test_initialization_with_train_size(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.05,
            n_instances=200,
            train_size=0.6,
            seed=42,
        )
        assert gen.train_size == 0.6

    def test_n_instances_stored_as_n_batches(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=500,
            seed=42,
        )
        assert gen.n_batches == 500


class TestNInstancesValidation:
    def test_n_instances_zero(self, small_dataset):
        with pytest.raises(ValueError, match="positive"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=0.1,
                n_instances=0,
                seed=42,
            )

    def test_n_instances_negative(self, small_dataset):
        with pytest.raises(ValueError, match="positive"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=0.1,
                n_instances=-100,
                seed=42,
            )

    def test_n_instances_positive(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=1,
            seed=42,
        )
        assert gen.n_batches == 1


class TestExactGlobalProportion:
    def test_one_percent_over_1000(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.01,
            n_instances=1000,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=1000))
        assert total_anomalies == 10

    def test_five_percent_over_200(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.05,
            n_instances=200,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=200))
        assert total_anomalies == 10

    def test_ten_percent_over_100(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        total_anomalies = sum(label for _, label in gen.generate(n_instances=100))
        assert total_anomalies == 10

    @pytest.mark.parametrize(
        "n_instances,proportion,expected",
        [
            (100, 0.02, 2),
            (500, 0.04, 20),
            (1000, 0.005, 5),
            (200, 0.15, 30),
        ],
    )
    def test_various_proportions(
        self, small_dataset, n_instances, proportion, expected
    ):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=proportion,
            n_instances=n_instances,
            seed=42,
        )
        total = sum(label for _, label in gen.generate(n_instances=n_instances))
        assert total == expected


class TestSingleInstanceGeneration:
    def test_yields_single_instance(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=10,
            seed=42,
        )
        for x_instance, y_label in gen.generate(n_instances=10):
            assert len(x_instance) == 1
            assert isinstance(y_label, int)
            assert y_label in [0, 1]

    def test_instance_shape(self, small_dataset):
        df = small_dataset(n_features=7)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.1,
            n_instances=5,
            seed=42,
        )
        for x_instance, _ in gen.generate(n_instances=5):
            assert x_instance.shape == (1, 7)

    def test_label_types(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.5,
            n_instances=20,
            seed=42,
        )
        labels = [label for _, label in gen.generate(n_instances=20)]
        assert all(isinstance(label, int) for label in labels)
        assert set(labels).issubset({0, 1})


class TestDefaultNInstances:
    def test_none_uses_max_instances(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        instances = list(gen.generate(n_instances=None))
        assert len(instances) == 50

    def test_explicit_equals_default(self, small_dataset):
        gen1 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=30,
            seed=42,
        )
        gen2 = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=30,
            seed=42,
        )
        instances1 = list(gen1.generate(n_instances=None))
        instances2 = list(gen2.generate(n_instances=30))
        assert len(instances1) == len(instances2) == 30


class TestExceedingNInstances:
    def test_exceeding_raises_error(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        with pytest.raises(ValueError, match="exceeds n_instances"):
            list(gen.generate(n_instances=200))

    def test_exact_n_instances_allowed(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=50,
            seed=42,
        )
        instances = list(gen.generate(n_instances=50))
        assert len(instances) == 50

    def test_less_than_n_instances_allowed(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        instances = list(gen.generate(n_instances=50))
        assert len(instances) == 50


class TestTrainingDataAccess:
    def test_get_training_data(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        x_train = gen.get_training_data()
        assert len(x_train) > 0
        assert "Class" not in x_train.columns

    def test_training_data_size(self, small_dataset):
        df = small_dataset(n_normal=200, n_anomaly=50)
        gen = OnlineGenerator(
            load_data_func=lambda: df,
            anomaly_proportion=0.1,
            n_instances=100,
            train_size=0.5,
            seed=42,
        )
        x_train = gen.get_training_data()
        assert len(x_train) == 100


class TestBaseGeneratorValidation:
    def test_invalid_anomaly_proportion_negative(self, small_dataset):
        with pytest.raises(ValueError, match="between 0 and 1"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=-0.1,
                n_instances=100,
                seed=42,
            )

    def test_invalid_anomaly_proportion_above_one(self, small_dataset):
        with pytest.raises(ValueError, match="between 0 and 1"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=1.5,
                n_instances=100,
                seed=42,
            )

    def test_invalid_train_size_zero(self, small_dataset):
        with pytest.raises(ValueError, match="train_size"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=0.1,
                n_instances=100,
                train_size=0.0,
                seed=42,
            )

    def test_invalid_train_size_one(self, small_dataset):
        with pytest.raises(ValueError, match="train_size"):
            OnlineGenerator(
                load_data_func=small_dataset,
                anomaly_proportion=0.1,
                n_instances=100,
                train_size=1.0,
                seed=42,
            )


class TestGenerationCounts:
    def test_generates_exact_count(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=75,
            seed=42,
        )
        count = sum(1 for _ in gen.generate(n_instances=75))
        assert count == 75

    def test_partial_generation(self, small_dataset):
        gen = OnlineGenerator(
            load_data_func=small_dataset,
            anomaly_proportion=0.1,
            n_instances=100,
            seed=42,
        )
        count = sum(1 for _ in gen.generate(n_instances=25))
        assert count == 25
