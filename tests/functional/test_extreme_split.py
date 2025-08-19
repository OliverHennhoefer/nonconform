import unittest

from online_fdr import BatchStoreyBH
from online_fdr.batching.bh import BatchBH
from online_fdr.batching.prds import BatchPRDS

from nonconform.estimation.extreme_conformal import ExtremeConformalDetector
from nonconform.strategy.bootstrap import Bootstrap
from nonconform.strategy.split import Split
from nonconform.utils.data.generator.batch import BatchGenerator
from nonconform.utils.data.load import load_fraud, load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseExtremeSplit(unittest.TestCase):
    def test_extreme_split_batch_bh_shuttle(self):

        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=300,
            n_batches=10,
            anomaly_proportion=0.13,
            train_size=0.6,
            random_state=42,
        )
        x_train = batch_gen.get_training_data()

        evt_detector = ExtremeConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(n_calib=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            seed=1,
        )

        evt_detector.fit(x_train)

        batch_bh = BatchBH(alpha=0.1)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate()):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_bh.test_batch(p_values)

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.962)
        self.assertEqual(false_discovery_rate(label, decision), 0.038)

    def test_extreme_split_batch_st_bh_shuttle(self):

        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=300,
            n_batches=10,
            anomaly_proportion=0.01,
            train_size=0.6,
            random_state=42,
        )
        x_train = batch_gen.get_training_data()

        evt_detector = ExtremeConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(n_calib=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            seed=1,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchStoreyBH(alpha=0.1, lambda_=0.75)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate()):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values)

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.8)
        self.assertEqual(false_discovery_rate(label, decision), 0.111)

    def test_extreme_split_batch_st_bh_single_anomaly_batch_shuttle(self):

        batch_gen = BatchGenerator(
            load_data_func=load_shuttle,
            batch_size=1000,
            n_batches=10,
            anomaly_proportion=0.001,
            train_size=0.6,
            random_state=42,
        )
        x_train = batch_gen.get_training_data()

        evt_detector = ExtremeConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Split(n_calib=1000),
            evt_threshold_method="percentile",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            seed=1,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchStoreyBH(alpha=0.1, lambda_=0.75)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate()):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values)

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.6)
        self.assertEqual(false_discovery_rate(label, decision), 0.0)

    def test_extreme_split_batch_st_bh_single_anomaly_batch_musk(self):

        batch_gen = BatchGenerator(
            load_data_func=load_fraud,
            batch_size=250,
            n_batches=10,
            anomaly_proportion=0.02,
            train_size=0.6,
            random_state=42,
        )
        x_train = batch_gen.get_training_data()

        evt_detector = ExtremeConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, resampling_ratio=0.995),
            evt_threshold_method="mean_excess",
            evt_threshold_value=0.95,
            evt_min_tail_size=25,
            seed=1,
        )

        evt_detector.fit(x_train)

        batch_st_bh = BatchPRDS(alpha=0.25)

        label = []
        decision = []

        for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate()):

            p_values = evt_detector.predict(x_batch)
            decisions = batch_st_bh.test_batch(p_values)

            label.extend(y_batch.tolist())
            decision.extend(decisions)

        self.assertEqual(statistical_power(label, decision), 0.04)
        self.assertEqual(false_discovery_rate(label, decision), 0.0)


if __name__ == "__main__":
    unittest.main()
