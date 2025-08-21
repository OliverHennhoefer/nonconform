import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.split import Split
from nonconform.utils.data import load_mammography, load_thyroid
from nonconform.utils.data.load import load_fraud, load_musk, load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=2_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.12)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.73)

    def test_split_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.19)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)

    def test_split_conformal_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.069)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.885)

    def test_split_conformal_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=ECOD(), strategy=Split(n_calib=1_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.0)

    def test_split_conformal_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=HBOS(), strategy=Split(n_calib=1_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.1
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.0)


if __name__ == "__main__":
    unittest.main()
