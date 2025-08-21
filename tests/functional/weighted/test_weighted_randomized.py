import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.experimental.randomized import Randomized
from nonconform.utils.data import load_mammography, load_thyroid
from nonconform.utils.data.load import load_fraud, load_musk, load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseRandomizedConformal(unittest.TestCase):
    def test_randomized_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=100_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.08)

    def test_randomized_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=100_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.101)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)

    def test_randomized_conformal_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=10_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.057)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.82)

    def test_randomized_conformal_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=ECOD(),
            strategy=Randomized(n_calib=100_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.077)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.12)

    def test_randomized_conformal_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=HBOS(),
            strategy=Randomized(n_calib=10_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.155)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 1.0)


if __name__ == "__main__":
    unittest.main()
