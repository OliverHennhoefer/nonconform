import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.split import Split
from nonconform.utils.data.load import load_fraud, load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=10_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.134)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.71)

    def test_split_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=10_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.075)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)


if __name__ == "__main__":
    unittest.main()
