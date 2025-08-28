import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.cross_val import CrossValidation
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal_fraud(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.22)

    def test_cross_val_conformal_plus_fraud(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.24)

    def test_cross_val_conformal_shuttle(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.109)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)

    def test_cross_val_conformal_plus_shuttle(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.093)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)

    def test_cross_val_conformal_thyroid(self):
        x_train, x_test, y_test = load(Dataset.THYROID, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.067)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.459)


if __name__ == "__main__":
    unittest.main()
