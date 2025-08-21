import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.cross_val import CrossValidation
from nonconform.utils.data import load_mammography, load_shuttle, load_thyroid
from nonconform.utils.data.load import load_fraud, load_musk
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.25)

    def test_cross_val_conformal_plus_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.25)

    def test_cross_val_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

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
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

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
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.036)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.869)

    def test_cross_val_conformal_plus_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.038)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.82)

    def test_cross_val_conformal_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.04)

    def test_cross_val_conformal_plus_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.04)

    def test_cross_val_conformal_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=HBOS(),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.0)

    def test_cross_val_conformal_plus_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=HBOS(),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.0)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.0)


if __name__ == "__main__":
    unittest.main()
