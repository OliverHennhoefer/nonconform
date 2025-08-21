import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.cross_val import CrossValidation
from nonconform.utils.data import load_mammography, load_shuttle, load_thyroid
from nonconform.utils.data.load import load_fraud, load_musk
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.115)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.77)

    def test_cross_val_conformal_plus_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.141)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.79)

    def test_cross_val_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.168)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.99)

    def test_cross_val_conformal_plus_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.147)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.99)

    def test_cross_val_conformal_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.23)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.934)

    def test_cross_val_conformal_plus_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.149)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.934)

    def test_cross_val_conformal_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.333)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.12)

    def test_cross_val_conformal_plus_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.133)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.13)

    def test_cross_val_conformal_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=HBOS(),
            strategy=CrossValidation(k=5),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.183)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 1.0)

    def test_cross_val_conformal_plus_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=HBOS(),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.183)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 1.0)


if __name__ == "__main__":
    unittest.main()
