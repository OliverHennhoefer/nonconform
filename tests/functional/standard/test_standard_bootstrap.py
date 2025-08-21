import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.experimental.bootstrap import Bootstrap
from nonconform.utils.data import load_fraud, load_mammography, load_musk, load_thyroid
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=15, n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.153)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.72)

    def test_bootstrap_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=15, n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.147)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.99)

    def test_bootstrap_conformal_thyroid(self):
        x_train, x_test, y_test = load_thyroid(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=15, n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.113)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.902)

    def test_bootstrap_conformal_mammography(self):
        x_train, x_test, y_test = load_mammography(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=15, n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.071)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.13)

    def test_bootstrap_conformal_musk(self):
        x_train, x_test, y_test = load_musk(setup=True, seed=1)

        ce = StandardConformalDetector(
            detector=HBOS(),
            strategy=Bootstrap(n_bootstraps=15, n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.169)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 1.0)


if __name__ == "__main__":
    unittest.main()
