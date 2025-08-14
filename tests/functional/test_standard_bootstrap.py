import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.bootstrap import Bootstrap
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.075)
        self.assertEqual(power, 0.98)

    def test_bootstrap_conformal_compute_n_calib(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 3419)
        self.assertEqual(fdr, 0.261)
        self.assertEqual(power, 0.99)

    def test_bootstrap_conformal_compute_resampling_ratio(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 1000)
        self.assertEqual(fdr, 0.175)
        self.assertEqual(power, 0.99)


if __name__ == "__main__":
    unittest.main()
