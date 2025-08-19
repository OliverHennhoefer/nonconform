import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.bootstrap import Bootstrap
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseBootstrapConformalWeighted(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.067)
        self.assertEqual(power, 0.98)

    def test_bootstrap_conformal_compute_n_calib_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 3419)
        self.assertEqual(fdr, 0.154)
        self.assertEqual(power, 0.99)

    def test_bootstrap_conformal_compute_resampling_ratio_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 1000)
        self.assertEqual(fdr, 0.067)
        self.assertEqual(power, 0.98)


if __name__ == "__main__":
    unittest.main()
