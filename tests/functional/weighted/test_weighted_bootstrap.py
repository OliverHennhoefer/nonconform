import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.experimental.bootstrap import Bootstrap
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_fraud(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.95, n_calib=100_000, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, places=1
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.19, places=2
        )

    def test_bootstrap_conformal_shuttle(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.95, n_calib=100_000, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.108, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.99, places=2
        )

    def test_bootstrap_conformal_thyroid(self):
        x_train, x_test, y_test = load(Dataset.THYROID, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=75, n_calib=10_000, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.056, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.836, places=3
        )

    def test_bootstrap_conformal_mammography(self):
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.95, n_calib=100_000, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, places=1
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.04, places=2
        )

    def test_bootstrap_conformal_musk(self):
        x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=HBOS(),
            strategy=Bootstrap(n_bootstraps=25, n_calib=10_000, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.155, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 1.0, places=1
        )


if __name__ == "__main__":
    unittest.main()
