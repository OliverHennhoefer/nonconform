import unittest

from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.detection.weight import (
    ForestWeightEstimator,
    IdentityWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.strategy import Bootstrap, JackknifeBootstrap
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import weighted_false_discovery_control
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_fraud_forest(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.975, n_calib=10_000, plus=True),
            weight_estimator=ForestWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.108, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.74, places=2
        )

    def test_bootstrap_conformal_shuttle_logistic_weighted_fdr(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=JackknifeBootstrap(n_bootstraps=50),
            weight_estimator=LogisticWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)

        scores = ce.predict(x_test, raw=True)
        w_cal, w_test = ce.weight_estimator.get_weights()

        decisions = weighted_false_discovery_control(
            scores, ce.calibration_set, w_test, w_cal, q=0.2, rand="dtm", seed=1
        )

        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, places=2
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.62, places=2
        )

    def test_bootstrap_conformal_thyroid_identity(self):
        x_train, x_test, y_test = load(Dataset.THYROID, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_bootstraps=25, n_calib=1_000, plus=True),
            weight_estimator=IdentityWeightEstimator(),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.08, places=1
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.885, places=3
        )

    def test_bootstrap_conformal_mammography_forest(self):
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=ECOD(),
            strategy=Bootstrap(resampling_ratio=0.99, n_calib=1_000, plus=True),
            weight_estimator=ForestWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0606, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.31, places=2
        )

    def test_bootstrap_conformal_musk_identity(self):
        x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Bootstrap(n_bootstraps=25, n_calib=1_000, plus=True),
            weight_estimator=IdentityWeightEstimator(),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.125, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 1.0, places=1
        )


if __name__ == "__main__":
    unittest.main()
