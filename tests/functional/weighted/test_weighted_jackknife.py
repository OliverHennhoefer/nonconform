import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation import ConformalDetector
from nonconform.estimation.weight import (
    ForestWeightEstimator,
    IdentityWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.strategy.jackknife import Jackknife
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseJackknifeConformal(unittest.TestCase):
    def test_jackknife_conformal_breast_logistic(self):
        x_train, x_test, y_test = load(Dataset.BREAST, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Jackknife(plus=False),
            weight_estimator=LogisticWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.143, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.857, places=3
        )

    def test_jackknife_conformal_plus_breast_forest(self):
        x_train, x_test, y_test = load(Dataset.BREAST, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Jackknife(plus=True),
            weight_estimator=ForestWeightEstimator(seed=1),
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

    def test_jackknife_conformal_breast_identity(self):
        x_train, x_test, y_test = load(Dataset.BREAST, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Jackknife(plus=False),
            weight_estimator=IdentityWeightEstimator(),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.222, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 1.0, places=1
        )


if __name__ == "__main__":
    unittest.main()
