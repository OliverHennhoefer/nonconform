import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation import WeightedConformalDetector
from nonconform.strategy import JackknifeBootstrap
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseJackknifeConformal(unittest.TestCase):

    def test_jackknife_bootstrap_conformal_breast(self):
        x_train, x_test, y_test = load_shuttle(setup=True, seed=1)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=JackknifeBootstrap(n_bootstraps=50),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertEqual(false_discovery_rate(y=y_test, y_hat=decisions), 0.067)
        self.assertEqual(statistical_power(y=y_test, y_hat=decisions), 0.98)


if __name__ == "__main__":
    unittest.main()
