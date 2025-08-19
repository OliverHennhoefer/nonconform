import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.split import Split
from nonconform.utils.data.load import load_fraud, load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_split_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=2_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.134)
        self.assertEqual(power, 0.84)

    def test_split_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.25)
        self.assertEqual(power, 0.99)


if __name__ == "__main__":
    unittest.main()
