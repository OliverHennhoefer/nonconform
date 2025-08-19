import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.cross_val import CrossValidation
from nonconform.utils.data.load import load_fraud
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseSplitConformalWeighted(unittest.TestCase):
    def test_cross_val_conformal_weighted(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = WeightedConformalDetector(
            detector=IForest(behaviour="new"), strategy=CrossValidation(k=5), seed=1
        )

        ce.fit(x_train)
        est = ce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.0)
        self.assertEqual(power, 0.25)

    def test_cross_val_conformal_plus_weighted(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.0)
        self.assertEqual(power, 0.25)


if __name__ == "__main__":
    unittest.main()
