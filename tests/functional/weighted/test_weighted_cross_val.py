import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation import ConformalDetector
from nonconform.estimation.weight import (
    ForestWeightEstimator,
    IdentityWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.strategy.cross_val import CrossValidation
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import weighted_false_discovery_control
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseSplitConformal(unittest.TestCase):
    def test_cross_val_conformal_fraud_logistic(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=False),
            weight_estimator=LogisticWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.141, places=2
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.79, places=2
        )

    def test_cross_val_conformal_plus_fraud_forest(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
            weight_estimator=ForestWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.153, places=2
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.83, places=2
        )

    def test_cross_val_conformal_shuttle_identity(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=False),
            weight_estimator=IdentityWeightEstimator(),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.168, places=3
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.99, places=2
        )

    def test_cross_val_conformal_plus_shuttle_logistic(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=True),
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
            statistical_power(y=y_test, y_hat=decisions), 0.57, places=2
        )

    def test_cross_val_conformal_thyroid_forest(self):
        x_train, x_test, y_test = load(Dataset.THYROID, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=CrossValidation(k=5, plus=False),
            weight_estimator=ForestWeightEstimator(seed=1),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2
        self.assertAlmostEqual(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.229, places=2
        )
        self.assertAlmostEqual(
            statistical_power(y=y_test, y_hat=decisions), 0.934, places=2
        )


if __name__ == "__main__":
    unittest.main()
