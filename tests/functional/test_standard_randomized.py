import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.randomized import Randomized
from nonconform.utils.data.load import load_fraud, load_shuttle
from nonconform.utils.func.enums import Distribution
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseRandomizedConformal(unittest.TestCase):
    def test_randomized_conformal_fraud(self):
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=2_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.119)
        self.assertEqual(power, 0.52)

    def test_randomized_conformal_shuttle(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=1_000),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.175)
        self.assertEqual(power, 0.99)

    def test_randomized_conformal_beta_binomial_custom_params(self):
        """Test BETA_BINOMIAL distribution with custom parameters (5.0, 2.0).

        Uses left-skewed beta distribution that favors larger holdout sizes.
        """
        x_train, x_test, y_test = load_fraud(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(
                n_calib=1_500,
                sampling_distr=Distribution.BETA_BINOMIAL,
                beta_params=(5.0, 2.0),
            ),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(fdr, 0.106)
        self.assertEqual(power, 0.59)

    def test_randomized_conformal_plus_mode(self):
        """Test plus=True mode which uses ensemble of models.

        Verifies that multiple detectors are trained and used for prediction.
        """
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_iterations=8, plus=True),
            seed=1,
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        # Verify ensemble behavior
        self.assertGreater(len(ce.strategy._detector_list), 1)
        self.assertEqual(len(ce.strategy._detector_list), 8)

        self.assertEqual(fdr, 0.132)
        self.assertEqual(power, 0.99)


if __name__ == "__main__":
    unittest.main()
