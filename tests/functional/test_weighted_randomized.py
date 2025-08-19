import unittest

from scipy.stats import false_discovery_control

from nonconform.estimation.weighted_conformal import WeightedConformalDetector
from nonconform.strategy.randomized import Randomized
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.func.enums import Distribution
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseRandomizedConformalWeighted(unittest.TestCase):
    def test_randomized_conformal_n_calib_relative_holdout_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=1_000, holdout_size_range=(0.1, 0.3)),
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

    def test_randomized_conformal_n_iterations_absolute_holdout_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_iterations=15, holdout_size_range=(20, 50)),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 521)
        self.assertEqual(fdr, 0.168)
        self.assertEqual(power, 0.99)

    def test_randomized_conformal_n_calib_small_relative_holdout_weighted(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(n_calib=500, holdout_size_range=(0.15, 0.25)),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 500)
        self.assertEqual(fdr, 0.02)
        self.assertEqual(power, 0.98)

    def test_randomized_conformal_beta_binomial_default_params_weighted(self):
        """Test BETA_BINOMIAL distribution with default parameters (2.0, 5.0).

        Uses default right-skewed beta distribution that favors smaller holdout sizes.
        """
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(
                n_calib=800,
                sampling_distr=Distribution.BETA_BINOMIAL
                # No beta_params - should use default (2.0, 5.0)
            ),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(wce.calibration_set), 800)
        self.assertEqual(fdr, 0.267)
        self.assertEqual(power, 0.99)

    def test_randomized_conformal_plus_weighted(self):
        """Test plus=True mode with weighted conformal and beta distribution.

        Uses ensemble of models with bell-shaped beta distribution (2.0, 2.0).
        """
        x_train, x_test, y_test = load_shuttle(setup=True)

        wce = WeightedConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Randomized(
                n_iterations=10,
                plus=True,
                sampling_distr=Distribution.BETA_BINOMIAL,
                beta_params=(2.0, 2.0)  # Bell-shaped, concentrated around middle
            ),
            seed=1,
        )

        wce.fit(x_train)
        est = wce.predict(x_test)
        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        # Verify ensemble behavior
        self.assertGreater(len(wce.strategy._detector_list), 1)
        self.assertEqual(len(wce.strategy._detector_list), 10)
        self.assertEqual(fdr, 0.084)
        self.assertEqual(power, 0.98)


if __name__ == "__main__":
    unittest.main()
