import unittest

import numpy as np
from scipy.stats import false_discovery_control

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.bootstrap import Bootstrap
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 1_000)
        self.assertEqual(fdr, 0.075)
        self.assertEqual(power, 0.98)

    def test_bootstrap_conformal_compute_n_calib(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 3419)
        self.assertEqual(fdr, 0.261)
        self.assertEqual(power, 0.99)

    def test_bootstrap_conformal_compute_resampling_ratio(self):
        x_train, x_test, y_test = load_shuttle(setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
        )

        ce.fit(x_train)
        est = ce.predict(x_test)

        decisions = false_discovery_control(est, method="bh") <= 0.2

        fdr = false_discovery_rate(y=y_test, y_hat=decisions)
        power = statistical_power(y=y_test, y_hat=decisions)

        self.assertEqual(len(ce.calibration_set), 1000)
        self.assertEqual(fdr, 0.175)
        self.assertEqual(power, 0.99)

    def test_bootstrap_iteration_callback(self):
        """Test that iteration callback receives correct data."""
        x_train, x_test, y_test = load_shuttle(setup=True)

        # Track callback invocations
        callback_data = []

        def track_iterations(iteration: int, scores: np.ndarray):
            callback_data.append(
                {
                    "iteration": iteration,
                    "num_scores": len(scores),
                    "mean_score": scores.mean(),
                    "scores_copy": scores.copy(),
                }
            )

        bootstrap = Bootstrap(resampling_ratio=0.9, n_bootstraps=5)
        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=bootstrap,
        )

        # Fit with callback
        ce.detector_set, ce.calibration_set = bootstrap.fit_calibrate(
            x_train, ce.detector, iteration_callback=track_iterations
        )

        # Verify callback was called correct number of times
        self.assertEqual(len(callback_data), 5)

        # Verify iteration numbers are sequential
        for i, data in enumerate(callback_data):
            self.assertEqual(data["iteration"], i)

        # Verify scores are reasonable
        for data in callback_data:
            self.assertGreater(data["num_scores"], 0)
            self.assertIsInstance(data["mean_score"], (float, np.floating))

        # Verify callback doesn't break normal functionality
        est = ce.predict(x_test)
        self.assertEqual(len(est), len(x_test))

    def test_bootstrap_no_callback(self):
        """Test that bootstrap works normally without callback."""
        x_train, x_test, y_test = load_shuttle(setup=True)

        bootstrap = Bootstrap(resampling_ratio=0.9, n_bootstraps=3)
        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=bootstrap,
        )

        # Fit without callback (should work as before)
        ce.fit(x_train)
        est = ce.predict(x_test)

        self.assertEqual(len(est), len(x_test))
        self.assertGreater(len(ce.calibration_set), 0)


if __name__ == "__main__":
    unittest.main()
