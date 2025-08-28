import unittest

import numpy as np

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.experimental.bootstrap import Bootstrap
from nonconform.utils.data import Dataset, load
from pyod.models.iforest import IForest


class TestBootstrapCallback(unittest.TestCase):
    def test_bootstrap_iteration_callback(self):
        """Test that iteration callback receives correct data."""
        x_train, x_test, _ = load(Dataset.SHUTTLE, setup=True)

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
        ce.fit(x_train, iteration_callback=track_iterations)

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
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True)

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
