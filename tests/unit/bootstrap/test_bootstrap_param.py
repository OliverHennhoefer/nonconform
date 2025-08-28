import unittest

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.experimental.bootstrap import Bootstrap
from nonconform.utils.data import Dataset, load
from pyod.models.iforest import IForest


class TestCaseBootstrapConformal(unittest.TestCase):
    def test_bootstrap_conformal_compute_n_bootstraps(self):
        x_train, _, _ = load(Dataset.SHUTTLE, setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.995, n_calib=1_000),
            seed=1,
        )

        ce.fit(x_train)

        self.assertEqual(len(ce.calibration_set), 1_000)
        nb = round(
            ce.strategy._n_calib / (len(x_train) * (1 - ce.strategy._resampling_ratio)),
            0,
        )
        self.assertEqual(ce.strategy._n_bootstraps, nb)

    def test_bootstrap_conformal_compute_n_calib(self):
        x_train, _, _ = load(Dataset.SHUTTLE, setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(resampling_ratio=0.99, n_bootstraps=15),
            seed=1,
        )

        ce.fit(x_train)

        self.assertEqual(len(ce.calibration_set), 3419)
        rs = round((len(x_train) * (1 - 0.99)) * 15, 0)
        self.assertEqual(ce.strategy._n_calib, rs)

    def test_bootstrap_conformal_compute_resampling_ratio(self):
        x_train, _, _ = load(Dataset.SHUTTLE, setup=True)

        ce = StandardConformalDetector(
            detector=IForest(behaviour="new"),
            strategy=Bootstrap(n_calib=1_000, n_bootstraps=25),
            seed=1,
        )

        ce.fit(x_train)

        self.assertEqual(len(ce.calibration_set), 1000)
        nb = 1 - (ce.strategy._n_calib / (ce.strategy._n_bootstraps * len(x_train)))
        self.assertEqual(ce.strategy._resampling_ratio, nb)


if __name__ == "__main__":
    unittest.main()
