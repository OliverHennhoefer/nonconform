import numpy as np
from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.strategy import (
    CrossValidation,
    Empirical,
    Jackknife,
    JackknifeBootstrap,
    Split,
)
from nonconform.utils.data import load
from nonconform.utils.func.enums import Dataset
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE


class TestStandardEmpirical:
    def test_split(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.2
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.2, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.94, decimal=2
        )

    def test_jackknife(self):
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=Jackknife(plus=False),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.25
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 1.0, decimal=2
        )

    def test_jackknife_bootstrap(self):
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=ECOD(),
            strategy=JackknifeBootstrap(n_bootstraps=100),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.1
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.071, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.26, decimal=2
        )

    def test_cv(self):
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = ConformalDetector(
            detector=INNE(),
            strategy=CrossValidation(k=5),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.2
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.176, decimal=3
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.89, decimal=2
        )
