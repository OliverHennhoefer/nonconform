import numpy as np
from oddball import Dataset, load
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import (
    ConformalDetector,
    CrossValidation,
    JackknifeBootstrap,
    Probabilistic,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
)


class TestStandardProbabilistic:
    def test_split(self):
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.2
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.105, decimal=3
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.94, decimal=2
        )

    def test_jackknife(self):
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(plus=False),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.25
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, decimal=2
        )
        # Note: Original expectation (0.666) was based on a parameter-swap bug in
        # old Jackknife class. The correct seeded behavior yields 0.333.
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.333, decimal=3
        )

    def test_jackknife_bootstrap(self):
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=ECOD(),
            strategy=JackknifeBootstrap(n_bootstraps=100),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        estimates = ce.predict(x_test)
        decisions = false_discovery_control(estimates, method="bh") <= 0.1
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0588, decimal=3
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.17, decimal=2
        )
