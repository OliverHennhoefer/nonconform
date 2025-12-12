import numpy as np
import pytest

pytest.importorskip("pyod", reason="pyod not installed")
pytest.importorskip("oddball", reason="oddball not installed")

from oddball import Dataset, load
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest

from nonconform import (
    ConformalDetector,
    CrossValidation,
    JackknifeBootstrap,
    Probabilistic,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
    weighted_false_discovery_control,
)


class TestWeightedProbabilistic:
    """Test Weighted Conformalized Selection (WCS) with probabilistic estimation.

    Note: Probabilistic (KDE-based) estimation provides continuous p-values
    which often result in better power than empirical estimation, while
    intentionally dropping the finite-sample guarantee.
    """

    def test_split(self):
        """Test WCS with split conformal on SHUTTLE dataset."""
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.105, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.94, decimal=2
        )

    def test_jackknife(self):
        """Test WCS with jackknife on WBC dataset."""
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(plus=False),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.25)
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.333, decimal=2
        )

    def test_jackknife_bootstrap(self):
        """Test WCS with jackknife+ bootstrap on MAMMOGRAPHY dataset."""
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=ECOD(),
            strategy=JackknifeBootstrap(n_bootstraps=100),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.1)
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.059, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.16, decimal=2
        )
