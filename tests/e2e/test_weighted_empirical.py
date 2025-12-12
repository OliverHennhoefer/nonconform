import numpy as np
import pytest

pytest.importorskip("pyod", reason="pyod not installed")
pytest.importorskip("oddball", reason="oddball not installed")

from oddball import Dataset, load
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE

from nonconform import (
    ConformalDetector,
    CrossValidation,
    Empirical,
    JackknifeBootstrap,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
    weighted_false_discovery_control,
)


class TestWeightedEmpirical:
    """Test Weighted Conformalized Selection (WCS) with empirical estimation.

    Note: WCS is more conservative than standard BH procedure because it:
    1. Accounts for covariate shift via weighted conformal p-values
    2. Uses auxiliary p-values and pruning for FDR control
    3. Requires sufficient calibration data for good resolution
    """

    def test_split(self):
        """Test WCS with split conformal on SHUTTLE dataset.

        Note: WCS may be conservative with limited calibration data,
        resulting in fewer discoveries than standard BH.
        """
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Empirical(),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        # WCS is conservative: 0 discoveries with this configuration
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.0, decimal=2
        )

    def test_jackknife(self):
        """Test WCS with jackknife on WBC dataset.

        Note: Small calibration set (106 samples) limits WCS resolution.
        """
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(plus=False),
            estimation=Empirical(),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.25)
        # WCS is conservative with small calibration: 0 discoveries
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.0, decimal=2
        )

    def test_jackknife_bootstrap(self):
        """Test WCS with jackknife+ bootstrap on MAMMOGRAPHY dataset."""
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=ECOD(),
            strategy=JackknifeBootstrap(n_bootstraps=100),
            estimation=Empirical(),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.1)
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.0714, decimal=3
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.13, decimal=2
        )

    def test_cv(self):
        """Test WCS with cross-validation on FRAUD dataset."""
        x_train, x_test, y_test = load(Dataset.FRAUD, setup=True, seed=1)

        ce = ConformalDetector(
            detector=INNE(),
            strategy=CrossValidation(k=5),
            estimation=Empirical(),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.predict(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        np.testing.assert_array_almost_equal(
            false_discovery_rate(y=y_test, y_hat=decisions), 0.205, decimal=2
        )
        np.testing.assert_array_almost_equal(
            statistical_power(y=y_test, y_hat=decisions), 0.89, decimal=2
        )
