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
    logistic_weight_estimator,
)
from nonconform.enums import ConformalMode
from nonconform.fdr import weighted_false_discovery_control
from nonconform.metrics import false_discovery_rate, statistical_power

METRIC_ATOL = 5e-3


class TestWeightedEmpirical:
    """Test Weighted Conformalized Selection (WCS) with empirical estimation.

    Note: WCS is more conservative than standard BH procedure because it:
    1. Accounts for covariate shift via weighted conformal p-values
    2. Uses auxiliary p-values and pruning for FDR control
    3. Requires sufficient calibration data for good resolution
    """

    def test_split(self):
        """Test WCS with split conformal on SHUTTLE dataset (non-randomized).

        Note: WCS may be conservative with limited calibration data,
        resulting in fewer discoveries than standard BH.
        """
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Empirical(tie_break="classical"),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.145454545455, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.94, rel=0.0, abs=METRIC_ATOL
        )

    def test_split_randomized(self):
        """Test WCS with split conformal on SHUTTLE dataset (randomized smoothing).

        Uses randomized p-values (Jin & Candes 2023) for tie smoothing.
        """
        x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

        ce = ConformalDetector(
            detector=HBOS(),
            strategy=Split(n_calib=1_000),
            estimation=Empirical(tie_break="randomized"),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
        )

    def test_jackknife(self):
        """Test WCS with jackknife on WBC dataset.

        Note: Small calibration set (106 samples) limits WCS resolution.
        """
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL),
            estimation=Empirical(),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.25)
        # WCS is conservative with small calibration: 0 discoveries
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
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
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.1)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.071428571429, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.13, rel=0.0, abs=METRIC_ATOL
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
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.205357142857, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.89, rel=0.0, abs=METRIC_ATOL
        )
