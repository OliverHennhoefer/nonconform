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
    logistic_weight_estimator,
)
from nonconform.enums import ConformalMode
from nonconform.fdr import weighted_false_discovery_control
from nonconform.metrics import false_discovery_rate, statistical_power

METRIC_ATOL = 5e-3


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
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.104761904762, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.94, rel=0.0, abs=METRIC_ATOL
        )

    def test_jackknife(self):
        """Test WCS with jackknife on WBC dataset."""
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL),
            estimation=Probabilistic(n_trials=10),
            weight_estimator=logistic_weight_estimator(),
            seed=1,
        )

        ce.fit(x_train)
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.25)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.333333333333, rel=0.0, abs=METRIC_ATOL
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
        ce.compute_p_values(x_test)
        decisions = weighted_false_discovery_control(result=ce.last_result, alpha=0.1)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.058823529412, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.16, rel=0.0, abs=METRIC_ATOL
        )
