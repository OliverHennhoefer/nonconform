import pytest

pytest.importorskip("pyod", reason="pyod not installed")
pytest.importorskip("oddball", reason="oddball not installed")

from oddball import Dataset, load
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.knn import KNN

from nonconform import (
    ConformalDetector,
    CrossValidation,
    Empirical,
    JackknifeBootstrap,
    Split,
)
from nonconform.enums import ConformalMode
from nonconform.metrics import false_discovery_rate, statistical_power

METRIC_ATOL = 5e-3


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
        decisions = ce.select(x_test, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.203389830508, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.94, rel=0.0, abs=METRIC_ATOL
        )

    def test_jackknife(self):
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        ce = ConformalDetector(
            detector=IForest(),
            strategy=CrossValidation.jackknife(mode=ConformalMode.SINGLE_MODEL),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        decisions = ce.select(x_test, alpha=0.25)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.0, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            1.0, rel=0.0, abs=METRIC_ATOL
        )

    def test_jackknife_bootstrap(self):
        x_train, x_test, y_test = load(Dataset.MAMMOGRAPHY, setup=True, seed=1)

        ce = ConformalDetector(
            detector=KNN(method="mean", n_neighbors=7),
            strategy=JackknifeBootstrap(n_bootstraps=100),
            estimation=Empirical(),
            seed=1,
        )

        ce.fit(x_train)
        decisions = ce.select(x_test, alpha=0.1)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.076923076923, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.24, rel=0.0, abs=METRIC_ATOL
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
        decisions = ce.select(x_test, alpha=0.2)
        assert false_discovery_rate(y=y_test, y_hat=decisions) == pytest.approx(
            0.175925925926, rel=0.0, abs=METRIC_ATOL
        )
        assert statistical_power(y=y_test, y_hat=decisions) == pytest.approx(
            0.89, rel=0.0, abs=METRIC_ATOL
        )
