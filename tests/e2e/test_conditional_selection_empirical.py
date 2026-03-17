import numpy as np
import pytest

pytest.importorskip("pyod", reason="pyod not installed")
pytest.importorskip("oddball", reason="oddball not installed")

from oddball import Dataset, load
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power
from nonconform.scoring import ConditionalEmpirical


def test_conditional_empirical_selection_matches_bh():
    x_train, x_test, y_test = load(Dataset.BREASTW, setup=True, seed=1)

    detector = ConformalDetector(
        detector=IForest(),
        strategy=Split(n_calib=0.2),
        estimation=ConditionalEmpirical(method="simes", delta=0.1),
        seed=1,
    )
    detector.fit(x_train)
    cs_mask = detector.select(x_test, alpha=0.2)
    result = detector.last_result
    assert result is not None and result.p_values is not None

    bh_mask = false_discovery_control(result.p_values, method="bh") <= 0.2

    assert cs_mask.dtype == bool
    assert cs_mask.shape == (len(x_test),)
    np.testing.assert_array_equal(cs_mask, bh_mask)
    assert np.isfinite(false_discovery_rate(y=y_test, y_hat=cs_mask))
    assert np.isfinite(statistical_power(y=y_test, y_hat=cs_mask))
