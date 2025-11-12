"""Integration tests that exercise ConformalDetector with several PyOD models."""

from __future__ import annotations

import sys
from collections import namedtuple

import numpy as np
import pytest

from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

try:  # TensorFlow-backed detector is optional/heavy
    from pyod.models.auto_encoder import AutoEncoder

    HAS_AUTO_ENCODER = True
except Exception:  # pragma: no cover - optional dependency missing
    AutoEncoder = None
    HAS_AUTO_ENCODER = False

from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.func.enums import Aggregation

DetectorCase = namedtuple(
    "DetectorCase", "name factory expects_random_state expects_n_jobs"
)


def _split_detector(detector):
    return ConformalDetector(
        detector=detector,
        strategy=Split(n_calib=0.2),
        aggregation=Aggregation.MEAN,
        seed=5,
    )


DETECTOR_CASES = [
    DetectorCase(
        "iforest",
        lambda seed: IForest(n_estimators=30, max_samples=0.7, random_state=seed),
        True,
        True,
    ),
    DetectorCase(
        "lof",
        lambda seed: LOF(n_neighbors=10, contamination=0.05),
        False,
        False,
    ),
    DetectorCase(
        "knn",
        lambda seed: KNN(method="mean", n_neighbors=7, contamination=0.05),
        False,
        False,
    ),
    DetectorCase(
        "ecod",
        lambda seed: ECOD(contamination=0.05),
        False,
        False,
    ),
    DetectorCase(
        "abod",
        lambda seed: ABOD(method="fast", contamination=0.05, n_neighbors=8),
        False,
        False,
    ),
    DetectorCase(
        "copod",
        lambda seed: COPOD(contamination=0.05),
        False,
        True,
    ),
]


@pytest.mark.parametrize("case", DETECTOR_CASES, ids=lambda c: c.name)
def test_pyod_detectors_end_to_end(simple_dataset, case):
    """Every supported detector should fit/predict through ConformalDetector."""
    x_train, x_test, _ = simple_dataset(n_train=64, n_test=30, n_features=4)
    base_detector = case.factory(seed=17)
    detector = _split_detector(base_detector)

    detector.fit(x_train)
    p_values = detector.predict(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all(np.isfinite(p_values))
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    assert len(detector.calibration_set) > 0

    fitted = detector.detector_set[0]
    assert fitted is not base_detector  # Defensive copy

    if case.expects_random_state and hasattr(fitted, "random_state"):
        assert fitted.random_state == detector.seed
    if case.expects_n_jobs and hasattr(fitted, "n_jobs"):
        assert fitted.n_jobs == -1


def test_contamination_parameter_overridden(simple_dataset):
    """Contamination must be set to float min for compatibility."""
    x_train, x_test, _ = simple_dataset(n_train=50, n_test=20, n_features=3)
    base = IForest(contamination=0.15, random_state=0)
    detector = _split_detector(base)

    detector.fit(x_train)
    detector.predict(x_test)

    fitted = detector.detector_set[0]
    assert hasattr(fitted, "contamination")
    assert np.isclose(fitted.contamination, sys.float_info.min)


@pytest.mark.skipif(
    not HAS_AUTO_ENCODER, reason="AutoEncoder dependencies not available"
)
def test_auto_encoder_detector(simple_dataset):
    """Neural detectors should also integrate when optional deps are installed."""
    x_train, x_test, _ = simple_dataset(n_train=60, n_test=24, n_features=3)
    base = AutoEncoder(
        epoch_num=1,
        batch_size=16,
        hidden_neuron_list=[8, 4, 8],
        verbose=0,
        contamination=0.05,
    )
    detector = _split_detector(base)
    detector.fit(x_train)
    p_values = detector.predict(x_test)
    assert np.all((0 <= p_values) & (p_values <= 1))
