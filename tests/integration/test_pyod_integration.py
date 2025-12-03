"""Integration tests that exercise ConformalDetector with several PyOD models."""

from __future__ import annotations

import sys
from collections import namedtuple

import numpy as np
import pytest

try:
    from pyod.models.abod import ABOD
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False
    ABOD = COPOD = ECOD = IForest = KNN = LOF = None

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


@pytest.mark.skipif(not HAS_PYOD, reason="PyOD not installed")
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


@pytest.mark.skipif(not HAS_PYOD, reason="PyOD not installed")
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


class CustomSklearnDetector:
    """Custom detector implementing AnomalyDetector protocol.

    Uses sklearn-style parameter names for compatibility testing.
    """

    def __init__(self, contamination=0.1, random_state=None, n_jobs=1):
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._mean = None
        self._std = None

    def fit(self, X, y=None):
        """Fit using simple mean/std threshold."""
        rng = np.random.default_rng(self.random_state)
        self._mean = np.mean(X, axis=0) + rng.normal(0, 0.01, X.shape[1])
        self._std = np.std(X, axis=0)
        return self

    def decision_function(self, X):
        """Return Euclidean distance from mean."""
        return np.linalg.norm((X - self._mean) / (self._std + 1e-8), axis=1)

    def get_params(self, deep=True):
        """Get parameters."""
        return {
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CustomDetectorWithAliases:
    """Custom detector using parameter aliases (seed instead of random_state)."""

    def __init__(self, seed=None, n_threads=1):
        self.seed = seed
        self.n_threads = n_threads
        self._threshold = None

    def fit(self, X, y=None):
        """Fit using simple median threshold."""
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, 0.1)
        self._threshold = np.median(np.linalg.norm(X, axis=1)) + noise
        return self

    def decision_function(self, X):
        """Return distance from threshold."""
        return np.abs(np.linalg.norm(X, axis=1) - self._threshold)

    def get_params(self, deep=True):
        """Get parameters."""
        return {"seed": self.seed, "n_threads": self.n_threads}

    def set_params(self, **params):
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def test_custom_sklearn_detector(simple_dataset):
    """Custom detector with sklearn-style parameters should work."""
    x_train, x_test, _ = simple_dataset(n_train=64, n_test=30, n_features=4)
    base = CustomSklearnDetector(contamination=0.05, random_state=42, n_jobs=1)
    detector = _split_detector(base)

    detector.fit(x_train)
    p_values = detector.predict(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    fitted = detector.detector_set[0]
    assert fitted.random_state == detector.seed
    assert fitted.n_jobs == -1
    assert np.isclose(fitted.contamination, sys.float_info.min)


def test_custom_detector_with_aliases(simple_dataset):
    """Custom detector with parameter aliases should work."""
    x_train, x_test, _ = simple_dataset(n_train=64, n_test=30, n_features=4)
    base = CustomDetectorWithAliases(seed=42, n_threads=1)
    detector = _split_detector(base)

    detector.fit(x_train)
    p_values = detector.predict(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    fitted = detector.detector_set[0]
    assert fitted.seed == detector.seed
    assert fitted.n_threads == -1


def test_detector_missing_random_state(caplog, capfd):
    """Detector without random_state param should warn when seed provided."""
    import logging

    # Ensure logger propagates to caplog by temporarily enabling it
    logger = logging.getLogger("nonconform")
    original_propagate = logger.propagate
    logger.propagate = True

    class DeterministicDetector:
        """Deterministic detector without random_state parameter."""

        def fit(self, X, y=None):
            return self

        def decision_function(self, X):
            return np.ones(len(X))

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

    try:
        detector = DeterministicDetector()
        with caplog.at_level(logging.WARNING, logger="nonconform"):
            ConformalDetector(detector=detector, strategy=Split(n_calib=0.2), seed=42)

        # Check caplog records, stderr, and caplog.text for warning
        captured = capfd.readouterr()
        all_output = caplog.text + captured.err
        record_messages = " ".join(r.message for r in caplog.records)
        all_output += record_messages

        assert "random_state" in all_output
        assert "Reproducibility cannot be guaranteed" in all_output
    finally:
        logger.propagate = original_propagate
