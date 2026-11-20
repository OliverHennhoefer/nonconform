"""Integration tests that exercise ConformalDetector with PyOD and custom models."""

from __future__ import annotations

import logging
import sys
from collections import namedtuple

import numpy as np
import pytest

from nonconform import ConformalDetector, Split

try:
    from pyod.models.abod import ABOD
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest
    from pyod.models.inne import INNE
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.pca import PCA

    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False
    ABOD = COPOD = ECOD = IForest = INNE = KNN = LOF = PCA = None

SEED = 5
MIN_CONTAMINATION = sys.float_info.min

DetectorCase = namedtuple(
    "DetectorCase", "name factory expects_random_state expects_n_jobs"
)


def _split_detector(detector, *, score_polarity="auto"):
    # Coverage intent:
    # - default "auto" validates polarity inference for real PyOD detectors
    # - explicit polarity is used in custom detector tests where inference
    #   is intentionally not expected to work
    return ConformalDetector(
        detector=detector,
        strategy=Split(n_calib=0.2),
        aggregation="mean",
        score_polarity=score_polarity,
        seed=SEED,
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
        "abod",
        lambda seed: ABOD(method="fast", contamination=0.05, n_neighbors=8),
        False,
        True,
    ),
    DetectorCase(
        "inne",
        lambda seed: INNE(contamination=0.05, random_state=seed),
        True,
        False,
    ),
    DetectorCase(
        "pca",
        lambda seed: PCA(contamination=0.05),
        True,
        False,
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
    p_values = detector.compute_p_values(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all(np.isfinite(p_values))
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    assert len(detector.calibration_set) > 0

    fitted = detector.detector_set[0]
    assert fitted is not base_detector  # Defensive copy
    # Contamination is set to float min for compatibility, also when given.
    assert np.isclose(fitted.contamination, MIN_CONTAMINATION)

    if case.expects_random_state and hasattr(fitted, "random_state"):
        assert fitted.random_state == detector.seed
    if case.expects_n_jobs and hasattr(fitted, "n_jobs"):
        assert fitted.n_jobs == -1


@pytest.mark.skipif(not HAS_PYOD, reason="PyOD not installed")
@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("ECOD", lambda seed: ECOD(contamination=0.05)),
        ("COPOD", lambda seed: COPOD(contamination=0.05)),
    ],
)
def test_blocked_pyod_detectors_raise_on_initialization(name, factory):
    """Blocked batch-adaptive PyOD detectors fail fast."""
    with pytest.raises(
        ValueError,
        match=(
            rf"{name}.*strict inductive conformal/FDR workflows.*"
            r"fixed training-only score map"
        ),
    ):
        _split_detector(factory(seed=17))


@pytest.mark.skipif(not HAS_PYOD, reason="PyOD not installed")
def test_auto_encoder_detector(simple_dataset):
    """Neural detectors should also integrate when optional deps are installed."""
    pytest.importorskip("torch")
    from pyod.models.auto_encoder import AutoEncoder

    x_train, x_test, _ = simple_dataset(n_train=60, n_test=24, n_features=3)
    base = AutoEncoder(
        epoch_num=1,
        batch_size=16,
        hidden_neuron_list=[8, 4, 8],
        verbose=0,
        contamination=0.05,
    )
    detector = _split_detector(base, score_polarity="higher_is_anomalous")
    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)
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


class DeterministicDetector:
    """Deterministic detector without any seed parameter."""

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.ones(len(X))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


@pytest.mark.parametrize(
    ("make_detector", "expected_params"),
    [
        (
            lambda: CustomSklearnDetector(
                contamination=0.05, random_state=42, n_jobs=1
            ),
            {"contamination": MIN_CONTAMINATION, "random_state": SEED, "n_jobs": -1},
        ),
        (
            lambda: CustomDetectorWithAliases(seed=42, n_threads=1),
            {"seed": SEED, "n_threads": -1},
        ),
    ],
    ids=["sklearn-style-names", "parameter-aliases"],
)
def test_custom_detector_parameters_are_configured(
    simple_dataset, make_detector, expected_params
):
    """Custom detectors with sklearn-style names or aliases should work."""
    x_train, x_test, _ = simple_dataset(n_train=64, n_test=30, n_features=4)
    detector = _split_detector(make_detector(), score_polarity="higher_is_anomalous")

    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)

    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))
    assert detector.detector_set[0].get_params() == expected_params


def test_detector_missing_random_state(caplog, monkeypatch):
    """A missing random_state param is logged at debug level, not as a warning."""
    monkeypatch.setattr(logging.getLogger("nonconform"), "propagate", True)

    with caplog.at_level(logging.DEBUG, logger="nonconform"):
        ConformalDetector(
            detector=DeterministicDetector(),
            strategy=Split(n_calib=0.2),
            score_polarity="higher_is_anomalous",
            seed=42,
        )

    assert "random_state" in caplog.text
    assert "Reproducibility cannot be guaranteed" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
