"""Unit tests for adapters.py."""

import pickle
from copy import copy, deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

import nonconform.adapters as adapters
from nonconform.adapters import (
    PyODAdapter,
    ScorePolarityAdapter,
    _looks_like_pyod,
    adapt,
)
from nonconform.enums import ScorePolarity
from nonconform.structures import AnomalyDetector

PROTOCOL_METHODS = ("fit", "decision_function", "get_params", "set_params")


class ProtocolDetector:
    """Detector that implements the AnomalyDetector protocol."""

    def __init__(self):
        self.nested = {"key": [1, 2, 3]}

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return np.arange(len(X), dtype=float)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def __repr__(self):
        return "ProtocolDetector()"


class DynamicMethodsDetector:
    """Detector whose protocol methods only exist through ``__getattr__``."""

    def __getattr__(self, name):
        if name in PROTOCOL_METHODS:
            return lambda *args, **kwargs: self
        raise AttributeError(name)


def _returns_self(self, *args, **kwargs):
    return self


def _detector(name="Detector", methods=PROTOCOL_METHODS, module=__name__):
    """Instantiate a detector class with the given methods and module path."""
    namespace = {"__module__": module} | dict.fromkeys(methods, _returns_self)
    return type(name, (), namespace)()


def _normality_adapter(detector):
    return ScorePolarityAdapter(detector, ScorePolarity.HIGHER_IS_NORMAL)


@pytest.fixture
def pyod_available(monkeypatch):
    """Allow PyODAdapter construction independently of the PyOD install."""
    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", True)


@pytest.mark.parametrize(
    "detector",
    [ProtocolDetector(), DynamicMethodsDetector()],
    ids=["protocol-methods", "dynamic-methods"],
)
def test_adapt_passes_compliant_detectors_through(detector):
    assert adapt(detector) is detector


@pytest.mark.parametrize(
    ("methods", "missing"),
    [
        (("fit",), ("decision_function",)),
        (("decision_function",), ("fit",)),
    ],
    ids=["only-fit", "only-decision-function"],
)
def test_adapt_lists_exactly_the_missing_protocol_methods(methods, missing):
    with pytest.raises(TypeError, match="fit\\(\\) and decision_function") as exc_info:
        adapt(_detector(methods=methods))

    assert str(exc_info.value).endswith(f"Missing methods: {', '.join(missing)}.")


def test_adapt_rejects_a_detector_class_with_a_hint():
    with pytest.raises(TypeError, match="Pass the class instance"):
        adapt(ProtocolDetector)


class FitScoreDetector:
    """Detector with fit and decision_function only."""

    def fit(self, X, y=None):
        self.center_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        return np.linalg.norm(np.asarray(X) - self.center_, axis=1)


def test_adapt_wraps_detectors_without_get_and_set_params():
    detector = FitScoreDetector()
    adapted = adapt(detector)
    assert isinstance(adapted, AnomalyDetector)
    assert adapted.get_params() == {}
    adapted.fit(np.zeros((3, 2)))
    np.testing.assert_allclose(adapted.decision_function(np.ones((2, 2))), [2**0.5] * 2)
    with pytest.raises(ValueError, match="has no set_params"):
        adapted.set_params(alpha=1)
    assert deepcopy(adapted).center_.tolist() == [0.0, 0.0]


def test_adapt_wraps_a_scoring_function():
    adapted = adapt(lambda X: np.asarray(X)[:, 0] * 2)
    assert isinstance(adapted, AnomalyDetector)
    assert adapted.fit(np.zeros((2, 2))) is adapted
    np.testing.assert_array_equal(adapted.decision_function([[1, 0], [3, 0]]), [2, 6])


@pytest.mark.parametrize(
    "detector_name",
    ["CD", "COF", "COPOD", "ECOD", "LMDD", "LOCI", "RGraph", "SOD", "SOS"],
)
def test_adapt_rejects_blocked_pyod_detector_names(detector_name):
    """Blocked PyOD detector names fail fast before protocol adaptation."""
    detector = _detector(detector_name, module="pyod.models.synthetic")
    with pytest.raises(
        ValueError,
        match=rf"{detector_name}.*strict inductive conformal/FDR workflows",
    ):
        adapt(detector)


def test_adapt_wraps_pyod_base_detector_subclass(monkeypatch):
    class FakeBaseDetector:
        pass

    class FakePyODDetector(FakeBaseDetector):
        pass

    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", True)
    monkeypatch.setattr(adapters, "PyODBaseDetector", FakeBaseDetector)

    assert isinstance(adapt(FakePyODDetector()), PyODAdapter)


def test_pyod_like_objects_require_pyod(monkeypatch):
    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", False)

    with pytest.raises(ImportError, match="appears to be a PyOD detector"):
        adapt(_detector("IForest", methods=(), module="pyod.models.iforest"))
    with pytest.raises(ImportError, match="PyOD is not installed"):
        PyODAdapter(ProtocolDetector())


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (ProtocolDetector(), False),
        ([1, 2, 3], False),
        ("string", False),
        (42, False),
        (_detector("IForest", module="pyod.models.iforest"), True),
    ],
    ids=["custom-detector", "list", "string", "integer", "pyod-module"],
)
def test_looks_like_pyod_checks_module_path(obj, expected):
    assert _looks_like_pyod(obj) is expected


@pytest.mark.parametrize(
    ("make_adapter", "expected_scores"),
    [(PyODAdapter, [0.1, 0.2, 0.3]), (_normality_adapter, [-0.1, -0.2, -0.3])],
    ids=["pyod", "score-polarity"],
)
def test_adapters_delegate_to_wrapped_detector(
    pyod_available, make_adapter, expected_scores
):
    detector = MagicMock()
    detector.decision_function.return_value = np.array([0.1, 0.2, 0.3])
    detector.get_params.return_value = {"n_estimators": 100}
    detector.custom_attr = "custom_value"
    adapter = make_adapter(detector)
    X = np.array([[1, 2], [3, 4], [5, 6]])

    assert adapter.fit(X) is adapter
    detector.fit.assert_called_once_with(X, None)
    np.testing.assert_array_equal(adapter.decision_function(X), expected_scores)
    detector.decision_function.assert_called_once_with(X)
    assert adapter.get_params() == {"n_estimators": 100}
    detector.get_params.assert_called_once_with(deep=True)
    assert adapter.set_params(n_estimators=200) is adapter
    detector.set_params.assert_called_once_with(n_estimators=200)
    assert adapter.custom_attr == "custom_value"


@pytest.mark.parametrize(
    ("make_adapter", "expected"),
    [
        (PyODAdapter, "PyODAdapter(ProtocolDetector())"),
        (
            _normality_adapter,
            "ScorePolarityAdapter(score_polarity=HIGHER_IS_NORMAL, "
            "detector=ProtocolDetector())",
        ),
    ],
    ids=["pyod", "score-polarity"],
)
def test_adapter_repr_names_wrapped_detector(pyod_available, make_adapter, expected):
    assert repr(make_adapter(ProtocolDetector())) == expected


@pytest.mark.parametrize(
    "make_adapter", [PyODAdapter, _normality_adapter], ids=["pyod", "score-polarity"]
)
def test_adapter_copies_wrap_copied_detectors(pyod_available, make_adapter):
    detector = ProtocolDetector()
    adapter = make_adapter(detector)
    X = np.zeros((3, 2))

    shallow, deep = copy(adapter), deepcopy(adapter)

    for duplicate in (shallow, deep):
        assert type(duplicate) is type(adapter)
        assert duplicate is not adapter
        assert duplicate._detector is not detector
        np.testing.assert_array_equal(
            duplicate.decision_function(X), adapter.decision_function(X)
        )
    detector.nested["key"].append(4)
    assert 4 not in deep._detector.nested["key"]


@pytest.mark.parametrize("adapter_cls", [PyODAdapter, ScorePolarityAdapter])
def test_uninitialized_adapter_getattr_raises(adapter_cls):
    raw = object.__new__(adapter_cls)
    with pytest.raises(AttributeError, match="missing"):
        raw.__getattr__("missing")


def test_real_pyod_detector_is_protocol_compliant_after_adapt():
    iforest = pytest.importorskip("pyod.models.iforest")

    adapted = adapt(iforest.IForest(n_estimators=10, random_state=42))

    assert isinstance(adapted, AnomalyDetector)


@pytest.mark.parametrize("fit_before_pickle", [True, False], ids=["fitted", "unfitted"])
def test_pyod_adapter_survives_pickle_roundtrip(fit_before_pickle):
    iforest = pytest.importorskip("pyod.models.iforest")
    X = np.random.default_rng(42).standard_normal((100, 5))
    adapter = PyODAdapter(iforest.IForest(n_estimators=10, random_state=42))
    if fit_before_pickle:
        adapter.fit(X)

    restored = pickle.loads(pickle.dumps(adapter))

    if not fit_before_pickle:
        adapter.fit(X)
        restored.fit(X)
    np.testing.assert_array_equal(
        restored.decision_function(X), adapter.decision_function(X)
    )
