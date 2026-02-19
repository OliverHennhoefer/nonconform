from __future__ import annotations

import numpy as np
import pytest

import nonconform.adapters as adapters
from nonconform.enums import ScorePolarity


class _MockDetector:
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _MockDetector:
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X), dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    def set_params(self, **params: object) -> _MockDetector:
        return self


class _DynamicMethodsOnly:
    def __getattr__(self, name: str):
        if name in {"fit", "decision_function", "get_params", "set_params"}:
            return lambda *args, **kwargs: self
        raise AttributeError(name)


def test_adapt_returns_object_with_required_dynamic_methods() -> None:
    detector = _DynamicMethodsOnly()
    adapted = adapters.adapt(detector)
    assert adapted is detector


def test_score_polarity_adapter_rejects_auto_value() -> None:
    with pytest.raises(ValueError, match="explicit non-auto"):
        adapters.ScorePolarityAdapter(_MockDetector(), ScorePolarity.AUTO)


def test_score_polarity_adapter_uninitialized_getattr_raises() -> None:
    raw = object.__new__(adapters.ScorePolarityAdapter)
    with pytest.raises(AttributeError, match="missing"):
        raw.__getattr__("missing")


def test_score_polarity_adapter_repr_includes_polarity_name() -> None:
    adapter = adapters.ScorePolarityAdapter(
        detector=_MockDetector(),
        score_polarity=ScorePolarity.HIGHER_IS_NORMAL,
    )
    repr_str = repr(adapter)
    assert "ScorePolarityAdapter" in repr_str
    assert "HIGHER_IS_NORMAL" in repr_str


def test_score_polarity_adapter_getattr_delegates_to_wrapped_detector() -> None:
    detector = _MockDetector()
    detector.custom_attr = "delegated"
    adapter = adapters.ScorePolarityAdapter(
        detector=detector,
        score_polarity=ScorePolarity.HIGHER_IS_NORMAL,
    )
    assert adapter.custom_attr == "delegated"


def test_pyod_adapter_raises_when_pyod_marked_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", False)
    with pytest.raises(ImportError, match="PyOD is not installed"):
        adapters.PyODAdapter(_MockDetector())


def test_adapt_raises_for_pyod_like_object_when_pyod_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePyODLike:
        pass

    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", False)
    monkeypatch.setattr(adapters, "_looks_like_pyod", lambda _: True)

    with pytest.raises(ImportError, match="appears to be a PyOD detector"):
        adapters.adapt(FakePyODLike())


def test_adapt_wraps_pyod_base_detector_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBaseDetector:
        pass

    class FakePyODDetector(FakeBaseDetector):
        pass

    monkeypatch.setattr(adapters, "PYOD_AVAILABLE", True)
    monkeypatch.setattr(adapters, "PyODBaseDetector", FakeBaseDetector)

    adapted = adapters.adapt(FakePyODDetector())
    assert isinstance(adapted, adapters.PyODAdapter)
