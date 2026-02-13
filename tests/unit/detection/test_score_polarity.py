"""Unit tests for score polarity parsing, inference, and adapters."""

from __future__ import annotations

import importlib.util
from copy import copy, deepcopy
from typing import Any, Self

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from nonconform.adapters import (
    ScorePolarityAdapter,
    apply_score_polarity,
    parse_score_polarity,
    resolve_score_polarity,
)
from nonconform.enums import ScorePolarity


class MockScoreDetector:
    """Simple protocol-compliant detector with deterministic scores."""

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.arange(len(X), dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self


def test_parse_score_polarity_accepts_enum_and_literals() -> None:
    assert parse_score_polarity(ScorePolarity.AUTO) is ScorePolarity.AUTO
    assert (
        parse_score_polarity("higher_is_anomalous") is ScorePolarity.HIGHER_IS_ANOMALOUS
    )
    assert parse_score_polarity("higher_is_normal") is ScorePolarity.HIGHER_IS_NORMAL


def test_parse_score_polarity_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Invalid score_polarity value"):
        parse_score_polarity("invalid")
    with pytest.raises(TypeError, match="score_polarity must be"):
        parse_score_polarity(1)  # type: ignore[arg-type]


def test_apply_score_polarity_keeps_anomalous_scores_unchanged() -> None:
    detector = MockScoreDetector()
    adapted = apply_score_polarity(detector, "higher_is_anomalous")
    assert adapted is detector


def test_apply_score_polarity_flips_normality_scores() -> None:
    detector = MockScoreDetector()
    adapted = apply_score_polarity(detector, ScorePolarity.HIGHER_IS_NORMAL)
    assert isinstance(adapted, ScorePolarityAdapter)
    X = np.zeros((4, 2))
    np.testing.assert_array_equal(
        adapted.decision_function(X),
        np.array([0, -1, -2, -3]),
    )


def test_apply_score_polarity_rejects_unresolved_auto() -> None:
    detector = MockScoreDetector()
    with pytest.raises(ValueError, match="must be resolved first"):
        apply_score_polarity(detector, ScorePolarity.AUTO)


def test_resolve_score_polarity_auto_recognizes_supported_sklearn() -> None:
    iforest = IsolationForest(random_state=1)
    ocsvm = OneClassSVM()
    assert resolve_score_polarity(iforest, "auto") is ScorePolarity.HIGHER_IS_NORMAL
    assert resolve_score_polarity(ocsvm, "auto") is ScorePolarity.HIGHER_IS_NORMAL


def test_resolve_score_polarity_auto_rejects_unknown_detector() -> None:
    detector = MockScoreDetector()
    with pytest.raises(ValueError, match="Unable to infer score polarity"):
        resolve_score_polarity(detector, "auto")


def test_resolve_score_polarity_auto_recognizes_pyod_when_available() -> None:
    if importlib.util.find_spec("pyod") is None:
        pytest.skip("PyOD not installed")

    from pyod.models.iforest import IForest

    detector = IForest(n_estimators=10, random_state=0)
    assert resolve_score_polarity(detector, "auto") is ScorePolarity.HIGHER_IS_ANOMALOUS


def test_score_polarity_adapter_copy_support() -> None:
    detector = MockScoreDetector()
    adapter = ScorePolarityAdapter(detector, ScorePolarity.HIGHER_IS_NORMAL)
    shallow = copy(adapter)
    deep = deepcopy(adapter)

    assert shallow is not adapter
    assert deep is not adapter
    assert isinstance(shallow, ScorePolarityAdapter)
    assert isinstance(deep, ScorePolarityAdapter)

    X = np.zeros((3, 2))
    expected = adapter.decision_function(X)
    np.testing.assert_array_equal(shallow.decision_function(X), expected)
    np.testing.assert_array_equal(deep.decision_function(X), expected)
