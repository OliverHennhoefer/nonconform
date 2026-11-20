"""Pipelines, assumed score polarity, and clear detector errors."""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
import pytest
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from nonconform import ConformalDetector, Split
from nonconform.adapters import (
    adapt,
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
from nonconform.enums import ScorePolarity

POLARITY_WARNING = "score_polarity was not given"


@pytest.fixture
def nonconform_caplog(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("nonconform"), "propagate", True)
    caplog.set_level(logging.INFO, logger="nonconform")
    return caplog


@pytest.fixture
def planted_batch():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(1500, 4))
    x_test = np.vstack([rng.normal(size=(100, 4)), rng.normal(loc=5.0, size=(10, 4))])
    is_anomaly = np.r_[np.zeros(100), np.ones(10)].astype(bool)
    return x_train, x_test, is_anomaly


class AnomalyScorer:
    """Custom detector whose scores are higher for anomalies."""

    def __init__(self) -> None:
        self._params: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(X), axis=1)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self._params.copy()

    def set_params(self, **params: Any) -> Self:
        self._params.update(params)
        return self


class SklearnStyleOutlierDetector(OutlierMixin, BaseEstimator):
    """Third-party sklearn outlier detector that nonconform does not know."""

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        return -np.linalg.norm(np.asarray(X), axis=1)


def scaled_forest() -> Pipeline:
    return make_pipeline(StandardScaler(), IsolationForest(random_state=0))


def test_pipeline_polarity_follows_final_step() -> None:
    assert (
        resolve_implicit_score_polarity(scaled_forest())
        is ScorePolarity.HIGHER_IS_NORMAL
    )
    assert resolve_score_polarity(scaled_forest(), "auto") is (
        ScorePolarity.HIGHER_IS_NORMAL
    )


def test_nested_pipeline_polarity_follows_innermost_final_step() -> None:
    nested = Pipeline([("prep", StandardScaler()), ("model", scaled_forest())])
    assert resolve_score_polarity(nested, "auto") is ScorePolarity.HIGHER_IS_NORMAL


def test_pipeline_with_unknown_final_step_still_fails_in_auto_mode() -> None:
    pipeline = make_pipeline(StandardScaler(), AnomalyScorer())
    with pytest.raises(ValueError, match="AnomalyScorer"):
        resolve_score_polarity(pipeline, "auto")


def test_pipeline_without_polarity_finds_planted_anomalies(planted_batch) -> None:
    x_train, x_test, is_anomaly = planted_batch
    detector = ConformalDetector(scaled_forest(), Split(n_calib=1000), seed=0)
    selected = np.asarray(detector.fit(x_train).select(x_test, alpha=0.1))

    assert detector.score_polarity is ScorePolarity.HIGHER_IS_NORMAL
    assert selected[is_anomaly].sum() == is_anomaly.sum()
    assert selected[~is_anomaly].sum() <= 3


def test_pipeline_matches_explicit_polarity(planted_batch) -> None:
    x_train, x_test, _ = planted_batch

    def p_values(**kwargs):
        detector = ConformalDetector(
            scaled_forest(), Split(n_calib=1000), seed=0, **kwargs
        )
        return detector.fit(x_train).compute_p_values(x_test)

    np.testing.assert_allclose(p_values(), p_values(score_polarity="higher_is_normal"))


def test_pipeline_steps_receive_seed_and_parallelism(nonconform_caplog) -> None:
    detector = ConformalDetector(scaled_forest(), Split(n_calib=0.3), seed=11)

    runtime_params = detector.detector.get_params()
    assert runtime_params["isolationforest__random_state"] == 11
    assert runtime_params["isolationforest__n_jobs"] == -1
    assert "random_state" not in nonconform_caplog.text


def test_pipeline_runs_are_reproducible(planted_batch) -> None:
    x_train, x_test, _ = planted_batch

    def p_values(seed):
        detector = ConformalDetector(scaled_forest(), Split(n_calib=500), seed=seed)
        return detector.fit(x_train).compute_p_values(x_test)

    np.testing.assert_array_equal(p_values(3), p_values(3))


def test_known_detectors_do_not_warn_about_polarity(nonconform_caplog) -> None:
    ConformalDetector(IsolationForest(), Split(n_calib=0.3))
    ConformalDetector(scaled_forest(), Split(n_calib=0.3))
    ConformalDetector(AnomalyScorer(), Split(), score_polarity="higher_is_anomalous")
    assert POLARITY_WARNING not in nonconform_caplog.text


def test_unknown_detector_warns_when_polarity_is_assumed(nonconform_caplog) -> None:
    detector = ConformalDetector(AnomalyScorer(), Split(n_calib=0.3))

    assert detector.score_polarity is ScorePolarity.HIGHER_IS_ANOMALOUS
    warnings = [
        r for r in nonconform_caplog.records if POLARITY_WARNING in r.getMessage()
    ]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "AnomalyScorer" in warnings[0].getMessage()


def test_assumed_polarity_warning_names_pipeline_final_step(
    nonconform_caplog,
) -> None:
    ConformalDetector(make_pipeline(StandardScaler(), AnomalyScorer()), Split())
    assert "Pipeline ending in AnomalyScorer" in nonconform_caplog.text


def test_unknown_sklearn_outlier_detector_gets_convention_hint(
    nonconform_caplog,
) -> None:
    ConformalDetector(SklearnStyleOutlierDetector(), Split())
    assert "higher_is_normal" in nonconform_caplog.text
    assert "scikit-learn outlier detectors" in nonconform_caplog.text


def test_lof_without_novelty_fails_at_construction() -> None:
    with pytest.raises(TypeError, match=r"novelty=True"):
        ConformalDetector(LocalOutlierFactor(), Split())
    with pytest.raises(TypeError, match=r"novelty=True"):
        adapt(make_pipeline(StandardScaler(), LocalOutlierFactor()))


def test_pipeline_ending_in_passthrough_names_missing_method() -> None:
    with pytest.raises(TypeError, match="Missing methods: decision_function"):
        adapt(make_pipeline(StandardScaler(), "passthrough"))


@pytest.mark.parametrize(
    ("make_detector", "score_polarity"),
    [
        (lambda: make_pipeline(StandardScaler(), AnomalyScorer()), None),
        (SklearnStyleOutlierDetector, "higher_is_normal"),
    ],
    ids=["custom-pipeline-default", "sklearn-style-explicit"],
)
def test_custom_detectors_find_planted_anomalies(
    planted_batch, make_detector, score_polarity
) -> None:
    x_train, x_test, is_anomaly = planted_batch
    detector = ConformalDetector(
        make_detector(), Split(n_calib=1000), score_polarity=score_polarity, seed=0
    )
    selected = np.asarray(detector.fit(x_train).select(x_test, alpha=0.1))
    assert selected[is_anomaly].sum() == is_anomaly.sum()


def test_lof_with_novelty_is_accepted() -> None:
    assert adapt(LocalOutlierFactor(novelty=True)) is not None


def test_floor_warning_mentions_polarity(nonconform_caplog, planted_batch) -> None:
    x_train, x_test, _ = planted_batch
    detector = ConformalDetector(
        IsolationForest(random_state=0), Split(n_calib=20), seed=0
    )
    detector.fit(x_train).select(x_test, alpha=0.05)
    assert "score_polarity" in nonconform_caplog.text


def test_cross_validation_seeds_pipeline_steps(planted_batch) -> None:
    from nonconform import CrossValidation

    x_train, x_test, _ = planted_batch

    def p_values():
        detector = ConformalDetector(
            make_pipeline(StandardScaler(), IsolationForest(n_estimators=20)),
            CrossValidation(k=3),
            seed=5,
        )
        return detector.fit(x_train[:300]).compute_p_values(x_test)

    np.testing.assert_array_equal(p_values(), p_values())


def test_pyod_pipeline_polarity_and_blocked_final_step(nonconform_caplog) -> None:
    pytest.importorskip("pyod")
    from pyod.models.ecod import ECOD
    from pyod.models.knn import KNN

    pipeline = make_pipeline(StandardScaler(), KNN())
    assert resolve_score_polarity(pipeline, "auto") is (
        ScorePolarity.HIGHER_IS_ANOMALOUS
    )
    ConformalDetector(pipeline, Split())
    assert POLARITY_WARNING not in nonconform_caplog.text

    with pytest.raises(ValueError, match="ECOD"):
        ConformalDetector(make_pipeline(StandardScaler(), ECOD()), Split())
