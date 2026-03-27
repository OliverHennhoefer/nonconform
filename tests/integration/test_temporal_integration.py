"""Integration tests for temporal session orchestration workflows."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale
from nonconform.temporal import TemporalSession


class ThresholdController:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold

    def test_one(self, p_value: float) -> bool:
        return p_value <= self.threshold


def test_temporal_session_end_to_end(simple_dataset):
    """TemporalSession should orchestrate detector + controller + martingale."""
    x_train, x_test, _ = simple_dataset(n_train=150, n_test=40, n_features=5)
    detector = ConformalDetector(
        detector=IsolationForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=4,
    )
    detector.fit(x_train)

    session = TemporalSession(
        detector=detector,
        online_controller=ThresholdController(threshold=0.2),
        martingale=PowerMartingale(epsilon=0.5),
    )
    result = session.step(x_test[:20], apply_batch_select=True, alpha=0.1)

    assert np.asarray(result.p_values).shape == (20,)
    assert np.asarray(result.online_decisions).shape == (20,)
    assert result.martingale_states is not None
    assert len(result.martingale_states) == 20
    assert result.martingale_states[-1].step == 20
    assert session.last_batch_decisions is not None
    assert np.asarray(session.last_batch_decisions).shape == (20,)


def test_temporal_session_online_fdr_controller_smoke(simple_dataset):
    """TemporalSession should work with real online_fdr controllers when installed."""
    pytest.importorskip("online_fdr", reason="online_fdr not installed")
    from online_fdr.investing.alpha.alpha import Gai

    x_train, x_test, _ = simple_dataset(n_train=120, n_test=30, n_features=4)
    detector = ConformalDetector(
        detector=IsolationForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=8,
    )
    detector.fit(x_train)

    controller = Gai(alpha=0.1, wealth=0.05)
    session = TemporalSession(detector=detector, online_controller=controller)
    result = session.step(x_test[:15])

    assert result.online_decisions is not None
    assert np.asarray(result.online_decisions).dtype == bool
    assert np.asarray(result.online_decisions).shape == (15,)
