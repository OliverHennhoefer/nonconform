"""Integration tests for martingales with real sklearn detectors and datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import (
    AlarmConfig,
    MixtureMartingale,
    PowerMartingale,
    SimpleJumperMartingale,
)
from nonconform.monitoring import ExchangeabilityMonitor


def test_isolation_forest_iris_streaming_signal_strengthens_on_shift():
    """Shifted anomaly stream should reduce p-values and raise evidence."""
    x_all, y_all = load_iris(return_X_y=True)

    # Use class 0 as normal reference distribution.
    x_normal = x_all[y_all == 0]
    x_anomaly = x_all[y_all != 0]

    x_train = x_normal[:30]
    x_stream_normal = x_normal[30:]
    x_stream_anomaly = x_anomaly

    detector = ConformalDetector(
        detector=IsolationForest(random_state=42),
        strategy=Split(n_calib=0.3),
        score_polarity="auto",
        seed=42,
    )
    detector.fit(x_train)

    p_normal = detector.compute_p_values(x_stream_normal)
    p_anomaly = detector.compute_p_values(x_stream_anomaly)

    # Smart signal check: shifted examples should be less conformal on average.
    assert float(np.mean(p_anomaly)) < float(np.mean(p_normal))

    martingale = PowerMartingale(epsilon=0.5)
    martingale.update_many(p_normal)
    log_after_normal = martingale.state.log_martingale

    martingale.update_many(p_anomaly)
    assert martingale.state.log_martingale > log_after_normal


def test_split_detector_bridge_runs_rigorous_sequential_rank_workflow():
    """A fitted real Split detector should bridge without changing its state."""
    x_all, y_all = load_iris(return_X_y=True)
    x_normal = x_all[y_all == 0]
    x_shifted = x_all[y_all != 0]
    detector = ConformalDetector(
        detector=IsolationForest(random_state=42),
        strategy=Split(n_calib=10),
        score_polarity="auto",
        seed=42,
    ).fit(x_normal[:40])
    calibration_before = detector.calibration_set

    monitor = ExchangeabilityMonitor.from_split_detector(
        detector,
        martingale=PowerMartingale(
            epsilon=0.5,
            alarm_config=AlarmConfig(restarted_ville_threshold=100.0),
        ),
        seed=42,
    )
    normal_states = monitor.update_many(x_normal[40:])
    log_after_normal = normal_states[-1].martingale_state.log_martingale
    shifted_states = monitor.update_many(x_shifted)

    assert shifted_states[-1].rank_step == len(calibration_before) + 10 + len(x_shifted)
    assert shifted_states[-1].martingale_state.log_martingale > log_after_normal
    np.testing.assert_array_equal(detector.calibration_set, calibration_before)


@pytest.mark.parametrize("use_bridge", [False, True])
def test_mixture_monitor_matches_independent_experts(use_bridge):
    rng = np.random.default_rng(42)
    x_train, x_reference, x_stream = (
        rng.normal(size=(40, 3)),
        rng.normal(size=(10, 3)),
        rng.normal(size=(8, 3)),
    )
    experts = [PowerMartingale(0.5), SimpleJumperMartingale()]
    prototype = MixtureMartingale(experts, weights=[1, 3])
    scorer = IsolationForest(n_estimators=8, random_state=42)
    if use_bridge:
        detector = ConformalDetector(
            scorer, strategy=Split(n_calib=10), score_polarity="auto", seed=42
        ).fit(x_train)
        calibration_before = detector.calibration_set
        monitor = ExchangeabilityMonitor.from_split_detector(
            detector, martingale=prototype, seed=42
        )
    else:
        monitor = (
            ExchangeabilityMonitor(
                scorer, martingale=prototype, score_polarity="auto", seed=42
            )
            .fit(x_train)
            .prime(x_reference)
        )

    # The monitor owns its mixture, which in turn owns its expert states.
    prototype.update(0)
    for step, state in enumerate(monitor.update_many(x_stream), 1):
        expected = [expert.update(state.p_value) for expert in experts]
        assert state.rank_step == 10 + step
        assert state.evidence_step == step
        for name in ("martingale", "restarted_martingale", "cusum", "shiryaev_roberts"):
            assert getattr(state.martingale_state, name) == pytest.approx(
                0.25 * getattr(expected[0], name) + 0.75 * getattr(expected[1], name)
            )
    if use_bridge:
        np.testing.assert_array_equal(detector.calibration_set, calibration_before)
    monitor.reset()
    assert monitor.state is None
    monitor.prime(x_reference)
    assert monitor.update(x_stream[0]).evidence_step == 1
    assert prototype.state.step == 1


def test_notebook_mixture_workflow_smoke(capsys):
    rng = np.random.default_rng(12)
    detector = ConformalDetector(
        IsolationForest(n_estimators=8, random_state=42),
        strategy=Split(n_calib=10),
        score_polarity="auto",
        seed=42,
    ).fit(rng.normal(size=(40, 3)))
    namespace = {
        "detector": detector,
        "x_stream": pd.DataFrame(rng.normal(size=(12, 3))),
    }
    path = Path(__file__).parents[2] / "examples" / "exchangeability_martingale.ipynb"
    notebook = json.loads(path.read_text())
    cell = next(
        cell for cell in notebook["cells"] if cell.get("id") == "mixture-workflow"
    )
    exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    assert namespace["mixture_monitor"].state.evidence_step == 12
    assert "Mixture SR first alarm:" in capsys.readouterr().out
