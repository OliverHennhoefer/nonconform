"""Integration tests for streaming p-value to martingale workflows."""

from __future__ import annotations

import numpy as np

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale, SimpleJumperMartingale
from tests.conftest import MockDetector


def test_detector_streaming_p_values_feed_martingale():
    """ConformalDetector p-values should update martingale state online."""
    rng = np.random.default_rng(123)
    x_train = rng.standard_normal((120, 5))
    x_stream = rng.standard_normal((30, 5))
    fixed_scores = np.linspace(-2.0, 2.0, 300)

    detector = ConformalDetector(
        detector=MockDetector(scores=fixed_scores),
        strategy=Split(n_calib=0.2),
        seed=7,
    )
    detector.fit(x_train)

    martingale = SimpleJumperMartingale()
    for i, x in enumerate(x_stream, start=1):
        p_value = float(detector.compute_p_values(x.reshape(1, -1))[0])
        state = martingale.update(p_value)
        assert 0.0 <= p_value <= 1.0
        assert state.step == i

    assert martingale.state.step == len(x_stream)
    assert np.isfinite(martingale.state.log_martingale)


def test_shifted_stream_tends_to_increase_power_martingale_evidence():
    """A distribution shift should increase evidence relative to baseline stream."""
    rng = np.random.default_rng(77)
    x_train = rng.standard_normal((180, 4))
    x_normal = rng.standard_normal((60, 4))
    x_shifted = rng.standard_normal((60, 4)) + 2.0
    fixed_scores = np.linspace(-1.0, 1.0, 600)

    detector = ConformalDetector(
        detector=MockDetector(scores=fixed_scores),
        strategy=Split(n_calib=0.25),
        seed=11,
    )
    detector.fit(x_train)

    normal_p = detector.compute_p_values(x_normal)
    shifted_p = detector.compute_p_values(x_shifted)

    normal_m = PowerMartingale(epsilon=0.5).update_many(normal_p)[-1].log_martingale
    shifted_m = PowerMartingale(epsilon=0.5).update_many(shifted_p)[-1].log_martingale

    assert shifted_m >= normal_m
