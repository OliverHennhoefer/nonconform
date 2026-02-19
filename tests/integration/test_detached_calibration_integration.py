"""Integration tests for detached calibration workflows."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split


def test_detached_calibration_with_prefitted_detector(simple_dataset):
    """Pre-fitted detector can be calibrated and used for conformal p-values."""
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=18, n_features=4, seed=9)
    x_fit, x_calib = x_train[:50], x_train[50:]

    base = IsolationForest(
        n_estimators=30,
        max_samples=0.8,
        random_state=9,
    )
    base.fit(x_fit)

    detector = ConformalDetector(
        detector=base,
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=9,
    )
    detector.calibrate(x_calib)

    p_values = detector.compute_p_values(x_test)
    assert len(p_values) == len(x_test)
    assert np.isfinite(p_values).all()
    assert np.all((0 <= p_values) & (p_values <= 1))
