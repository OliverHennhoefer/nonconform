"""Integration tests for martingales with real sklearn detectors and datasets."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import PowerMartingale


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
