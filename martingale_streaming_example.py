"""Minimal flat streaming example for conformal martingales.

Train an IsolationForest on a subset, then process the remaining data as a
stream and update a martingale online from conformal p-values.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, PowerMartingale

x_all, y_all = load_iris(return_X_y=True)

# Class 0 is treated as normal/reference data.
x_normal = x_all[y_all == 0]
x_anomaly = x_all[y_all != 0]

rng = np.random.default_rng(42)
x_normal = x_normal[rng.permutation(len(x_normal))]
x_anomaly = x_anomaly[rng.permutation(len(x_anomaly))]

x_train = x_normal[:30]
x_stream = np.vstack([x_normal[30:], x_anomaly])
y_stream = np.hstack(
    [
        np.zeros(len(x_normal) - 30, dtype=int),
        np.ones(len(x_anomaly), dtype=int),
    ]
)

detector = ConformalDetector(
    detector=IsolationForest(random_state=42),
    strategy=Split(n_calib=0.3),
    score_polarity="auto",
    seed=42,
)
detector.fit(x_train)

martingale = PowerMartingale(
    epsilon=0.5,
    alarm_config=AlarmConfig(ville_threshold=20.0),
)

for step, (x_t, is_anomaly) in enumerate(zip(x_stream, y_stream, strict=True), start=1):
    p_t = float(detector.compute_p_values(x_t.reshape(1, -1))[0])
    state = martingale.update(p_t)
    if step % 20 == 0:
        print(
            f"step={step:3d} p={p_t:.4f} anomaly={bool(is_anomaly)} "
            f"M={state.martingale:.3g}"
        )
    if "ville" in state.triggered_alarms:
        print(f"Ville alarm at step={step}, M={state.martingale:.3g}")
        break

print(
    f"final_step={martingale.state.step}, "
    f"final_M={martingale.state.martingale:.3g}, "
    f"log_M={martingale.state.log_martingale:.3f}"
)
