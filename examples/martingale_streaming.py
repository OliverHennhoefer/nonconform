"""Streaming martingale monitoring with conformal p-values."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, Split
from nonconform.martingales import (
    AlarmConfig,
    PowerMartingale,
    SimpleJumperMartingale,
    SimpleMixtureMartingale,
)


def main() -> None:
    rng = np.random.default_rng(42)

    # Train/calibration data
    x_train = rng.standard_normal((400, 5))

    # Stream with a shift in the second half
    x_stream_normal = rng.standard_normal((80, 5))
    x_stream_shifted = rng.standard_normal((80, 5)) + 2.0
    x_stream = np.vstack([x_stream_normal, x_stream_shifted])

    detector = ConformalDetector(
        detector=IsolationForest(random_state=42),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=42,
    )
    detector.fit(x_train)

    alarms = AlarmConfig(ville_threshold=100.0)
    martingales = {
        "power": PowerMartingale(epsilon=0.5, alarm_config=alarms),
        "mixture": SimpleMixtureMartingale(
            epsilons=[0.25, 0.5, 0.75, 1.0], alarm_config=alarms
        ),
        "jumper": SimpleJumperMartingale(jump=0.01, alarm_config=alarms),
    }

    first_alarm_step: dict[str, int | None] = {name: None for name in martingales}

    for step, x_t in enumerate(x_stream, start=1):
        p_t = float(detector.compute_p_values(x_t.reshape(1, -1))[0])
        for name, martingale in martingales.items():
            state = martingale.update(p_t)
            if "ville" in state.triggered_alarms and first_alarm_step[name] is None:
                first_alarm_step[name] = step

    print("Final states:")
    for name, martingale in martingales.items():
        state = martingale.state
        print(
            f"{name:>7}: step={state.step}, M={state.martingale:.3g}, "
            f"CUSUM={state.cusum:.3g}, SR={state.shiryaev_roberts:.3g}, "
            f"first_ville_alarm={first_alarm_step[name]}"
        )


if __name__ == "__main__":
    main()
