from scipy.stats import false_discovery_control

from nonconform.estimation import ConformalDetector
from nonconform.estimation.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
    weighted_false_discovery_control,
)
from pyod.models.iforest import IForest

if __name__ == "__main__":
    # Example Setup
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True)

    # One-Class Classification
    model = IForest(behaviour="new")

    # Conformal Strategy
    strategy = Split(n_calib=10_000)

    # Weighted Conformal Anomaly Detector
    ce = ConformalDetector(
        detector=model,
        strategy=strategy,
        weight_estimator=LogisticWeightEstimator(seed=42),
    )
    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    # Get raw scores
    scores = ce.predict(x_test, raw=True)

    # Apply weighted FDR control
    ce.weight_estimator.fit(ce.calibration_samples, x_test)
    w_cal, w_test = ce.weight_estimator.get_weights()

    w_decisions = weighted_false_discovery_control(
        scores, ce.calibration_set, w_test, w_cal, q=0.2, rand="dtm", seed=1
    )

    print(
        f"Classical: \n"
        f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}\n"
        f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}"
    )
    print(
        f"Weighted: \n"
        f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}\n"
        f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}"
    )
