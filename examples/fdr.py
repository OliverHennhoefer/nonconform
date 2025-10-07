from scipy.stats import false_discovery_control

from nonconform.estimation import ConformalDetector
from nonconform.estimation.weight import LogisticWeightEstimator
from nonconform.strategy import Split
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
    weighted_conformalized_selection,
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

    # Get raw scores
    scores = ce.predict(x_test, raw=True)

    # Fit weight estimator and get weights
    ce.weight_estimator.fit(ce.calibration_samples, x_test)
    w_cal, w_test = ce.weight_estimator.get_weights()

    # Apply WCS: Weighted Conformalized Selection with FDR control
    _, discoveries, _, _ = weighted_conformalized_selection(
        scores, ce.calibration_set, w_test, w_cal, q=0.2, rand="homo"
    )

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=discoveries)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=discoveries)}")

    raw_scores = ce.predict(x_test, raw=False)
    decisions = false_discovery_control(raw_scores, method="bh") <= 0.2
    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
