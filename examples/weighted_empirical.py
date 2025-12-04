from oddball import Dataset, load
from pyod.models.hbos import HBOS

from nonconform import (
    BootstrapBaggedWeightEstimator,
    ConformalDetector,
    Pruning,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
    weighted_bh,
    weighted_false_discovery_control,
)

if __name__ == "__main__":
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

    # Weighted Conformal Anomaly Detector
    wce = ConformalDetector(
        detector=HBOS(),
        strategy=Split(n_calib=1_000),
        weight_estimator=BootstrapBaggedWeightEstimator(
            base_estimator=logistic_weight_estimator(),
            n_bootstrap=100,
        ),
        seed=1,
    )

    wce.fit(x_train)
    weighted_p_values = wce.predict(x_test)

    # Apply weighted FDR control
    w_decisions = weighted_false_discovery_control(
        result=wce.last_result,
        alpha=0.2,
        pruning=Pruning.DETERMINISTIC,
        seed=1,
    )

    print("Standard Benjamini-Hochberg")
    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}")  # 0.00
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}")  # 0.00

    w_decisions = weighted_bh(wce.last_result, alpha=0.2)

    print("Weighted Benjamini-Hochberg")
    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}")  # 0.10
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}")  # 0.94
