from nonconform.detection import ConformalDetector
from nonconform.detection.weight import (
    BootstrapBaggedWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.strategy import Split
from nonconform.utils.data import Dataset, load
from nonconform.utils.func.enums import Pruning
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
    weighted_false_discovery_control,
)
from nonconform.utils.stat.weighted_fdr import weighted_bh
from pyod.models.hbos import HBOS

if __name__ == "__main__":
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

    # Weighted Conformal Anomaly Detector
    wce = ConformalDetector(
        detector=HBOS(),
        strategy=Split(n_calib=1_000),
        weight_estimator=BootstrapBaggedWeightEstimator(
            base_estimator=LogisticWeightEstimator(),
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

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}")  # 0.00
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}")  # 0.00

    w_decisions = weighted_bh(wce.last_result, alpha=0.2)

    print("Weighted Benjamini-Hochberg")
    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}")  # 0.11
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}")  # 0.94
