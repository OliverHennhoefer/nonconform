from nonconform.detection import ConformalDetector
from nonconform.detection.weight import (
    BootstrapBaggedWeightEstimator,
    LogisticWeightEstimator,
)
from nonconform.strategy import Probabilistic, Split
from nonconform.utils.data import Dataset, load
from nonconform.utils.func.enums import Pruning
from nonconform.utils.stat import (
    false_discovery_rate,
    statistical_power,
    weighted_false_discovery_control,
)
from pyod.models.hbos import HBOS

if __name__ == "__main__":
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

    # Weighted Conformal Anomaly Detector
    ce = ConformalDetector(
        detector=HBOS(),
        strategy=Split(n_calib=1_000),
        weight_estimator=BootstrapBaggedWeightEstimator(
            base_estimator=LogisticWeightEstimator(),
            n_bootstrap=100,
        ),
        estimation=Probabilistic(),
        seed=1,
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    # Apply weighted FDR control
    scores = ce.predict(x_test, raw=True)
    w_cal, w_test = ce.weight_estimator.get_weights()

    w_decisions = weighted_false_discovery_control(
        scores,
        ce.calibration_set,
        w_test,
        w_cal,
        q=0.2,
        pruning=Pruning.DETERMINISTIC,
        seed=1,
    )

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=w_decisions)}")  # 0.10
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=w_decisions)}")  # 0.94
