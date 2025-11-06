from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.strategy import Probabilistic, Split
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.hbos import HBOS

if __name__ == "__main__":
    x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

    # Probabilistic KDE-based Conformal Detector
    pce = ConformalDetector(
        detector=HBOS(),
        strategy=Split(n_calib=1_000),
        estimation=Probabilistic(n_trials=10),
        seed=1,
    )

    pce.fit(x_train)
    estimates = pce.predict(x_test)
    print(estimates.min())

    # Apply FDR control
    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")  # 0.16
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")  # 0.94
