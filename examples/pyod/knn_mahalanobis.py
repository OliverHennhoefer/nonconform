import numpy as np
from oddball import Dataset, load
from pyod.models.knn import KNN
from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.stat import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True)

ce = ConformalDetector(
    detector=KNN(
        algorithm="auto",
        metric="mahalanobis",
        metric_params={"V": np.cov(x_train, rowvar=False)},
    ),
    strategy=Split(n_calib=1_000),
)

ce.fit(x_train)
estimates = ce.predict(x_test)
# Apply FDR control
decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
