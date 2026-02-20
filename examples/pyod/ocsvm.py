from oddball import Dataset, load
from pyod.models.ocsvm import OCSVM
from scipy.stats import false_discovery_control

from nonconform import (
    ConformalDetector,
    Split,
)
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True)

ce = ConformalDetector(detector=OCSVM(), strategy=Split(0.2))

ce.fit(x_train)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
