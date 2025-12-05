from oddball import Dataset, load
from pyod.models.lof import LOF
from scipy.stats import false_discovery_control

from nonconform import (
    ConformalDetector,
    CrossValidation,
    false_discovery_rate,
    statistical_power,
)

x_train, x_test, y_test = load(Dataset.MUSK, setup=True)

ce = ConformalDetector(detector=LOF(), strategy=CrossValidation.jackknife(plus=True))

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
