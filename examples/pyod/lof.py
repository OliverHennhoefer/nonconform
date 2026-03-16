from oddball import Dataset, load
from pyod.models.lof import LOF

from nonconform import (
    ConformalDetector,
    CrossValidation,
)
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.MUSK, setup=True)

ce = ConformalDetector(detector=LOF(), strategy=CrossValidation.jackknife(mode="plus"))

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
