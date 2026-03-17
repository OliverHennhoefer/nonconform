from oddball import Dataset, load
from pyod.models.hbos import HBOS

from nonconform import (
    ConformalDetector,
    CrossValidation,
)
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True)

ce = ConformalDetector(
    detector=HBOS(),
    strategy=CrossValidation(k=15),
)

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
