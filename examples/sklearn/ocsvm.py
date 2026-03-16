from oddball import Dataset, load
from sklearn.svm import OneClassSVM

from nonconform import (
    ConformalDetector,
    Split,
)
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

ce = ConformalDetector(
    detector=OneClassSVM(kernel="rbf", nu=0.05),
    strategy=Split(1_000),
    score_polarity="auto",
    seed=1,
)

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
