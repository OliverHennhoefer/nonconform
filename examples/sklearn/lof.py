from oddball import Dataset, load
from sklearn.neighbors import LocalOutlierFactor

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

ce = ConformalDetector(
    detector=LocalOutlierFactor(n_neighbors=35, novelty=True),
    strategy=Split(n_calib=0.2),
    score_polarity="auto",
    seed=1,
)

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
