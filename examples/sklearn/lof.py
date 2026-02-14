from oddball import Dataset, load
from scipy.stats import false_discovery_control
from sklearn.neighbors import LocalOutlierFactor

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

ce = ConformalDetector(
    detector=LocalOutlierFactor(n_neighbors=35, novelty=True),
    strategy=Split(n_calib=0.2),
    score_polarity="higher_is_normal",
    seed=1,
)

ce.fit(x_train)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
