from oddball import Dataset, load
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import ConformalDetector, JackknifeBootstrap
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.BREASTW, setup=True, seed=1)

ce = ConformalDetector(
    detector=IsolationForest(random_state=1),
    strategy=JackknifeBootstrap(n_bootstraps=100),
    score_polarity="higher_is_normal",
    seed=1,
)

ce.fit(x_train)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
