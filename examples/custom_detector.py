import numpy as np
from oddball import Dataset, load
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split, false_discovery_rate, statistical_power


class CentroidDetector:
    """Minimal anomaly detector: distance from training centroid."""

    def __init__(self, random_state=None):
        self.random_state = random_state
        self._center = None

    def fit(self, x):
        self._center = x.mean(axis=0)
        return self

    def decision_function(self, x):
        return np.linalg.norm(x - self._center, axis=1)

    def get_params(self):
        return {"random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

ce = ConformalDetector(
    detector=CentroidDetector(),
    strategy=Split(n_calib=1_000),
    score_polarity="higher_is_anomalous",
    seed=1,
)

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
