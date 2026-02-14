import numpy as np
from oddball import Dataset, load
from scipy.stats import false_discovery_control
from sklearn.neighbors import NearestNeighbors

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power


class KNNDistanceDetector:
    """Anomaly detector based on average k-nearest-neighbor distance."""

    def __init__(self, n_neighbors=5, metric="euclidean", random_state=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.random_state = random_state
        self._nn = None

    def fit(self, x, y=None):
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self._nn.fit(x)
        return self

    def decision_function(self, x):
        distances, _ = self._nn.kneighbors(x)
        return np.mean(distances, axis=1)

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

ce = ConformalDetector(
    detector=KNNDistanceDetector(n_neighbors=7),
    strategy=Split(n_calib=1_000),
    score_polarity="higher_is_anomalous",
    seed=1,
)

ce.fit(x_train)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
