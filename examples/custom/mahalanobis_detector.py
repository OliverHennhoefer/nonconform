import numpy as np
from oddball import Dataset, load

from nonconform import ConformalDetector, CrossValidation
from nonconform.metrics import false_discovery_rate, statistical_power


class MahalanobisDetector:
    """Mahalanobis distance anomaly detector with covariance regularization."""

    def __init__(self, random_state=None, regularization=1e-6):
        self.random_state = random_state
        self.regularization = regularization
        self._mean = None
        self._cov_inv = None

    def fit(self, x, y=None):
        self._mean = np.mean(x, axis=0)
        cov = np.cov(x.T) + self.regularization * np.eye(x.shape[1])
        self._cov_inv = np.linalg.inv(cov)
        return self

    def decision_function(self, x):
        diff = x - self._mean
        return np.sqrt(np.sum(diff @ self._cov_inv * diff, axis=1))

    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "regularization": self.regularization,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

ce = ConformalDetector(
    detector=MahalanobisDetector(),
    strategy=CrossValidation(k=25),
    score_polarity="higher_is_anomalous",
    seed=1,
)

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
