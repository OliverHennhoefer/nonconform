import numpy as np
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power


class MahalanobisDetector:
    """Mahalanobis distance detector with covariance regularization."""

    def __init__(self, random_state=None, regularization=1e-4):
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


rng = np.random.default_rng(1)

n_features = 6
n_fit = 2_000
n_calib = 1_000
n_test_normal = 1_200
n_test_anomaly = 250
alpha = 0.1

cov = np.array(
    [
        [1.0, 0.25, 0.10, 0.00, 0.00, 0.00],
        [0.25, 1.2, 0.20, 0.10, 0.00, 0.00],
        [0.10, 0.20, 0.9, 0.25, 0.05, 0.00],
        [0.00, 0.10, 0.25, 1.1, 0.20, 0.05],
        [0.00, 0.00, 0.05, 0.20, 1.0, 0.15],
        [0.00, 0.00, 0.00, 0.05, 0.15, 0.8],
    ]
)
x_fit = rng.multivariate_normal(np.zeros(n_features), cov, size=n_fit)
x_calib = rng.multivariate_normal(np.zeros(n_features), cov, size=n_calib)

x_test_normal = rng.multivariate_normal(
    np.full(n_features, 0.10),
    cov,
    size=n_test_normal,
)
x_test_anomaly = rng.multivariate_normal(
    np.array([2.4, -2.2, 2.1, -2.4, 2.0, -2.1]),
    1.1 * cov,
    size=n_test_anomaly,
)

x_test = np.vstack([x_test_normal, x_test_anomaly])
y_test = np.hstack(
    [np.zeros(n_test_normal, dtype=int), np.ones(n_test_anomaly, dtype=int)]
)
shuffle_idx = rng.permutation(len(x_test))
x_test = x_test[shuffle_idx]
y_test = y_test[shuffle_idx]

base_detector = MahalanobisDetector()
base_detector.fit(x_fit)

ce = ConformalDetector(
    detector=base_detector,
    strategy=Split(n_calib=0.2),
    score_polarity="higher_is_anomalous",
    seed=1,
)

ce.calibrate(x_calib)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= alpha

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
