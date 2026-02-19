import numpy as np
from scipy.stats import false_discovery_control

from nonconform import (
    ConformalDetector,
    Empirical,
    Probabilistic,
    Split,
)
from nonconform.enums import Pruning
from nonconform.fdr import (
    weighted_bh_from_result,
    weighted_false_discovery_control,
)
from nonconform.metrics import false_discovery_rate, statistical_power
from nonconform.weighting import (
    BootstrapBaggedWeightEstimator,
    logistic_weight_estimator,
)


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
n_train = 3_500
n_test_normal = 1_200
n_test_anomaly = 220
alpha = 0.1
alpha_wcs = 0.15

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
x_train = rng.multivariate_normal(np.zeros(n_features), cov, size=n_train)
x_test_normal = rng.multivariate_normal(
    np.full(n_features, 0.20),
    cov,
    size=n_test_normal,
)
hard_anomalies = rng.multivariate_normal(
    np.array([1.25, -1.15, 1.10, -1.25, 1.00, -1.10]),
    1.35 * cov,
    size=int(0.7 * n_test_anomaly),
)
easy_anomalies = rng.multivariate_normal(
    np.array([2.1, -2.0, 1.9, -2.2, 1.8, -1.9]),
    1.1 * cov,
    size=n_test_anomaly - len(hard_anomalies),
)
x_test_anomaly = np.vstack([hard_anomalies, easy_anomalies])
x_test = np.vstack([x_test_normal, x_test_anomaly])
y_test = np.hstack(
    [np.zeros(n_test_normal, dtype=int), np.ones(n_test_anomaly, dtype=int)]
)
shuffle_idx = rng.permutation(len(x_test))
x_test = x_test[shuffle_idx]
y_test = y_test[shuffle_idx]

strategy = Split(n_calib=0.3)

# Standard Empirical (Classical)
ce = ConformalDetector(
    detector=MahalanobisDetector(),
    strategy=strategy,
    estimation=Empirical(tie_break="classical"),
    seed=1,
)
ce.fit(x_train)
p_values = ce.compute_p_values(x_test)
decisions = false_discovery_control(p_values, method="bh") <= alpha

print("Standard Empirical (Classical)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

# Standard Empirical (Randomized)
pce = ConformalDetector(
    detector=MahalanobisDetector(),
    strategy=strategy,
    estimation=Empirical(tie_break="randomized"),
    seed=1,
)
pce.fit(x_train)
p_values = pce.compute_p_values(x_test)
decisions = false_discovery_control(p_values, method="bh") <= alpha

print("\nStandard Empirical (Randomized)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

# Weighted Empirical (Classical)
wce = ConformalDetector(
    detector=MahalanobisDetector(),
    strategy=strategy,
    weight_estimator=BootstrapBaggedWeightEstimator(
        base_estimator=logistic_weight_estimator(),
        n_bootstraps=60,
    ),
    estimation=Empirical(tie_break="classical"),
    seed=1,
)
wce.fit(x_train)
p_values = wce.compute_p_values(x_test)

decisions = weighted_false_discovery_control(
    result=wce.last_result,
    alpha=alpha_wcs,
    pruning=Pruning.HETEROGENEOUS,
    seed=1,
)
print("\nWeighted Empirical (Classical, Conformal Selection)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

decisions = weighted_bh_from_result(wce.last_result, alpha=alpha)
print("\nWeighted Empirical (Classical, Weighted BH)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

# Weighted Empirical (Randomized)
wpce = ConformalDetector(
    detector=MahalanobisDetector(),
    strategy=strategy,
    weight_estimator=BootstrapBaggedWeightEstimator(
        base_estimator=logistic_weight_estimator(),
        n_bootstraps=60,
    ),
    estimation=Empirical(tie_break="randomized"),
    seed=1,
)
wpce.fit(x_train)
_ = wpce.compute_p_values(x_test)

decisions = weighted_false_discovery_control(
    result=wpce.last_result,
    alpha=alpha_wcs,
    pruning=Pruning.HETEROGENEOUS,
    seed=1,
)
print("\nWeighted Empirical (Randomized, Conformal Selection)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

decisions = weighted_bh_from_result(wpce.last_result, alpha=alpha)
print("\nWeighted Empirical (Randomized, Weighted BH)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

# Optional: Probabilistic (KDE)
try:
    spce = ConformalDetector(
        detector=MahalanobisDetector(),
        strategy=strategy,
        estimation=Probabilistic(n_trials=10),
        seed=1,
    )
    spce.fit(x_train)
    p_values = spce.compute_p_values(x_test)
    decisions = false_discovery_control(p_values, method="bh") <= alpha

    print("\nStandard Probabilistic (KDE, optional)")
    print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
    print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

    swpce = ConformalDetector(
        detector=MahalanobisDetector(),
        strategy=strategy,
        weight_estimator=BootstrapBaggedWeightEstimator(
            base_estimator=logistic_weight_estimator(),
            n_bootstraps=60,
        ),
        estimation=Probabilistic(n_trials=10),
        seed=1,
    )
    swpce.fit(x_train)
    _ = swpce.compute_p_values(x_test)

    decisions = weighted_false_discovery_control(
        result=swpce.last_result,
        alpha=alpha_wcs,
        pruning=Pruning.HETEROGENEOUS,
        seed=1,
    )
    print("\nWeighted Probabilistic (KDE, optional, Conformal Selection)")
    print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
    print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")

    decisions = weighted_bh_from_result(swpce.last_result, alpha=alpha)
    print("\nWeighted Probabilistic (KDE, optional, Weighted BH)")
    print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")
    print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")
except ImportError:
    print("\nProbabilistic (KDE, optional)")
    print(
        "  Skipped: install optional dependencies via "
        '"pip install nonconform[probabilistic]"'
    )
