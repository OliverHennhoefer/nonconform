from oddball import Dataset, load
from pyod.models.hbos import HBOS
from scipy.stats import false_discovery_control

from nonconform import (
    BootstrapBaggedWeightEstimator,
    ConformalDetector,
    Empirical,
    Probabilistic,
    Pruning,
    Split,
    false_discovery_rate,
    logistic_weight_estimator,
    statistical_power,
    weighted_bh,
    weighted_false_discovery_control,
)

x_train, x_test, y_test = load(Dataset.SHUTTLE, setup=True, seed=1)

strategy = Split(n_calib=1_000)
alpha = 0.2

# Standard Empirical
ce = ConformalDetector(
    detector=HBOS(),
    strategy=strategy,
    estimation=Empirical(),
    seed=1,
)
ce.fit(x_train)
p_values = ce.predict(x_test)
decisions = false_discovery_control(p_values, method="bh") <= alpha

print("Standard Empirical")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.20
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.94

# Standard Probabilistic (KDE)
pce = ConformalDetector(
    detector=HBOS(),
    strategy=strategy,
    estimation=Probabilistic(n_trials=10),
    seed=1,
)
pce.fit(x_train)
p_values = pce.predict(x_test)
decisions = false_discovery_control(p_values, method="bh") <= alpha

print("\nStandard Probabilistic (KDE)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.16
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.94

# Weighted Empirical
wce = ConformalDetector(
    detector=HBOS(),
    strategy=strategy,
    weight_estimator=BootstrapBaggedWeightEstimator(
        base_estimator=logistic_weight_estimator(),
        n_bootstrap=100,
    ),
    seed=1,
)
wce.fit(x_train)
p_values = wce.predict(x_test)

decisions = weighted_false_discovery_control(
    result=wce.last_result,
    alpha=alpha,
    pruning=Pruning.DETERMINISTIC,
    seed=1,
)
print("\nWeighted Empirical (Conformal Selection)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.00
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.00

decisions = weighted_bh(wce.last_result, alpha=alpha)
print("\nWeighted Empirical (Weighted BH)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.10
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.94

# Weighted Probabilistic (KDE)
wpce = ConformalDetector(
    detector=HBOS(),
    strategy=strategy,
    weight_estimator=BootstrapBaggedWeightEstimator(
        base_estimator=logistic_weight_estimator(),
        n_bootstrap=100,
    ),
    estimation=Probabilistic(n_trials=10),
    seed=1,
)
wpce.fit(x_train)
_ = wpce.predict(x_test)

decisions = weighted_false_discovery_control(
    result=wpce.last_result,
    alpha=alpha,
    pruning=Pruning.DETERMINISTIC,
    seed=1,
)
print("\nWeighted Probabilistic (Conformal Selection)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.00
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.93

decisions = weighted_bh(wpce.last_result, alpha=alpha)
print("\nWeighted Probabilistic (Weighted BH)")
print(f"  FDR: {false_discovery_rate(y=y_test, y_hat=decisions):.2f}")  # 0.10
print(f"  Power: {statistical_power(y=y_test, y_hat=decisions):.2f}")  # 0.94
