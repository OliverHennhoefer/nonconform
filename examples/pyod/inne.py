from scipy.stats import false_discovery_control

from nonconform.estimation import StandardConformalDetector
from nonconform.strategy import Bootstrap
from nonconform.utils.data import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.inne import INNE

x_train, x_test, y_test = load_shuttle(setup=True)

ce = StandardConformalDetector(
    detector=INNE(),
    strategy=Bootstrap(resampling_ratio=0.99, n_calib=2_000),
)

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
