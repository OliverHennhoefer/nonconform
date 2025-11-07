from scipy.stats import false_discovery_control

from nonconform.detection import ConformalDetector
from nonconform.strategy import CrossValidation
from nonconform.utils.data import Dataset, load
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.qmcd import QMCD

x_train, x_test, y_test = load(Dataset.IONOSPHERE, setup=True)

ce = ConformalDetector(detector=QMCD(), strategy=CrossValidation(k=15))

ce.fit(x_train)
estimates = ce.predict(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
