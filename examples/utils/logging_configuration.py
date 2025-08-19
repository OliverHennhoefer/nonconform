import logging

from scipy.stats import false_discovery_control

from nonconform.estimation import StandardConformalDetector
from nonconform.strategy import Randomized
from nonconform.utils.data import load_wbc
from nonconform.utils.func.enums import Distribution
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest

if __name__ == "__main__":

    # Configure logging to be more verbose (default: WARNING)
    logging.basicConfig(level=logging.INFO)

    x_train, x_test, y_test = load_wbc(setup=True)
    ce = StandardConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Randomized(n_calib=1_000, sampling_distr=Distribution.BETA_BINOMIAL),
    )

    ce.fit(x_train)
    estimates = ce.predict(x_test)

    decisions = false_discovery_control(estimates, method="bh") <= 0.2

    print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
    print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
