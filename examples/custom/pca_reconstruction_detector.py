import numpy as np
from oddball import Dataset, load
from scipy.stats import false_discovery_control
from sklearn.decomposition import PCA

from nonconform import ConformalDetector, JackknifeBootstrap
from nonconform.metrics import false_discovery_rate, statistical_power


class PCAReconstructionDetector:
    """Anomaly detector using PCA reconstruction error."""

    def __init__(self, n_components=0.9, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self._pca = None

    def fit(self, x, y=None):
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(x)
        return self

    def decision_function(self, x):
        latent = self._pca.transform(x)
        reconstructed = self._pca.inverse_transform(latent)
        return np.mean((x - reconstructed) ** 2, axis=1)

    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


x_train, x_test, y_test = load(Dataset.MUSK, setup=True, seed=1)

ce = ConformalDetector(
    detector=PCAReconstructionDetector(n_components=0.9, random_state=1),
    strategy=JackknifeBootstrap(n_bootstraps=100),
    score_polarity="higher_is_anomalous",
    seed=1,
)

ce.fit(x_train)
estimates = ce.compute_p_values(x_test)

decisions = false_discovery_control(estimates, method="bh") <= 0.2

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
