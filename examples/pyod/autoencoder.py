from oddball import Dataset, load
from pyod.models.auto_encoder import AutoEncoder

from nonconform import ConformalDetector, Split
from nonconform.metrics import false_discovery_rate, statistical_power

x_train, x_test, y_test = load(Dataset.FRAUD, setup=True)

ce = ConformalDetector(
    detector=AutoEncoder(epoch_num=10, batch_size=256),
    strategy=Split(n_calib=2_000),
)

ce.fit(x_train)
decisions = ce.select(x_test, alpha=0.2)

print(f"Empirical FDR: {false_discovery_rate(y=y_test, y_hat=decisions)}")
print(f"Empirical Power: {statistical_power(y=y_test, y_hat=decisions)}")
