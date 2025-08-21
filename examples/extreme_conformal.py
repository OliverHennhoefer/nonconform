from online_fdr import BatchPRDS

from nonconform.estimation import ExtremeConformalDetector, StandardConformalDetector
from nonconform.strategy import Split
from nonconform.utils.data.generator import BatchGenerator
from nonconform.utils.data.load import load_shuttle
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest

if __name__ == "__main__":

    batch_gen = BatchGenerator(
        load_data_func=load_shuttle,
        batch_size=100,
        n_batches=10,
        anomaly_proportion=0.01,
        train_size=0.6,
        seed=42,
    )

    # Get training data from generator
    x_train = batch_gen.get_training_data()

    # EVT-Enhanced Conformal Anomaly Detector
    extreme_ce = ExtremeConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Split(n_calib=1_000),
        evt_threshold_method="percentile",
        evt_threshold_value=0.95,
        evt_min_tail_size=10,
    )

    standard_ce = StandardConformalDetector(
        detector=IForest(behaviour="new"), strategy=Split(n_calib=1_000)
    )

    extreme_ce.fit(x_train)
    standard_ce.fit(x_train)

    batch_fdr = BatchPRDS(alpha=0.2)

    label = []
    x_decision = []
    s_decision = []

    for batch_id, (x_batch, y_batch) in enumerate(batch_gen.generate()):
        xp = extreme_ce.predict(x_batch)
        sp = standard_ce.predict(x_batch)

        x_decisions = batch_fdr.test_batch(xp)
        s_decisions = batch_fdr.test_batch(sp)

        label.extend(y_batch.tolist())
        x_decision.extend(x_decisions)
        s_decision.extend(s_decisions)

    print(f"Empir. FDR: {false_discovery_rate(y=label, y_hat=x_decision)} (Extreme)")
    print(f"Empir. Power: {statistical_power(y=label, y_hat=x_decision)} (Extreme)")
    print("=" * 20)
    print(f"Empir. FDR: {false_discovery_rate(y=label, y_hat=s_decision)} (Standard)")
    print(f"Empir. Power: {statistical_power(y=label, y_hat=s_decision)} (Standard)")
