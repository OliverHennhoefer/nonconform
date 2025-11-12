"""Integration tests for data generators and dataset manager pipelines."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from nonconform.detection import ConformalDetector
from nonconform.strategy import Split
from nonconform.utils.data.generator import BatchGenerator, OnlineGenerator
from nonconform.utils.data.load import DatasetManager
from nonconform.utils.func.enums import Dataset
from pyod.models.iforest import IForest


def _make_fake_loader(
    *,
    n_normal: int = 120,
    n_anomaly: int = 30,
    n_features: int = 4,
    seed: int = 0,
):
    """Return callable producing synthetic dataset with Class labels."""
    rng = np.random.default_rng(seed)

    def _loader():
        normal = rng.standard_normal((n_normal, n_features))
        anomaly = rng.standard_normal((n_anomaly, n_features)) + 4.0
        df_normal = pd.DataFrame(
            normal, columns=[f"V{i + 1}" for i in range(n_features)]
        )
        df_normal["Class"] = 0
        df_anomaly = pd.DataFrame(
            anomaly, columns=[f"V{i + 1}" for i in range(n_features)]
        )
        df_anomaly["Class"] = 1
        return pd.concat([df_normal, df_anomaly], ignore_index=True)

    return _loader


def _build_detector():
    return ConformalDetector(
        detector=IForest(n_estimators=25, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        seed=29,
    )


def test_batch_generator_pipeline():
    """Batch generator output should plug directly into ConformalDetector."""
    loader = _make_fake_loader(seed=1)
    generator = BatchGenerator(
        load_data_func=loader,
        batch_size=40,
        anomaly_proportion=0.2,
        anomaly_mode="proportional",
        n_batches=3,
        seed=2,
    )
    detector = _build_detector()
    detector.fit(generator.get_training_data().to_numpy())

    batch_x, batch_y = next(generator.generate())
    p_values = detector.predict(batch_x.to_numpy())

    assert p_values.shape[0] == len(batch_x)
    assert batch_y.sum() == int(generator.batch_size * generator.anomaly_proportion)


def test_online_generator_stream(monkeypatch):
    """Online generator should stream instances with correct labels."""
    loader = _make_fake_loader(seed=3)
    generator = OnlineGenerator(
        load_data_func=loader,
        anomaly_proportion=0.1,
        n_instances=20,
        seed=4,
    )
    detector = _build_detector()
    detector.fit(generator.get_training_data().to_numpy())

    stream = list(generator.generate(n_instances=20))
    x_stream = np.vstack([instance.to_numpy() for instance, _ in stream])
    y_stream = np.array([label for _, label in stream])

    assert y_stream.sum() == int(0.1 * 20)
    preds = detector.predict(x_stream)
    assert preds.shape[0] == 20


def test_dataset_manager_setup_flow(monkeypatch):
    """DatasetManager.load(setup=True) should integrate without network access."""
    loader = _make_fake_loader(seed=5)
    df = loader()
    features = df.drop(columns=["Class"]).to_numpy()
    labels = df["Class"].to_numpy()
    buffer = io.BytesIO()
    np.savez(buffer, X=features, y=labels)
    payload = buffer.getvalue()

    manager = DatasetManager()
    monkeypatch.setattr(manager, "_download", lambda filename: payload)

    x_train, x_test, y_test = manager.load(Dataset.BREAST, setup=True, seed=6)
    detector = _build_detector()
    detector.fit(x_train.to_numpy())
    p_values = detector.predict(x_test.to_numpy())

    assert p_values.shape[0] == len(y_test)
