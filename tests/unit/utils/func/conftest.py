import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_detector():
    """Create a mock PyOD detector with configurable parameters."""

    def _create(has_contamination=True, has_n_jobs=True, has_random_state=True):
        detector = MagicMock()
        params = {}

        if has_contamination:
            params["contamination"] = 0.1
        if has_n_jobs:
            params["n_jobs"] = 1
        if has_random_state:
            params["random_state"] = None

        detector.get_params.return_value = params
        detector.set_params = MagicMock(
            side_effect=lambda **kwargs: params.update(kwargs)
        )

        return detector

    return _create


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrames for decorator testing."""

    def _create(n_rows=10, n_cols=3, dtype=np.float32, seed=42):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((n_rows, n_cols)).astype(dtype)
        return pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])

    return _create


@pytest.fixture
def sample_series():
    """Create sample Series for decorator testing."""

    def _create(n_rows=10, dtype=np.float32, seed=42):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal(n_rows).astype(dtype)
        return pd.Series(data, name="values")

    return _create


@pytest.fixture
def sample_array():
    """Create sample numpy arrays for decorator testing."""

    def _create(shape=(10, 3), dtype=np.float32, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(dtype)

    return _create


@pytest.fixture
def capture_logs():
    """Capture log records for testing logger behavior."""

    class LogCapture:
        def __init__(self):
            self.records = []
            self.handler = logging.Handler()
            self.handler.emit = lambda record: self.records.append(record)

        def attach(self, logger):
            logger.addHandler(self.handler)
            return self

        def detach(self, logger):
            logger.removeHandler(self.handler)

        def get_messages(self, level=None):
            if level is None:
                return [r.getMessage() for r in self.records]
            return [r.getMessage() for r in self.records if r.levelno == level]

        def clear(self):
            self.records.clear()

    return LogCapture()
