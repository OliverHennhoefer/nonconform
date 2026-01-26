"""Tests for strategy support of pandas DataFrame inputs."""

import numpy as np
import pandas as pd

from nonconform import CrossValidation, JackknifeBootstrap
from tests.conftest import MockDetector


def test_crossvalidation_accepts_dataframe():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((12, 3)))
    strategy = CrossValidation(k=3, plus=True, shuffle=True)
    detector_set, calib_scores = strategy.fit_calibrate(
        x=df, detector=MockDetector(), seed=42
    )
    assert len(detector_set) == 3
    assert len(calib_scores) == len(df)


def test_jackknife_bootstrap_accepts_dataframe():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.standard_normal((12, 2)))
    strategy = JackknifeBootstrap(n_bootstraps=3, plus=True)
    detector_set, calib_scores = strategy.fit_calibrate(
        x=df, detector=MockDetector(), seed=7
    )
    assert len(detector_set) == 3
    assert len(calib_scores) == len(df)
