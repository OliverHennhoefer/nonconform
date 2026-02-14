"""Integration tests for sklearn score polarity auto-resolution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from nonconform import ConformalDetector, Split
from nonconform.enums import ScorePolarity


@pytest.mark.parametrize(
    ("detector_cls", "detector_kwargs"),
    [
        (IsolationForest, {"random_state": 7}),
        (OneClassSVM, {"nu": 0.05, "kernel": "rbf"}),
    ],
    ids=["IsolationForest", "OneClassSVM"],
)
def test_auto_polarity_end_to_end(
    simple_dataset,
    detector_cls,
    detector_kwargs,
) -> None:
    """AUTO polarity should resolve and run for supported sklearn detectors."""
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=25, n_features=4)
    detector = ConformalDetector(
        detector=detector_cls(**detector_kwargs),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=7,
    )
    detector.fit(x_train)
    p_values = detector.compute_p_values(x_test)

    assert detector.score_polarity is ScorePolarity.HIGHER_IS_NORMAL
    assert p_values.shape == (len(x_test),)
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))


def test_dataframe_input_preserves_index_with_series_output(simple_dataset) -> None:
    """DataFrame inputs should produce indexed pandas Series outputs."""
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=25, n_features=4)
    detector = ConformalDetector(
        detector=IsolationForest(random_state=7),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=7,
    )
    detector.fit(x_train)

    index = pd.RangeIndex(start=1000, stop=1025, step=1)
    x_test_df = pd.DataFrame(x_test, index=index)

    p_values = detector.compute_p_values(x_test_df)
    scores = detector.score_samples(x_test_df)

    assert isinstance(p_values, pd.Series)
    assert isinstance(scores, pd.Series)
    assert p_values.index.equals(index)
    assert scores.index.equals(index)
    assert p_values.between(0.0, 1.0).all()


def test_series_input_preserves_index_for_single_feature(simple_dataset) -> None:
    """Series input should be treated as a single-feature batch."""
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=25, n_features=1)
    detector = ConformalDetector(
        detector=IsolationForest(random_state=7),
        strategy=Split(n_calib=0.2),
        score_polarity="auto",
        seed=7,
    )
    detector.fit(x_train)

    index = pd.RangeIndex(start=300, stop=325, step=1)
    x_test_series = pd.Series(x_test[:, 0], index=index)

    p_values = detector.compute_p_values(x_test_series)
    scores = detector.score_samples(x_test_series)

    assert isinstance(p_values, pd.Series)
    assert isinstance(scores, pd.Series)
    assert p_values.index.equals(index)
    assert scores.index.equals(index)
    assert p_values.between(0.0, 1.0).all()
