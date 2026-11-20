"""Detector and result tests for BH/Storey selection and the resolution guard."""

from __future__ import annotations

import logging
from typing import Self

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from nonconform import (
    ConformalDetector,
    CrossValidation,
    DerandomizedSplits,
    Empirical,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform.scoring import ConditionalEmpirical
from nonconform.structures import ConformalResult


class DistanceDetector:
    """Deterministic detector with continuous anomalous-higher scores."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state
        self.center_: np.ndarray | None = None

    def fit(self, X, y=None) -> Self:
        _ = y
        self.center_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        return np.linalg.norm(np.asarray(X) - self.center_, axis=1)

    def get_params(self, deep=True):
        _ = deep
        return {"random_state": self.random_state}

    def set_params(self, **params) -> Self:
        self.random_state = params.get("random_state", self.random_state)
        return self


@pytest.fixture
def fdr_records():
    """Collect warnings emitted on the nonconform.fdr logger."""

    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.WARNING)
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    logger = logging.getLogger("nonconform.fdr")
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _data(
    seed: int = 0,
    n_reference: int = 1_200,
    n_null: int = 60,
    n_alt: int = 40,
    shift: float = 2.5,
):
    rng = np.random.default_rng(seed)
    x_reference = rng.normal(size=(n_reference, 3))
    x_test = np.vstack(
        [rng.normal(size=(n_null, 3)), rng.normal(loc=shift, size=(n_alt, 3))]
    )
    return x_reference, x_test


def _detector(**kwargs) -> ConformalDetector:
    kwargs.setdefault("strategy", Split(n_calib=600))
    return ConformalDetector(
        detector=DistanceDetector(),
        score_polarity="higher_is_anomalous",
        seed=3,
        **kwargs,
    )


def _reference_storey_mask(p_values, n_calibration, alpha):
    """Independent Storey-BH implementation on the classical p-value grid."""
    k = int(np.ceil((n_calibration + 1) / 2))
    lam = k / (n_calibration + 1)
    pi0 = (1 + np.sum(p_values >= lam)) / (p_values.size * (1 - lam))
    m = p_values.size
    order = np.argsort(p_values, kind="mergesort")
    passes = np.flatnonzero(p_values[order] <= alpha / pi0 * np.arange(1, m + 1) / m)
    mask = np.zeros(m, dtype=bool)
    if passes.size:
        mask[order[: passes[-1] + 1]] = True
    return mask


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_storey_select_matches_reference_and_adjusted_snapshot(alpha):
    x_reference, x_test = _data()
    detector = _detector().fit(x_reference)

    mask = detector.select(x_test, alpha=alpha, method="storey")
    result = detector.last_result

    assert result is not None
    np.testing.assert_array_equal(
        mask, _reference_storey_mask(result.p_values, 600, alpha)
    )
    np.testing.assert_array_equal(
        mask, result.adjusted_p_values(method="storey") <= alpha
    )


def test_storey_finds_more_than_bh_when_anomalies_are_common():
    x_reference, x_test = _data(seed=2, n_null=60, n_alt=40, shift=2.0)
    detector = _detector().fit(x_reference)

    bh = detector.select(x_test, alpha=0.1)
    storey = detector.select(x_test, alpha=0.1, method="storey")

    assert np.all(storey[bh])
    assert storey.sum() > bh.sum()


def test_default_method_is_unchanged_bh():
    x_reference, x_test = _data(seed=1)
    detector = _detector().fit(x_reference)

    default = detector.select(x_test, alpha=0.1)
    p_values = detector.last_result.p_values
    explicit = detector.select(x_test, alpha=0.1, method="bh")

    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(
        default, false_discovery_control(p_values, method="bh") <= 0.1
    )
    np.testing.assert_array_equal(
        detector.last_result.adjusted_p_values(),
        false_discovery_control(p_values, method="bh"),
    )


def test_storey_select_preserves_pandas_index():
    x_reference, x_test = _data(seed=2)
    detector = _detector().fit(x_reference)
    index = pd.RangeIndex(start=500, stop=500 + len(x_test))

    selected = detector.select(pd.DataFrame(x_test, index=index), method="storey")

    assert isinstance(selected, pd.Series)
    assert selected.name == "selected"
    assert selected.index.equals(index)


def test_storey_supports_detached_calibration():
    x_reference, x_test = _data(seed=3)
    base = DistanceDetector().fit(x_reference[:600])
    detector = ConformalDetector(
        detector=base,
        strategy=Split(n_calib=0.5),
        score_polarity="higher_is_anomalous",
    ).calibrate(x_reference[600:])

    mask = detector.select(x_test, alpha=0.1, method="storey")

    np.testing.assert_array_equal(
        mask, _reference_storey_mask(detector.last_result.p_values, 600, 0.1)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimation": ConditionalEmpirical(method="simes")},
        {"strategy": CrossValidation(k=3)},
    ],
    ids=["conditional-empirical", "cross-validation"],
)
def test_storey_accepts_other_classical_p_value_constructions(kwargs):
    x_reference, x_test = _data(seed=4)
    detector = _detector(**kwargs).fit(x_reference)

    mask = detector.select(x_test, alpha=0.1, method="storey")
    result = detector.last_result

    np.testing.assert_array_equal(
        mask, result.adjusted_p_values(method="storey") <= 0.1
    )


@pytest.mark.parametrize(
    "estimation",
    [
        Empirical(tie_break="randomized"),
        ConditionalEmpirical(method="simes", tie_break="randomized"),
        Probabilistic(n_trials=0),
    ],
    ids=["randomized", "conditional-randomized", "probabilistic"],
)
def test_storey_rejects_non_classical_p_values_before_scoring(estimation):
    x_reference, x_test = _data(seed=5)
    detector = _detector(estimation=estimation).fit(x_reference)

    with pytest.raises(ValueError, match="tie_break='classical'"):
        detector.select(x_test, method="storey")
    assert detector.last_result is None

    detector.select(x_test)
    with pytest.raises(ValueError, match="tie_break='classical'"):
        detector.last_result.adjusted_p_values(method="storey")
    assert detector.last_result.adjusted_p_values().shape == (len(x_test),)


def test_storey_rejects_weighted_mode():
    x_reference, x_test = _data(seed=6, n_reference=300)
    detector = _detector(
        strategy=Split(n_calib=0.5),
        weight_estimator=logistic_weight_estimator(),
    ).fit(x_reference)

    with pytest.raises(ValueError, match="unweighted"):
        detector.select(x_test, method="storey")
    assert detector.last_result is None

    detector.compute_p_values(x_test)
    for method in ("bh", "storey"):
        with pytest.raises(ValueError, match="unweighted"):
            detector.last_result.adjusted_p_values(method=method)


def test_storey_rejects_derandomized_splits():
    x_reference, x_test = _data(seed=7, n_reference=300)
    detector = _detector(strategy=DerandomizedSplits(n_repetitions=3, n_calib=0.5)).fit(
        x_reference
    )

    with pytest.raises(ValueError, match="DerandomizedSplits"):
        detector.select(x_test, method="storey")
    assert detector.select(x_test).shape == (len(x_test),)


@pytest.mark.parametrize(
    ("method", "error"),
    [("by", ValueError), ("", ValueError), (None, TypeError), (1, TypeError)],
)
def test_select_rejects_unknown_methods(method, error):
    x_reference, x_test = _data(seed=8, n_reference=300)
    detector = _detector(strategy=Split(n_calib=0.5)).fit(x_reference)

    with pytest.raises(error, match="'bh' or 'storey'"):
        detector.select(x_test, method=method)
    detector.compute_p_values(x_test)
    with pytest.raises(error, match="'bh' or 'storey'"):
        detector.last_result.adjusted_p_values(method=method)


def test_adjusted_p_values_require_native_p_value_snapshots():
    x_reference, x_test = _data(seed=9, n_reference=300)
    detector = _detector(strategy=Split(n_calib=0.5)).fit(x_reference)

    detector.score_samples(x_test)
    with pytest.raises(ValueError, match="missing p_values"):
        detector.last_result.adjusted_p_values()

    manual = ConformalResult(
        p_values=np.array([0.01, 0.5]),
        test_scores=np.array([2.0, 0.1]),
        calib_scores=np.array([0.0, 0.3, 0.5]),
    )
    with pytest.raises(ValueError, match="native detector snapshot"):
        manual.adjusted_p_values()

    detector.compute_p_values(x_test)
    tampered = detector.last_result
    tampered.p_values = tampered.p_values[:-1]
    with pytest.raises(ValueError, match="batch dimensions"):
        tampered.adjusted_p_values()


def test_adjusted_p_values_return_independent_arrays():
    x_reference, x_test = _data(seed=10)
    detector = _detector().fit(x_reference)
    detector.compute_p_values(x_test)
    result = detector.last_result

    adjusted = result.adjusted_p_values(method="storey")
    adjusted[:] = 0.0

    assert np.all(result.adjusted_p_values(method="storey") > 0.0)
    assert np.all(result.p_values > 0.0)


def test_select_warns_when_calibration_floor_blocks_any_discovery(fdr_records):
    x_reference, _ = _data(seed=11, n_reference=400)
    rng = np.random.default_rng(12)
    x_clean = rng.normal(size=(500, 3))
    detector = _detector(strategy=Split(n_calib=49)).fit(x_reference)

    for method in ("bh", "storey"):
        selected = detector.select(x_clean, alpha=0.05, method=method)
        assert not selected.any()

    assert len(fdr_records) == 2
    message = fdr_records[0].getMessage()
    assert "With 49 calibration scores" in message
    assert "1/50 = 0.02" in message
    assert "at least 9999 calibration scores" in message
    assert "Storey-BH" in fdr_records[1].getMessage()


def test_select_does_not_warn_when_floor_permits_a_discovery(fdr_records):
    x_reference, _ = _data(seed=13, n_reference=2_400)
    x_clean = np.random.default_rng(14).normal(size=(50, 3))
    detector = _detector(strategy=Split(n_calib=999)).fit(x_reference)

    assert not detector.select(x_clean, alpha=0.05).any()
    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "Calibration size is not the limit" in message
    assert "cannot fall below" not in message


def test_select_does_not_warn_after_discoveries(fdr_records):
    x_reference, x_test = _data(seed=15, n_reference=400, n_null=200, n_alt=300)
    detector = _detector(strategy=Split(n_calib=49)).fit(x_reference)

    assert detector.select(x_test, alpha=0.2).any()
    assert fdr_records == []


@pytest.mark.parametrize("method", ["bh", "storey"])
def test_select_returns_empty_mask_for_empty_batch(fdr_records, method):
    x_reference, _ = _data(seed=18, n_reference=400)
    detector = _detector(strategy=Split(n_calib=49)).fit(x_reference)

    selected = detector.select(np.empty((0, 3)), alpha=0.05, method=method)
    adjusted = detector.last_result.adjusted_p_values(method=method)

    assert selected.shape == (0,)
    assert selected.dtype == bool
    assert adjusted.shape == (0,)
    assert fdr_records == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"estimation": Empirical(tie_break="randomized")},
        {"estimation": ConditionalEmpirical(method="dkwm")},
        {"weight_estimator": logistic_weight_estimator()},
    ],
    ids=["randomized", "conditional", "weighted"],
)
def test_floor_warning_is_limited_to_classical_empirical_p_values(fdr_records, kwargs):
    x_reference, _ = _data(seed=16, n_reference=400)
    x_clean = np.random.default_rng(17).normal(size=(500, 3))
    detector = _detector(strategy=Split(n_calib=49), **kwargs).fit(x_reference)

    detector.select(x_clean, alpha=0.05)

    messages = [record.getMessage() for record in fdr_records]
    assert not any("cannot fall below" in message for message in messages)
    if "weight_estimator" in kwargs:
        assert messages == []
    else:
        assert len(messages) == 1
        assert "too few calibration scores" in messages[0]
