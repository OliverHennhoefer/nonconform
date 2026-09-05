"""Integration tests for weighted FDR control utilities."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import pytest
from pyod.models.iforest import IForest
from scipy.stats import false_discovery_control
from sklearn.ensemble import IsolationForest

from nonconform import (
    ConformalDetector,
    CrossValidation,
    DerandomizedSplits,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform.enums import Kernel, Pruning
from nonconform.fdr import select_conformal_e_values


class RoundedDistanceDetector:
    """Small deterministic detector that intentionally produces score ties."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state
        self.center_: np.ndarray | None = None

    def fit(self, X, y=None) -> Self:
        _ = y
        self.center_ = np.mean(X, axis=0)
        return self

    def decision_function(self, X):
        if self.center_ is None:
            raise RuntimeError("detector is not fitted")
        return np.round(np.linalg.norm(X - self.center_, axis=1))

    def get_params(self, deep=True):
        _ = deep
        return {"random_state": self.random_state}

    def set_params(self, **params) -> Self:
        if "random_state" in params:
            self.random_state = params["random_state"]
        return self


class ContinuousDistanceDetector(RoundedDistanceDetector):
    """Deterministic detector with continuous Euclidean-distance scores."""

    def decision_function(self, X):
        if self.center_ is None:
            raise RuntimeError("detector is not fitted")
        return np.linalg.norm(X - self.center_, axis=1)


@pytest.mark.parametrize("model_type", [IsolationForest, IForest])
def test_derandomized_strategy_real_detector_matches_manual_splits(
    simple_dataset, model_type
):
    x_train, x_test, _ = simple_dataset(n_train=120, n_test=40, n_features=4)
    detector = ConformalDetector(
        detector=model_type(n_estimators=10),
        strategy=DerandomizedSplits(3, 0.3, tie_seed=99),
        seed=7,
    ).fit(x_train)
    mask = detector.select(x_test, alpha=0.2)
    results = []
    for stream in np.random.SeedSequence(7).spawn(2)[0].spawn(3):
        split_seed = int(stream.generate_state(1)[0])
        single = ConformalDetector(
            detector=model_type(n_estimators=10),
            strategy=Split(0.3),
            seed=split_seed,
        ).fit(x_train)
        single.score_samples(x_test)
        results.append(single.last_result)
    expected = select_conformal_e_values(results, alpha=0.2, tie_seed=99)
    np.testing.assert_array_equal(mask, expected.selected)
    np.testing.assert_array_equal(
        detector.last_selection_result.e_values, expected.e_values
    )


def _fit_weighted_detector(x_train):
    detector = ConformalDetector(
        detector=IForest(n_estimators=30, max_samples=0.8, random_state=0),
        strategy=Split(n_calib=0.2),
        estimation=Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0),
        weight_estimator=logistic_weight_estimator(),
        seed=4,
    )
    detector.fit(x_train)
    return detector


@pytest.mark.parametrize("pruning", list(Pruning))
def test_pruning_modes_control_false_discoveries(simple_dataset, pruning):
    """select() should support weighted pruning modes end-to-end."""
    x_train, x_test, y_test = simple_dataset(n_train=120, n_test=60, n_features=5)
    detector = _fit_weighted_detector(x_train)

    selections = detector.select(
        x_test,
        alpha=0.25,
        pruning=pruning,
        seed=0,
    )
    assert selections.dtype == bool
    assert selections.shape == (len(x_test),)

    discoveries = int(np.count_nonzero(selections))
    if discoveries > 0:
        false_pos = int(np.count_nonzero(selections & (y_test == 0)))
        observed_fdr = false_pos / discoveries
        assert observed_fdr <= 0.35  # empirical control with generous slack


def test_standard_bh_on_weighted_pvalues_respects_ordering(simple_dataset):
    """Selected discoveries must correspond to the smallest p-values."""
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=50, n_features=4)
    detector = _fit_weighted_detector(x_train)
    detector.compute_p_values(x_test)
    result = detector.last_result
    assert result is not None and result.p_values is not None

    mask = false_discovery_control(result.p_values, method="bh") <= 0.2
    assert mask.shape == (len(x_test),)

    if np.any(mask):
        max_sel = np.max(result.p_values[mask])
        assert np.all(result.p_values[~mask] >= max_sel - 1e-12)


def test_repeated_split_results_feed_select_conformal_e_values(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=120, n_test=40, n_features=4)

    def collect_result():
        results = []
        for seed in range(4):
            detector = ConformalDetector(
                detector=IsolationForest(n_estimators=20, random_state=seed),
                strategy=Split(n_calib=0.3),
                score_polarity="auto",
                seed=seed,
            )
            detector.fit(x_train)
            detector.score_samples(x_test)
            result = detector.last_result
            assert result is not None
            assert result.metadata == {}
            results.append(result)

        return select_conformal_e_values(
            results,
            alpha=0.2,
            tie_seed=2026,
        )

    first = collect_result()
    second = collect_result()

    assert first.selected.dtype == bool
    assert first.selected.shape == (len(x_test),)
    assert first.e_values.shape == (len(x_test),)
    assert first.n_repetitions == 4
    assert first.n_calibration == 36
    np.testing.assert_array_equal(first.selected, second.selected)
    np.testing.assert_allclose(first.e_values, second.e_values)


def test_discrete_detector_requires_seeded_randomized_score_ties(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=120, n_test=40, n_features=4)
    results = []
    for seed in range(3):
        detector = ConformalDetector(
            detector=RoundedDistanceDetector(random_state=seed),
            strategy=Split(n_calib=0.3),
            score_polarity="higher_is_anomalous",
            seed=seed,
        ).fit(x_train)
        detector.score_samples(x_test)
        result = detector.last_result
        assert result is not None
        results.append(result)

    with pytest.raises(ValueError, match="tied scores"):
        select_conformal_e_values(results, alpha=0.2)

    first = select_conformal_e_values(
        results,
        alpha=0.2,
        tie_seed=99,
    )
    replay = select_conformal_e_values(
        results,
        alpha=0.2,
        tie_seed=99,
    )

    np.testing.assert_array_equal(first.e_values, replay.e_values)
    np.testing.assert_array_equal(first.selected, replay.selected)


def test_continuous_detector_uses_default_strict_score_policy(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=30, n_features=4)
    results = []
    for seed in range(3):
        detector = ConformalDetector(
            detector=ContinuousDistanceDetector(random_state=seed),
            strategy=Split(n_calib=0.3),
            score_polarity="higher_is_anomalous",
            seed=seed,
        ).fit(x_train)
        detector.score_samples(x_test)
        result = detector.last_result
        assert result is not None
        results.append(result)

    selection = select_conformal_e_values(results, alpha=0.2)

    assert selection.tie_seed is None
    assert selection.selected.shape == (len(x_test),)


def test_score_and_p_value_results_share_native_provenance(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    detector = ConformalDetector(
        detector=IsolationForest(n_estimators=10, random_state=4),
        strategy=Split(n_calib=0.25),
        score_polarity="higher_is_normal",
        seed=4,
    ).fit(x_train)

    detector.score_samples(x_test)
    score_result = detector.last_result
    detector.compute_p_values(x_test)
    p_value_result = detector.last_result

    assert score_result is not None
    assert p_value_result is not None
    assert score_result.metadata == {}
    assert p_value_result.metadata["nonconform"] == {
        "strategy": "Split",
        "estimation": "Empirical",
        "weighted": False,
    }
    assert score_result._provenance == p_value_result._provenance


def test_detached_calibration_result_is_identified_and_rejected(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    base = IsolationForest(n_estimators=10, random_state=7).fit(x_train[:50])
    detector = ConformalDetector(
        detector=base,
        strategy=Split(n_calib=0.25),
        score_polarity="higher_is_normal",
    ).calibrate(x_train[50:])
    detector.score_samples(x_test)
    result = detector.last_result

    assert result is not None
    with pytest.raises(ValueError, match="integrated Split"):
        select_conformal_e_values([result])


def test_split_subclass_is_accepted_by_native_provenance(simple_dataset):
    class CustomSplit(Split):
        pass

    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    detector = ConformalDetector(
        detector=ContinuousDistanceDetector(),
        strategy=CustomSplit(n_calib=0.25),
        score_polarity="higher_is_anomalous",
        seed=1,
    ).fit(x_train)
    detector.score_samples(x_test)
    result = detector.last_result

    assert result is not None
    selection = select_conformal_e_values([result], alpha=0.2)
    assert selection.n_repetitions == 1


def test_unrelated_strategy_named_split_is_rejected(simple_dataset):
    fake_split_type = type("Split", (CrossValidation,), {})
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    detector = ConformalDetector(
        detector=ContinuousDistanceDetector(),
        strategy=fake_split_type(k=3),
        score_polarity="higher_is_anomalous",
        seed=1,
    ).fit(x_train)
    detector.score_samples(x_test)
    result = detector.last_result

    assert result is not None
    assert result.metadata == {}
    with pytest.raises(ValueError, match="integrated Split"):
        select_conformal_e_values([result], tie_seed=1)


@pytest.mark.parametrize("mutation", ["changed", "reordered"])
def test_result_selection_verifies_test_batch_identity(simple_dataset, mutation):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    detector = ConformalDetector(
        detector=ContinuousDistanceDetector(),
        strategy=Split(n_calib=0.25),
        score_polarity="higher_is_anomalous",
        seed=1,
    ).fit(x_train)
    detector.score_samples(x_test)
    first = detector.last_result.copy()

    changed_test = x_test.copy()
    if mutation == "changed":
        changed_test[0, 0] += 1.0
    else:
        changed_test = changed_test[::-1]
    detector.score_samples(changed_test)
    second = detector.last_result

    assert second is not None
    with pytest.raises(ValueError, match=r"identical test batch.*same row order"):
        select_conformal_e_values([first, second], tie_seed=1)


def test_result_selection_accepts_identical_test_batch_copy(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    detector = ConformalDetector(
        detector=ContinuousDistanceDetector(),
        strategy=Split(n_calib=0.25),
        score_polarity="higher_is_anomalous",
        seed=1,
    ).fit(x_train)
    detector.score_samples(x_test)
    first = detector.last_result.copy()
    detector.score_samples(x_test.copy())
    second = detector.last_result

    assert second is not None
    selection = select_conformal_e_values([first, second], tie_seed=1)
    assert selection.n_repetitions == 2


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
@pytest.mark.parametrize("method", ["score_samples", "compute_p_values"])
def test_result_selection_verifies_nullable_dataframe_batches(
    simple_dataset, dtype, method
):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    batch = pd.DataFrame(np.round(x_test * 100), dtype=dtype)
    results = []
    for seed in range(2):
        detector = ConformalDetector(
            detector=IsolationForest(n_estimators=10, random_state=seed),
            strategy=Split(n_calib=0.25),
            score_polarity="higher_is_normal",
            seed=seed,
        ).fit(x_train * 100)
        getattr(detector, method)(batch.copy())
        results.append(detector.last_result)

    selection = select_conformal_e_values(results, alpha=0.2, tie_seed=1)
    assert selection.n_repetitions == 2
    assert selection.selected.shape == (len(batch),)

    for changed_batch in (batch + 1, batch.iloc[::-1]):
        getattr(detector, method)(changed_batch)
        with pytest.raises(ValueError, match="identical test batch"):
            select_conformal_e_values([results[0], detector.last_result], tie_seed=1)


def test_result_selection_does_not_depend_on_p_value_estimator(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=80, n_test=12, n_features=3)
    results = []
    for estimation in (None, Probabilistic(kernel=[Kernel.GAUSSIAN], n_trials=0)):
        kwargs = {} if estimation is None else {"estimation": estimation}
        detector = ConformalDetector(
            detector=ContinuousDistanceDetector(),
            strategy=Split(n_calib=0.25),
            score_polarity="higher_is_anomalous",
            seed=1,
            **kwargs,
        ).fit(x_train)
        detector.score_samples(x_test)
        result = detector.last_result
        assert result is not None
        results.append(result)

    selection = select_conformal_e_values(results, alpha=0.2, tie_seed=1)
    assert selection.n_repetitions == 2


def test_result_selection_rejects_weighted_detector_result(simple_dataset):
    x_train, x_test, _ = simple_dataset(n_train=100, n_test=20, n_features=3)
    detector = _fit_weighted_detector(x_train)
    detector.score_samples(x_test)
    result = detector.last_result

    assert result is not None
    with pytest.raises(ValueError, match="weighted scores"):
        select_conformal_e_values([result], tie_seed=1)
