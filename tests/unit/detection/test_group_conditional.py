"""Group-conditional (Mondrian) calibration through ConformalDetector."""

from __future__ import annotations

import logging
import pickle
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
    JackknifeBootstrap,
    Probabilistic,
    Split,
    logistic_weight_estimator,
)
from nonconform._internal.detector_state import FittedCalibration
from nonconform._internal.provenance import ResultProvenance
from nonconform.fdr import select_conformal_e_values
from nonconform.resampling import BaseStrategy
from nonconform.scoring import ConditionalEmpirical

GROUP_SHIFT = 2.0


class FirstFeatureDetector:
    """Score each row by its first feature; fitting learns nothing."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X, y=None) -> Self:
        _ = X, y
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float)[:, 0].copy()

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


def _rows(rng, labels):
    """Rows whose first feature is shifted upward in group ``b``."""
    labels = np.asarray(labels)
    x = rng.normal(size=(labels.size, 2))
    x[:, 0] += GROUP_SHIFT * (labels == "b")
    return x


def _reference(seed=0, n=600, share_b=0.2):
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.array(["a", "b"]), size=n, p=[1 - share_b, share_b])
    return _rows(rng, labels), labels


def _test_batch(seed=1, n_a=20, n_b=20):
    rng = np.random.default_rng(seed)
    labels = np.array(["a"] * n_a + ["b"] * n_b)
    return _rows(rng, labels), labels


def _detector(strategy=None, **kwargs):
    return ConformalDetector(
        FirstFeatureDetector(),
        strategy if strategy is not None else Split(n_calib=0.5),
        score_polarity="higher_is_anomalous",
        seed=kwargs.pop("seed", 3),
        **kwargs,
    )


def _manual_group_p_values(detector, scores, labels):
    """Rank every score among the calibration scores of its own group."""
    calibration = detector.calibration_set
    calibration_labels = detector.calibration_groups
    expected = np.empty(scores.size)
    for label in np.unique(labels):
        rows = labels == label
        expected[rows] = Empirical().compute_p_values(
            scores[rows], calibration[calibration_labels == label]
        )
    return expected


@pytest.mark.parametrize(
    "strategy",
    [
        Split(n_calib=0.5),
        CrossValidation(k=3),
        JackknifeBootstrap(n_bootstraps=4),
    ],
    ids=["split", "cv", "bootstrap"],
)
def test_p_values_rank_within_the_test_point_group(strategy):
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector(strategy).fit(x_reference, groups=g_reference)

    # Scores equal the first feature (bootstrap averages repeated copies), so
    # each calibration score identifies its reference row; its recorded label
    # must be that row's label.
    distance = np.abs(detector.calibration_set[:, None] - x_reference[None, :, 0])
    assert distance.min(axis=1).max() < 1e-12
    rows = distance.argmin(axis=1)
    np.testing.assert_array_equal(detector.calibration_groups, g_reference[rows])

    p_values = detector.compute_p_values(x_test, groups=g_test)
    np.testing.assert_allclose(
        p_values, _manual_group_p_values(detector, x_test[:, 0], g_test)
    )
    pooled = Empirical().compute_p_values(x_test[:, 0], detector.calibration_set)
    assert not np.allclose(p_values, pooled)


def test_detached_calibration_with_groups():
    x_calibration, g_calibration = _reference(seed=4, n=300)
    x_test, g_test = _test_batch()
    detector = _detector()
    detector.calibrate(x_calibration, groups=g_calibration)

    np.testing.assert_array_equal(detector.calibration_groups, g_calibration)
    np.testing.assert_allclose(
        detector.compute_p_values(x_test, groups=g_test),
        _manual_group_p_values(detector, x_test[:, 0], g_test),
    )


def test_compute_p_value_matches_batch_value():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference, groups=g_reference)
    batch = detector.compute_p_values(x_test, groups=g_test)

    for row in (0, len(g_test) - 1):
        assert detector.compute_p_value(x_test[row], group=g_test[row]) == batch[row]


def test_select_applies_bh_to_group_conditional_p_values():
    x_reference, g_reference = _reference(n=2_000)
    rng = np.random.default_rng(7)
    g_test = np.array(["a"] * 30 + ["b"] * 30)
    x_test = _rows(rng, g_test)
    x_test[:8, 0] += 5.0
    x_test[30:38, 0] += 5.0
    detector = _detector().fit(x_reference, groups=g_reference)

    selected = detector.select(x_test, alpha=0.1, groups=g_test)
    result = detector.last_result
    assert result is not None
    expected = false_discovery_control(result.p_values, method="bh") <= 0.1
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(result.adjusted_p_values() <= 0.1, selected)
    assert selected[:8].all()
    assert selected[30:38].all()


@pytest.mark.parametrize(
    "as_labels",
    [
        list,
        pd.Series,
        pd.Categorical,
        lambda labels: np.where(labels == "a", 0, 1),
        lambda labels: [(str(label), 1) for label in labels],
    ],
    ids=["list", "series", "categorical", "integers", "tuples"],
)
def test_label_containers_give_identical_p_values(as_labels):
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    expected = (
        _detector()
        .fit(x_reference, groups=g_reference)
        .compute_p_values(x_test, groups=g_test)
    )

    detector = _detector().fit(x_reference, groups=as_labels(g_reference))
    p_values = detector.compute_p_values(x_test, groups=as_labels(g_test))
    np.testing.assert_array_equal(p_values, expected)


def test_numpy_and_python_integer_labels_match():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference, groups=np.where(g_reference == "a", 0, 1))

    expected = detector.compute_p_values(x_test, groups=np.where(g_test == "a", 0, 1))
    python_ints = [0 if label == "a" else 1 for label in g_test]
    np.testing.assert_array_equal(
        detector.compute_p_values(x_test, groups=python_ints), expected
    )


def test_pandas_input_preserves_index():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    index = pd.Index([f"row{i}" for i in range(len(g_test))])
    frame = pd.DataFrame(x_test, index=index)
    detector = _detector().fit(pd.DataFrame(x_reference), groups=g_reference)

    p_values = detector.compute_p_values(frame, groups=pd.Series(g_test, index=index))
    selected = detector.select(frame, alpha=0.1, groups=g_test)
    assert isinstance(p_values, pd.Series)
    assert isinstance(selected, pd.Series)
    assert p_values.index.equals(index)
    assert selected.index.equals(index)


def test_group_conditional_p_values_survive_a_group_mix_shift():
    """Pooled p-values break when the test mix differs; grouped ones do not."""
    n_trials, alpha = 300, 0.1
    pooled_fdp, grouped_fdp = [], []
    rng = np.random.default_rng(11)
    for _ in range(n_trials):
        # The reference is 90% group a; the test nulls are half group b, whose
        # scores are higher, plus ten group-a anomalies.
        g_reference = rng.choice(np.array(["a", "b"]), size=400, p=[0.9, 0.1])
        x_reference = _rows(rng, g_reference)
        g_test = np.r_[rng.choice(np.array(["a", "b"]), size=40), ["a"] * 10]
        x_test = _rows(rng, g_test)
        x_test[40:, 0] += 4.0
        is_null = np.arange(50) < 40
        seed = int(rng.integers(2**31))

        for groups, fdp in ((None, pooled_fdp), (g_reference, grouped_fdp)):
            detector = _detector(seed=seed).fit(x_reference, groups=groups)
            test_groups = None if groups is None else g_test
            selected = np.asarray(
                detector.select(x_test, alpha=alpha, groups=test_groups)
            )
            fdp.append((selected & is_null).sum() / max(selected.sum(), 1))

    pooled, grouped = np.array(pooled_fdp), np.array(grouped_fdp)
    grouped_se = grouped.std(ddof=1) / np.sqrt(n_trials)
    assert grouped.mean() <= alpha + 3 * grouped_se
    assert pooled.mean() > 2 * alpha


def test_group_conditional_null_p_values_are_super_uniform_per_group():
    n_trials, level = 400, 0.2
    hits = {"a": [], "b": []}
    rng = np.random.default_rng(5)
    for _ in range(n_trials):
        g_reference = rng.choice(np.array(["a", "b"]), size=200, p=[0.7, 0.3])
        detector = _detector(seed=int(rng.integers(2**31))).fit(
            _rows(rng, g_reference), groups=g_reference
        )
        g_test = np.array(["a", "b"] * 5)
        p_values = detector.compute_p_values(_rows(rng, g_test), groups=g_test)
        for label in hits:
            hits[label].append(np.mean(p_values[g_test == label] <= level))

    for label, values in hits.items():
        values = np.asarray(values)
        se = values.std(ddof=1) / np.sqrt(n_trials)
        assert values.mean() <= level + 3 * se, label


def test_floor_warning_uses_the_largest_test_group(fdr_records):
    rng = np.random.default_rng(2)
    g_calibration = np.array(["a"] * 60 + ["b"] * 20)
    detector = _detector()
    detector.calibrate(_rows(rng, g_calibration), groups=g_calibration)
    g_test = np.array(["b"] * 50)

    selected = detector.select(_rows(rng, g_test), alpha=0.05, groups=g_test)
    assert not selected.any()
    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "at most 20 calibration scores per test group" in message
    assert "1/21" in message
    assert "at least 999 calibration scores per group" in message


def test_grouped_empty_selection_keeps_calibration_size_as_a_cause(fdr_records):
    rng = np.random.default_rng(4)
    g_calibration = np.array(["a"] * 1_000 + ["b"] * 9)
    detector = _detector()
    detector.calibrate(_rows(rng, g_calibration), groups=g_calibration)
    g_test = np.array(["a"] * 99 + ["b"])

    selected = detector.select(_rows(rng, g_test), alpha=0.1, groups=g_test)

    assert not selected.any()
    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "too few calibration scores" in message
    assert "Calibration size is not the limit" not in message
    assert "fpr_bounds()" not in message


def test_empty_grouped_batch_selects_nothing_without_warning(fdr_records):
    x_reference, g_reference = _reference()
    detector = _detector().fit(x_reference, groups=g_reference)

    selected = detector.select(np.empty((0, 2)), alpha=0.1, groups=[])

    assert selected.shape == (0,)
    assert fdr_records == []


def test_refit_without_groups_returns_to_pooled_calibration():
    x_reference, g_reference = _reference()
    x_test, _ = _test_batch()
    detector = _detector().fit(x_reference, groups=g_reference)
    detector.fit(x_reference)

    assert detector.calibration_groups is None
    np.testing.assert_allclose(
        detector.compute_p_values(x_test),
        Empirical().compute_p_values(x_test[:, 0], detector.calibration_set),
    )


def test_calibration_groups_is_a_copy_and_none_when_ungrouped():
    x_reference, g_reference = _reference()
    assert _detector().calibration_groups is None
    assert _detector().fit(x_reference).calibration_groups is None

    detector = _detector().fit(x_reference, groups=g_reference)
    labels = detector.calibration_groups
    labels[:] = "changed"
    assert not np.any(detector.calibration_groups == "changed")


def test_grouped_detector_pickle_roundtrip():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference, groups=g_reference)
    expected = detector.compute_p_values(x_test, groups=g_test)

    restored = pickle.loads(pickle.dumps(detector))
    np.testing.assert_array_equal(
        restored.calibration_groups, detector.calibration_groups
    )
    np.testing.assert_array_equal(
        restored.compute_p_values(x_test, groups=g_test), expected
    )


def test_state_pickled_before_groups_restores_as_pooled():
    x_reference, _ = _reference()
    detector = _detector().fit(x_reference)
    calibration = detector._calibration
    provenance = detector._result_provenance(None)

    old_calibration = FittedCalibration.__new__(FittedCalibration)
    old_calibration.__setstate__(calibration.__getstate__()[:-1])
    assert old_calibration.groups is None
    np.testing.assert_array_equal(old_calibration.scores, calibration.scores)

    old_provenance = ResultProvenance.__new__(ResultProvenance)
    old_provenance.__setstate__(provenance.__getstate__()[:-1])
    assert old_provenance.grouped is False
    assert old_provenance.tie_break is provenance.tie_break


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"weight_estimator": logistic_weight_estimator()},
            "unavailable in weighted mode",
        ),
        ({"strategy": DerandomizedSplits(2, 0.5)}, "DerandomizedSplits"),
        ({"estimation": Empirical(tie_break="randomized")}, "tie_break='classical'"),
        ({"estimation": ConditionalEmpirical(method="simes")}, "do not support groups"),
        ({"estimation": Probabilistic(n_trials=0)}, "do not support groups"),
    ],
    ids=["weighted", "derandomized", "randomized", "conditional", "kde"],
)
def test_unsupported_configurations_reject_groups(kwargs, match):
    x_reference, g_reference = _reference()
    detector = _detector(**kwargs)
    with pytest.raises(ValueError, match=match):
        detector.fit(x_reference, groups=g_reference)
    assert not detector.is_fitted
    # The same configurations still fit without groups.
    detector.fit(x_reference)
    assert detector.is_fitted


def test_detached_calibration_rejects_unsupported_groups_before_reset():
    x_reference, g_reference = _reference()
    detector = _detector(estimation=Empirical(tie_break="randomized"))
    detector.calibrate(x_reference)
    with pytest.raises(ValueError, match="tie_break='classical'"):
        detector.calibrate(x_reference, groups=g_reference)
    assert detector.is_fitted


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (lambda g: g[:-1], "one label per row"),
        (lambda g: np.c_[g, g], "one-dimensional"),
        (lambda g: np.c_[g, g].tolist(), "one-dimensional"),
        (lambda g: np.r_[g[:-1], [None]], "missing labels"),
        (lambda g: np.r_[np.zeros(g.size - 1), [np.nan]], "missing labels"),
        (lambda g: "a", "one-dimensional"),
    ],
    ids=["short", "two-dimensional", "nested-list", "none", "nan", "scalar"],
)
def test_invalid_reference_labels_are_rejected(labels, match):
    x_reference, g_reference = _reference()
    detector = _detector()
    with pytest.raises(ValueError, match=match):
        detector.fit(x_reference, groups=labels(g_reference))
    with pytest.raises(ValueError, match=match):
        detector.calibrate(x_reference, groups=labels(g_reference))


def test_grouped_detector_requires_valid_test_labels():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference, groups=g_reference)

    with pytest.raises(ValueError, match="calibrated by group"):
        detector.compute_p_values(x_test)
    with pytest.raises(ValueError, match="calibrated by group"):
        detector.select(x_test)
    with pytest.raises(ValueError, match="calibrated by group"):
        detector.compute_p_value(x_test[0])
    with pytest.raises(ValueError, match=r"not passed to fit\(\).*'c'"):
        detector.compute_p_values(x_test, groups=np.r_[g_test[:-1], ["c"]])
    with pytest.raises(ValueError, match="one label per row"):
        detector.compute_p_values(x_test, groups=g_test[:-1])
    with pytest.raises(ValueError, match="missing labels"):
        detector.compute_p_value(x_test[0], group=np.nan)


def test_ungrouped_detector_rejects_test_labels():
    x_reference, _ = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference)

    with pytest.raises(ValueError, match="fitted or calibrated with groups"):
        detector.compute_p_values(x_test, groups=g_test)
    with pytest.raises(ValueError, match="fitted or calibrated with groups"):
        detector.select(x_test, groups=g_test)
    with pytest.raises(ValueError, match="fitted or calibrated with groups"):
        detector.compute_p_value(x_test[0], group="a")


def test_derandomized_select_rejects_test_labels():
    x_reference, _ = _reference()
    x_test, g_test = _test_batch()
    detector = _detector(DerandomizedSplits(2, 0.5)).fit(x_reference)
    with pytest.raises(ValueError, match="DerandomizedSplits"):
        detector.select(x_test, groups=g_test)


def test_group_without_calibration_scores_is_rejected():
    x_reference, g_reference = _reference()
    # One reference row for group c; find a split seed that trains on it.
    g_reference = np.r_[g_reference, ["c"]]
    x_reference = np.vstack([x_reference, np.zeros((1, 2))])
    for seed in range(50):
        detector = _detector(seed=seed).fit(x_reference, groups=g_reference)
        if "c" not in detector.calibration_groups:
            break
    else:
        pytest.fail("no split seed put the single group-c row into training")

    x_test, g_test = _test_batch()
    with pytest.raises(ValueError, match="No calibration scores for groups 'c'"):
        detector.compute_p_values(x_test, groups=np.r_[g_test[:-1], ["c"]])
    # Groups that do have calibration scores still work.
    assert detector.compute_p_values(x_test, groups=g_test).shape == (len(g_test),)


def test_storey_and_certificates_reject_grouped_calibration():
    x_reference, g_reference = _reference()
    x_test, g_test = _test_batch()
    detector = _detector().fit(x_reference, groups=g_reference)

    with pytest.raises(ValueError, match="method='storey' does not support group"):
        detector.select(x_test, method="storey", groups=g_test)
    assert detector.last_result is None
    with pytest.raises(ValueError, match="group-conditional"):
        detector.fdp_bounds(x_test)
    with pytest.raises(ValueError, match="group-conditional"):
        detector.fpr_bounds()

    detector.compute_p_values(x_test, groups=g_test)
    result = detector.last_result
    assert result is not None
    with pytest.raises(ValueError, match="method='storey' does not support group"):
        result.adjusted_p_values(method="storey")
    with pytest.raises(ValueError, match="group-conditional"):
        result.fdp_bounds()
    with pytest.raises(ValueError, match="group-conditional"):
        result.fpr_bounds()

    detector.score_samples(x_test)
    with pytest.raises(ValueError, match="group-conditional calibration"):
        select_conformal_e_values([detector.last_result], alpha=0.1)


def test_unsupported_strategy_rejects_groups():
    class HoldoutTail(BaseStrategy):
        """Custom strategy outside the supported group-conditional list."""

        def fit_calibrate(self, x, detector, seed=None, weighted=False):
            _ = seed, weighted
            detector.fit(x[:-50])
            self._calibration_ids = list(range(len(x) - 50, len(x)))
            return [detector], detector.decision_function(x[-50:])

        @property
        def calibration_ids(self):
            return self._calibration_ids.copy()

    x_reference, g_reference = _reference()
    detector = _detector(HoldoutTail())
    with pytest.raises(ValueError, match="got HoldoutTail"):
        detector.fit(x_reference, groups=g_reference)
    assert detector.fit(x_reference).is_fitted
