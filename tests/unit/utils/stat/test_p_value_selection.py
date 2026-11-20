"""Unit tests for BH and Storey-BH selection primitives."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from nonconform._internal import p_value_selection as selection
from nonconform.scoring import calculate_p_val


@pytest.fixture
def fdr_records():
    """Collect records emitted on the nonconform.fdr logger."""

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


def _step_up(p_values: np.ndarray, level: float) -> np.ndarray:
    """Reference step-up rule: reject the k smallest for the largest valid k."""
    m = p_values.size
    order = np.argsort(p_values, kind="mergesort")
    passes = np.flatnonzero(p_values[order] <= level * np.arange(1, m + 1) / m)
    mask = np.zeros(m, dtype=bool)
    if passes.size:
        mask[order[: passes[-1] + 1]] = True
    return mask


def _conformal_p_values(rng, n_calibration, n_test, n_anomalies, shift):
    calibration = rng.standard_normal(n_calibration)
    test = rng.standard_normal(n_test)
    test[:n_anomalies] += shift
    return calculate_p_val(test, calibration)


@pytest.mark.parametrize("method", ["bh", "storey", " BH ", "Storey"])
def test_normalize_method_accepts_case_and_whitespace(method):
    assert selection.normalize_method(method) == method.strip().lower()


def test_normalize_method_rejects_unknown_and_non_string_values():
    with pytest.raises(ValueError, match="'bh' or 'storey'"):
        selection.normalize_method("by")
    with pytest.raises(TypeError, match="'bh' or 'storey'"):
        selection.normalize_method(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_calibration", [1, 2, 3, 4, 9, 10, 99, 100, 1_000])
def test_storey_lambda_is_smallest_grid_point_not_below_one_half(n_calibration):
    lam = selection.storey_lambda(n_calibration)
    k = round(lam * (n_calibration + 1))

    assert lam == k / (n_calibration + 1)
    assert lam >= 0.5
    assert (k - 1) / (n_calibration + 1) < 0.5
    assert 1 <= k <= n_calibration


def test_storey_lambda_matches_classical_p_value_grid_exactly():
    n_calibration = 9
    calibration = np.arange(n_calibration, dtype=float)
    p_values = calculate_p_val(calibration - 0.5, calibration)
    lam = selection.storey_lambda(n_calibration)

    assert lam == 0.5
    assert np.count_nonzero(p_values == lam) == 1
    assert np.count_nonzero(p_values >= lam) == 6


def test_storey_null_proportion_uses_non_strict_count_and_leading_one():
    # n = 9 gives lambda = 5 / 10; three of the six p-values are >= lambda.
    p_values = np.array([0.1, 0.2, 0.4, 0.5, 0.8, 1.0])

    estimate = selection.storey_null_proportion(p_values, n_calibration=9)

    assert estimate == pytest.approx((1 + 3) / (6 * 0.5))


def test_storey_null_proportion_is_not_capped_at_one():
    p_values = np.ones(4)

    estimate = selection.storey_null_proportion(p_values, n_calibration=9)

    assert estimate == pytest.approx(5 / 2)


def test_bh_adjustment_matches_scipy():
    rng = np.random.default_rng(0)
    p_values = rng.uniform(size=50)

    adjusted = selection.adjusted_p_values(p_values, method="bh", n_calibration=99)

    np.testing.assert_array_equal(
        adjusted, false_discovery_control(p_values, method="bh")
    )


def test_storey_adjustment_scales_bh_by_null_proportion_and_caps_at_one():
    rng = np.random.default_rng(1)
    p_values = _conformal_p_values(rng, 99, 60, 20, 3.0)
    estimate = selection.storey_null_proportion(p_values, n_calibration=99)

    adjusted = selection.adjusted_p_values(p_values, method="storey", n_calibration=99)

    expected = np.minimum(1.0, estimate * false_discovery_control(p_values))
    np.testing.assert_allclose(adjusted, expected)
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))


@pytest.mark.parametrize("method", ["bh", "storey"])
def test_empty_family_has_empty_adjustment_and_no_warning(fdr_records, method):
    p_values = np.empty(0)

    adjusted = selection.adjusted_p_values(p_values, method=method, n_calibration=19)
    selection.warn_if_floor_blocks_discovery(
        p_values, alpha=0.1, method=method, n_calibration=19
    )

    assert adjusted.shape == (0,)
    assert fdr_records == []


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3])
def test_storey_adjusted_mask_matches_step_up_at_scaled_level(seed, alpha):
    rng = np.random.default_rng(seed)
    n_calibration = int(rng.integers(5, 200))
    p_values = _conformal_p_values(
        rng, n_calibration, int(rng.integers(1, 80)), int(rng.integers(0, 30)), 2.5
    )
    estimate = selection.storey_null_proportion(p_values, n_calibration=n_calibration)

    mask = (
        selection.adjusted_p_values(
            p_values, method="storey", n_calibration=n_calibration
        )
        <= alpha
    )

    np.testing.assert_array_equal(mask, _step_up(p_values, alpha / estimate))


def test_storey_selects_superset_of_bh_when_null_proportion_is_below_one():
    rng = np.random.default_rng(2)
    p_values = _conformal_p_values(rng, 499, 200, 80, 3.0)
    estimate = selection.storey_null_proportion(p_values, n_calibration=499)
    bh = selection.adjusted_p_values(p_values, method="bh", n_calibration=499)
    storey = selection.adjusted_p_values(p_values, method="storey", n_calibration=499)

    assert estimate < 1.0
    assert np.all(storey[bh <= 0.1] <= 0.1)
    assert np.count_nonzero(storey <= 0.1) > np.count_nonzero(bh <= 0.1)


def _monte_carlo(n_calibration, n_test, n_anomalies, alpha, method, trials, seed):
    rng = np.random.default_rng(seed)
    fdp = np.empty(trials)
    power = np.empty(trials)
    for trial in range(trials):
        calibration = rng.standard_normal(n_calibration)
        test = rng.standard_normal(n_test)
        test[:n_anomalies] += 3.0
        p_values = calculate_p_val(test, calibration)
        mask = (
            selection.adjusted_p_values(
                p_values, method=method, n_calibration=n_calibration
            )
            <= alpha
        )
        n_selected = int(mask.sum())
        fdp[trial] = mask[n_anomalies:].sum() / max(n_selected, 1)
        power[trial] = mask[:n_anomalies].sum() / max(n_anomalies, 1)
    return fdp.mean(), fdp.std() / np.sqrt(trials), power.mean()


@pytest.mark.parametrize(
    ("n_calibration", "n_test", "n_anomalies", "alpha"),
    [
        (19, 50, 0, 0.3),
        (19, 50, 15, 0.3),
        (99, 100, 0, 0.2),
        (99, 100, 40, 0.1),
        (200, 100, 60, 0.1),
    ],
)
def test_storey_controls_fdr_for_classical_conformal_p_values(
    n_calibration, n_test, n_anomalies, alpha
):
    fdr, standard_error, _ = _monte_carlo(
        n_calibration, n_test, n_anomalies, alpha, "storey", 3_000, seed=7
    )

    assert fdr <= alpha + 3 * standard_error


def test_storey_gains_power_when_anomalies_are_common():
    _, _, bh_power = _monte_carlo(200, 100, 60, 0.1, "bh", 1_000, seed=11)
    _, _, storey_power = _monte_carlo(200, 100, 60, 0.1, "storey", 1_000, seed=11)

    assert storey_power > bh_power


def test_floor_warning_reports_bh_requirements(fdr_records):
    p_values = np.full(100, 0.5)

    selection.warn_if_floor_blocks_discovery(
        p_values, alpha=0.05, method="bh", n_calibration=199
    )

    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "1/200 = 0.005" in message
    assert "alpha/m = 0.0005" in message
    assert "at least 10 test points" in message
    assert "at least 1999 calibration scores" in message


def test_floor_warning_reports_impossible_batches(fdr_records):
    selection.warn_if_floor_blocks_discovery(
        np.full(5, 0.5), alpha=0.05, method="bh", n_calibration=10
    )

    assert len(fdr_records) == 1
    assert "No selection is possible" in fdr_records[0].getMessage()


def test_floor_warning_uses_storey_threshold(fdr_records):
    p_values = np.full(100, 0.9)
    estimate = selection.storey_null_proportion(p_values, n_calibration=199)

    selection.warn_if_floor_blocks_discovery(
        p_values, alpha=0.05, method="storey", n_calibration=199
    )

    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "Storey-BH" in message
    assert f"pi0 = {estimate:.3g}" in message


@pytest.mark.parametrize("n_calibration", [1_999, 5_000])
def test_floor_warning_is_silent_when_one_discovery_is_attainable(
    fdr_records, n_calibration
):
    selection.warn_if_floor_blocks_discovery(
        np.full(100, 0.5), alpha=0.05, method="bh", n_calibration=n_calibration
    )

    assert fdr_records == []


def test_floor_warning_reports_whether_it_warned(fdr_records):
    assert selection.warn_if_floor_blocks_discovery(
        np.full(100, 0.5), alpha=0.05, method="bh", n_calibration=199
    )
    assert not selection.warn_if_floor_blocks_discovery(
        np.full(100, 0.5), alpha=0.05, method="bh", n_calibration=5_000
    )


@pytest.mark.parametrize("suggest", [True, False])
def test_no_discovery_warning_reports_the_missed_threshold(fdr_records, suggest):
    p_values = np.linspace(0.02, 1.0, 100)

    selection.warn_no_discoveries(
        p_values,
        alpha=0.05,
        method="bh",
        n_calibration=5_000,
        suggest_fpr_bounds=suggest,
        calibration_ruled_out=True,
    )

    assert len(fdr_records) == 1
    message = fdr_records[0].getMessage()
    assert "smallest p-value in this batch is 0.02" in message
    assert "alpha/m = 0.0005" in message
    assert "Calibration size is not the limit" in message
    assert "score_polarity" in message
    assert ("fpr_bounds()" in message) is suggest


def test_no_discovery_warning_uses_storey_threshold(fdr_records):
    p_values = np.linspace(0.02, 1.0, 100)
    estimate = selection.storey_null_proportion(p_values, n_calibration=5_000)

    selection.warn_no_discoveries(
        p_values,
        alpha=0.05,
        method="storey",
        n_calibration=5_000,
        suggest_fpr_bounds=False,
        calibration_ruled_out=False,
    )

    message = fdr_records[0].getMessage()
    assert "too few calibration scores" in message
    assert f"alpha/(m*pi0) = {0.05 / (100 * estimate):.3g}" in message
