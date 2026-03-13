import numpy as np
import pytest
from scipy.stats import false_discovery_control

from nonconform.enums import Pruning
from nonconform.fdr import (
    conformalized_selection,
    conformalized_selection_from_arrays,
    weighted_false_discovery_control_from_arrays,
)
from nonconform.scoring import calculate_weighted_p_val
from nonconform.structures import ConformalResult


def _bh_rejection_count(p_values: np.ndarray, alpha: float) -> int:
    m = len(p_values)
    if m == 0:
        return 0
    sorted_p = np.sort(p_values)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _reference_wcs_deterministic(
    *,
    p_values: np.ndarray,
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    test_weights: np.ndarray,
    calib_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    m = len(test_scores)
    sum_calib_weight = float(np.sum(calib_weights))
    r_sizes = np.zeros(m, dtype=float)

    for j in range(m):
        aux_p = np.zeros(m, dtype=float)
        denominator = sum_calib_weight + test_weights[j]
        for k in range(m):
            if k == j:
                continue
            calib_mass = float(np.sum(calib_weights[calib_scores > test_scores[k]]))
            aux_p[k] = (
                calib_mass + test_weights[k] * float(test_scores[k] > test_scores[j])
            ) / denominator
        r_sizes[j] = _bh_rejection_count(aux_p, alpha)

    thresholds = alpha * r_sizes / m
    first_sel_idx = np.flatnonzero(p_values <= thresholds)
    if len(first_sel_idx) == 0:
        return np.zeros(m, dtype=bool)

    sizes_sel = r_sizes[first_sel_idx]
    sorted_sizes = np.sort(sizes_sel)
    r_star = 0
    for k in range(len(sorted_sizes), 0, -1):
        if sorted_sizes[k - 1] <= k:
            r_star = k
            break
    if r_star == 0:
        return np.zeros(m, dtype=bool)

    final_idx = np.sort(first_sel_idx[sizes_sel <= r_star])
    mask = np.zeros(m, dtype=bool)
    mask[final_idx] = True
    return mask


def test_conformalized_selection_matches_scipy_bh() -> None:
    p_values = np.array([0.003, 0.21, 0.04, 0.12, 0.001, 0.08])
    alpha = 0.1

    expected = false_discovery_control(p_values, method="bh") <= alpha
    actual = conformalized_selection_from_arrays(p_values=p_values, alpha=alpha)
    np.testing.assert_array_equal(actual, expected)


def test_conformalized_selection_from_result_bundle() -> None:
    p_values = np.array([0.01, 0.04, 0.5, 0.9], dtype=float)
    result = ConformalResult(p_values=p_values)

    selected = conformalized_selection(result=result, alpha=0.05)
    expected = conformalized_selection_from_arrays(p_values=p_values, alpha=0.05)
    assert selected.dtype == bool
    assert selected.shape == (4,)
    np.testing.assert_array_equal(selected, expected)


def test_conformalized_selection_rejects_missing_p_values() -> None:
    with pytest.raises(ValueError, match="missing required CS field"):
        conformalized_selection(result=ConformalResult(p_values=None), alpha=0.05)


def test_conformalized_selection_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError, match="must be a 1D array"):
        conformalized_selection_from_arrays(
            p_values=np.array([[0.1], [0.2]]),
            alpha=0.1,
        )


def test_weighted_fdr_deterministic_matches_reference_formulation() -> None:
    rng = np.random.default_rng(17)
    calib_scores = rng.normal(size=40)
    test_scores = rng.normal(size=12)
    calib_scores += np.linspace(0, 1e-4, len(calib_scores))
    test_scores += np.linspace(0, 1e-4, len(test_scores))

    calib_weights = rng.uniform(0.4, 1.8, size=len(calib_scores))
    test_weights = rng.uniform(0.4, 1.8, size=len(test_scores))
    p_values = calculate_weighted_p_val(
        scores=test_scores,
        calibration_set=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        tie_break="classical",
    )

    alpha = 0.15
    expected = _reference_wcs_deterministic(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
    )
    actual = weighted_false_discovery_control_from_arrays(
        p_values=p_values,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        alpha=alpha,
        pruning=Pruning.DETERMINISTIC,
        seed=11,
    )
    np.testing.assert_array_equal(actual, expected)
