import numpy as np

from nonconform.strategy.estimation.probabilistic import Probabilistic
from nonconform.utils.stat import weighted_fdr
from nonconform.utils.stat.results import ConformalResult
from nonconform.utils.stat.statistical import calculate_weighted_p_val
from nonconform.utils.stat.weighted_fdr import weighted_false_discovery_control


def _expected_selection(first_sel_idx: np.ndarray, metrics: np.ndarray) -> np.ndarray:
    sorted_metrics = np.sort(metrics)
    r_star = 0
    for k in range(sorted_metrics.size, 0, -1):
        if sorted_metrics[k - 1] <= k:
            r_star = k
            break
    if r_star == 0:
        return np.array([], dtype=int)
    selected = first_sel_idx[metrics <= r_star]
    return np.sort(selected)


def test_prune_deterministic_applies_threshold():
    first_sel_idx = np.array([2, 0, 1])
    sizes_sel = np.array([5, 1, 2])

    result = weighted_fdr._prune_deterministic(first_sel_idx, sizes_sel)
    expected = _expected_selection(first_sel_idx, sizes_sel.astype(float))

    np.testing.assert_array_equal(result, expected)


def test_prune_heterogeneous_matches_reference_metrics():
    first_sel_idx = np.array([5, 2, 7, 1])
    sizes_sel = np.array([4, 3, 6, 2], dtype=float)

    seed = 12345
    rng = np.random.default_rng(seed)
    result = weighted_fdr._prune_heterogeneous(first_sel_idx, sizes_sel, rng)

    reference_rng = np.random.default_rng(seed)
    xi = reference_rng.uniform(size=len(first_sel_idx))
    metrics = xi * sizes_sel
    expected = _expected_selection(first_sel_idx, metrics)

    np.testing.assert_array_equal(result, expected)


def test_prune_homogeneous_uses_shared_random_variable():
    first_sel_idx = np.array([4, 0, 3])
    sizes_sel = np.array([3, 5, 2], dtype=float)

    seed = 987
    rng = np.random.default_rng(seed)
    result = weighted_fdr._prune_homogeneous(first_sel_idx, sizes_sel, rng)

    reference_rng = np.random.default_rng(seed)
    xi = reference_rng.uniform()
    metrics = xi * sizes_sel
    expected = _expected_selection(first_sel_idx, metrics)

    np.testing.assert_array_equal(result, expected)


def test_weighted_false_discovery_control_matches_internal_computation():
    calib_scores = np.array([1.0, 2.5, 3.0])
    test_scores = np.array([2.0, 0.5])
    calib_weights = np.array([1.0, 1.0, 1.0])
    test_weights = np.array([0.5, 0.5])

    p_values = calculate_weighted_p_val(
        test_scores, calib_scores, test_weights, calib_weights
    )

    bundle = ConformalResult(
        p_values=p_values.copy(),
        test_scores=test_scores.copy(),
        calib_scores=calib_scores.copy(),
        test_weights=test_weights.copy(),
        calib_weights=calib_weights.copy(),
    )

    result_explicit = weighted_false_discovery_control(
        p_values=p_values,
        alpha=0.2,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=123,
    )

    result_from_bundle = weighted_false_discovery_control(
        result=bundle,
        alpha=0.2,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=123,
    )

    np.testing.assert_array_equal(result_from_bundle, result_explicit)


def test_weighted_false_discovery_control_handles_missing_p_values_in_bundle():
    calib_scores = np.array([0.2, 0.6, 1.1])
    test_scores = np.array([0.5, 0.9])
    calib_weights = np.array([0.8, 1.2, 1.0])
    test_weights = np.array([0.7, 0.9])

    bundle = ConformalResult(
        p_values=None,
        test_scores=test_scores.copy(),
        calib_scores=calib_scores.copy(),
        test_weights=test_weights.copy(),
        calib_weights=calib_weights.copy(),
    )

    direct = weighted_false_discovery_control(
        p_values=None,
        alpha=0.1,
        test_scores=test_scores,
        calib_scores=calib_scores,
        test_weights=test_weights,
        calib_weights=calib_weights,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=212,
    )

    via_bundle = weighted_false_discovery_control(
        result=bundle,
        alpha=0.1,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=212,
    )

    np.testing.assert_array_equal(via_bundle, direct)


def test_weighted_false_discovery_control_uses_kde_metadata_without_self_weight():
    calib_scores = np.array([0.2, 0.6, 1.4, 1.8])
    test_scores = np.array([0.8, 1.5])
    calib_weights = np.array([2.5, 2.5, 2.5, 2.5])
    test_weights = np.array([1.0, 1.0])
    total_weight = float(calib_weights.sum())

    p_values = np.array([0.4, 0.0])
    eval_grid = np.array([0.0, 1.0, 2.0])
    cdf_values = np.array([0.0, 0.6, 1.0])

    bundle = ConformalResult(
        p_values=p_values.copy(),
        test_scores=test_scores.copy(),
        calib_scores=calib_scores.copy(),
        test_weights=test_weights.copy(),
        calib_weights=calib_weights.copy(),
        metadata={
            "kde": {
                "eval_grid": eval_grid.copy(),
                "cdf_values": cdf_values.copy(),
                "total_weight": total_weight,
            }
        },
    )

    mask_from_values = weighted_false_discovery_control(
        result=bundle,
        p_values=p_values,
        alpha=0.5,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=42,
    )

    mask_from_metadata = weighted_false_discovery_control(
        result=bundle,
        p_values=None,
        alpha=0.5,
        pruning=weighted_fdr.Pruning.DETERMINISTIC,
        seed=42,
    )

    np.testing.assert_array_equal(mask_from_values, mask_from_metadata)


def test_probabilistic_estimator_allows_zero_p_values():
    calib_scores = np.array([0.1, 0.3, 0.5, 0.7])
    test_scores = np.array([0.9, 0.2])
    weights = (np.ones_like(calib_scores), np.ones_like(test_scores))

    estimator = Probabilistic(n_trials=0)
    p_values = estimator.compute_p_values(test_scores, calib_scores, weights)

    assert np.min(p_values) == 0.0
