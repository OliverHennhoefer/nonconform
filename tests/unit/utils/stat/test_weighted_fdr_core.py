import numpy as np
import pytest
from scipy.stats import false_discovery_control

from nonconform.enums import Pruning
from nonconform.fdr import (
    weighted_false_discovery_control,
    weighted_false_discovery_control_from_arrays,
)
from nonconform.scoring import calculate_weighted_p_val


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


class TestBasicFDRControl:
    def test_returns_boolean_array(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert isinstance(discoveries, np.ndarray)
        assert discoveries.dtype == bool

    def test_output_length_matches_test_set(self, conformal_result):
        result = conformal_result(n_test=30, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 30

    def test_with_different_alpha_levels(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)

        discoveries_001 = weighted_false_discovery_control(result=result, alpha=0.01)
        discoveries_010 = weighted_false_discovery_control(result=result, alpha=0.10)
        discoveries_020 = weighted_false_discovery_control(result=result, alpha=0.20)

        assert (
            np.sum(discoveries_001)
            <= np.sum(discoveries_010)
            <= np.sum(discoveries_020)
        )

    def test_with_direct_inputs(self, sample_scores, sample_weights, sample_p_values):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)
        p_values = sample_p_values(n=15)

        discoveries = weighted_false_discovery_control_from_arrays(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )

        assert len(discoveries) == 15


class TestWeightedBH:
    def test_returns_boolean_array(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = false_discovery_control(result.p_values, method="bh") <= 0.1
        assert isinstance(discoveries, np.ndarray)
        assert discoveries.dtype == bool

    def test_output_length_matches_test_set(self, conformal_result):
        result = conformal_result(n_test=25, n_calib=100)
        discoveries = false_discovery_control(result.p_values, method="bh") <= 0.1
        assert len(discoveries) == 25

    def test_low_p_values_discovered(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries = false_discovery_control(result.p_values, method="bh") <= 0.2
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 20

    def test_accepts_direct_p_values(self, sample_p_values):
        p_values = sample_p_values(n=12)
        discoveries = false_discovery_control(p_values, method="bh") <= 0.1
        assert len(discoveries) == 12
        assert discoveries.dtype == bool

    def test_recomputes_from_raw_arrays_without_result(
        self, sample_scores, sample_weights
    ):
        test_scores, calib_scores = sample_scores(n_test=12, n_calib=40)
        test_weights, calib_weights = sample_weights(n_test=12, n_calib=40)
        p_values = calculate_weighted_p_val(
            scores=test_scores,
            calibration_set=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
        )
        discoveries = false_discovery_control(p_values, method="bh") <= 0.1
        assert len(discoveries) == 12
        assert discoveries.dtype == bool

    def test_recompute_with_seed_is_reproducible(self, sample_scores, sample_weights):
        test_scores, calib_scores = sample_scores(n_test=20, n_calib=80, seed=7)
        test_weights, calib_weights = sample_weights(n_test=20, n_calib=80, seed=7)
        first_p_values = calculate_weighted_p_val(
            scores=test_scores,
            calibration_set=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            tie_break="randomized",
            rng=np.random.default_rng(42),
        )
        second_p_values = calculate_weighted_p_val(
            scores=test_scores,
            calibration_set=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            tie_break="randomized",
            rng=np.random.default_rng(42),
        )
        first = false_discovery_control(first_p_values, method="bh") <= 0.1
        second = false_discovery_control(second_p_values, method="bh") <= 0.1
        np.testing.assert_array_equal(first, second)

    def test_uses_explicit_p_values(self):
        p_values = np.array([0.001, 0.002, 0.003], dtype=float)
        discoveries = false_discovery_control(p_values, method="bh") <= 0.05
        assert np.sum(discoveries) == 3


class TestAlphaLevels:
    def test_alpha_005(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.05)
        assert len(discoveries) == 20

    def test_alpha_010(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.10)
        assert len(discoveries) == 20

    def test_alpha_020(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.20)
        assert len(discoveries) == 20


class TestResultBundleInput:
    def test_with_result_bundle(self, conformal_result):
        result = conformal_result(
            n_test=20, n_calib=100, include_p_values=True, include_weights=True
        )
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 20

    def test_without_p_values_in_result_raises(self, conformal_result):
        result = conformal_result(
            n_test=20, n_calib=100, include_p_values=False, include_weights=True
        )
        with pytest.raises(ValueError):
            weighted_false_discovery_control(result=result, alpha=0.1)

    def test_with_metadata(self, conformal_result):
        result = conformal_result(
            n_test=20,
            n_calib=100,
            include_p_values=True,
            include_weights=True,
            include_metadata=True,
        )
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 20


class TestDirectInput:
    def test_with_p_values_only(self, sample_p_values, sample_scores, sample_weights):
        p_values = sample_p_values(n=15)
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)

        discoveries = weighted_false_discovery_control_from_arrays(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 15

    def test_without_p_values(self, sample_scores, sample_weights):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)
        p_values = calculate_weighted_p_val(
            scores=test_scores,
            calibration_set=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
        )

        discoveries = weighted_false_discovery_control_from_arrays(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 15


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.15, 0.20])
@pytest.mark.parametrize("n_calib,n_test", [(40, 12), (80, 20), (100, 30)])
def test_weighted_fdr_deterministic_matches_reference_formulation(
    alpha: float,
    n_calib: int,
    n_test: int,
) -> None:
    rng = np.random.default_rng(17)
    calib_scores = rng.normal(size=n_calib)
    test_scores = rng.normal(size=n_test)
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
    )
    np.testing.assert_array_equal(actual, expected)
