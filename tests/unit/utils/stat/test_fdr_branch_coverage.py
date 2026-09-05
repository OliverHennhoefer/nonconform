import numpy as np
import pytest

from nonconform._internal import Pruning
from nonconform._internal.validation import (
    as_1d_numeric,
    validate_finite,
    validate_non_negative_finite,
    validate_p_values,
)
from nonconform._internal.wcs import (
    _compute_r_star,
    _rejection_set_size_for_instance,
    _validate_pruning,
    _WCSInputs,
    extract_kde_support,
    run,
)
from nonconform.structures import ConformalResult


def _valid_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    p_values = np.array([0.2, 0.7], dtype=float)
    test_scores = np.array([1.0, 2.0], dtype=float)
    calib_scores = np.array([0.5, 1.5, 2.5], dtype=float)
    test_weights = np.ones(2, dtype=float)
    calib_weights = np.ones(3, dtype=float)
    return p_values, test_scores, calib_scores, test_weights, calib_weights


def _test_inputs() -> _WCSInputs:
    """Return a small validated-shape WCS bundle for helper tests."""
    return _WCSInputs(*_valid_arrays())


def test_as_1d_numeric_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="numeric array"):
        as_1d_numeric("values", np.array(["x"], dtype=object))


def test_validation_helpers_accept_empty_arrays() -> None:
    empty = np.array([], dtype=float)
    validate_non_negative_finite("weights", empty)
    validate_finite("scores", empty)
    validate_p_values(empty)


def test_validate_non_negative_finite_rejects_non_finite_and_negative() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_non_negative_finite("weights", np.array([np.inf]))
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_non_negative_finite("weights", np.array([-1.0]))


def test_validate_finite_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_finite("scores", np.array([np.nan]))


def test_validate_pruning_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="instance of Pruning"):
        _validate_pruning("deterministic")  # type: ignore[arg-type]


def test_compute_r_star_empty_metrics_returns_zero() -> None:
    assert _compute_r_star(np.array([], dtype=float)) == 0


def test_compute_rejection_set_size_requires_score_rank_cache() -> None:
    with pytest.raises(ValueError, match="missing score-rank cache"):
        _rejection_set_size_for_instance(
            index=0,
            inputs=_test_inputs(),
            sum_calib_weight=2.0,
            bh_thresholds=np.array([0.05, 0.10]),
            calib_mass_strictly_above=np.array([1.0, 0.5]),
            scratch=np.empty(2, dtype=float),
            include_self_weight=True,
            sorted_test_idx=None,
            lt_cutoffs=None,
        )


def test_compute_rejection_set_size_rejects_non_positive_effective_mass() -> None:
    with pytest.raises(ValueError, match="positive finite effective calibration mass"):
        _rejection_set_size_for_instance(
            index=0,
            inputs=_test_inputs(),
            sum_calib_weight=0.0,
            bh_thresholds=np.array([0.05, 0.10]),
            calib_mass_strictly_above=np.array([0.0, 0.0]),
            scratch=np.empty(2, dtype=float),
            include_self_weight=False,
            sorted_test_idx=None,
            lt_cutoffs=None,
        )


def test_extract_kde_support_requires_dict_metadata() -> None:
    result = ConformalResult(metadata={"kde": 1})
    with pytest.raises(ValueError, match="must be a dictionary"):
        extract_kde_support(result)


def test_extract_kde_support_rejects_non_numeric_total_weight() -> None:
    result = ConformalResult(
        metadata={
            "kde": {
                "eval_grid": np.array([0.0, 1.0]),
                "cdf_values": np.array([0.0, 1.0]),
                "total_weight": "not-a-number",
            }
        }
    )
    with pytest.raises(ValueError, match="finite positive float"):
        extract_kde_support(result)


def test_extract_kde_support_rejects_too_short_eval_grid() -> None:
    result = ConformalResult(
        metadata={
            "kde": {
                "eval_grid": np.array([0.0]),
                "cdf_values": np.array([0.0]),
                "total_weight": 1.0,
            }
        }
    )
    with pytest.raises(ValueError, match="at least 2 points"):
        extract_kde_support(result)


def test_extract_kde_support_rejects_length_mismatch() -> None:
    result = ConformalResult(
        metadata={
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 2.0]),
                "cdf_values": np.array([0.0, 1.0]),
                "total_weight": 1.0,
            }
        }
    )
    with pytest.raises(ValueError, match="must have equal length"):
        extract_kde_support(result)


def test_extract_kde_support_rejects_cdf_out_of_bounds() -> None:
    result = ConformalResult(
        metadata={
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 2.0]),
                "cdf_values": np.array([0.0, 1.1, 1.2]),
                "total_weight": 1.0,
            }
        }
    )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        extract_kde_support(result)


def test_run_wcs_rejects_mismatched_test_lengths() -> None:
    p_values, test_scores, calib_scores, _, calib_weights = _valid_arrays()
    with pytest.raises(ValueError, match="must have the same length"):
        run(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=np.ones(1, dtype=float),
            calib_weights=calib_weights,
            alpha=0.1,
            pruning=Pruning.DETERMINISTIC,
            seed=0,
        )


def test_run_wcs_rejects_mismatched_calibration_lengths() -> None:
    p_values, test_scores, calib_scores, test_weights, _ = _valid_arrays()
    with pytest.raises(ValueError, match="must have the same length"):
        run(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=np.ones(2, dtype=float),
            alpha=0.1,
            pruning=Pruning.DETERMINISTIC,
            seed=0,
        )


def test_run_wcs_rejects_non_positive_total_calibration_weight() -> None:
    p_values, test_scores, calib_scores, test_weights, _ = _valid_arrays()
    with pytest.raises(ValueError, match="positive finite total calibration weight"):
        run(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=np.zeros(3, dtype=float),
            alpha=0.1,
            pruning=Pruning.DETERMINISTIC,
            seed=0,
            include_self_weight=False,
        )
