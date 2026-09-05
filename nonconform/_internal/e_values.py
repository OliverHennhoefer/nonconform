"""Private statistical core for conformal e-values and e-BH."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from nonconform.structures import ConformalResult

from .provenance import (
    BatchSignature,
    CalibrationMode,
    StrategyFamily,
    parse_result_provenance,
)
from .validation import (
    as_1d_numeric,
    validate_finite,
    validate_optional_seed,
    validate_probability,
)

_RAW_ARRAY_GUIDANCE = (
    " Use conformal_e_values(...) with raw score arrays only after "
    "independently verifying the method assumptions."
)


def normalize_repeated_scores(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return score arrays as validated repetition-by-observation matrices."""
    test_arr = _as_1d_or_2d_numeric("test_scores", test_scores)
    calib_arr = _as_1d_or_2d_numeric("calib_scores", calib_scores)
    validate_finite("test_scores", test_arr)
    validate_finite("calib_scores", calib_arr)

    if test_arr.ndim != calib_arr.ndim:
        raise ValueError("test_scores and calib_scores must have the same dimension.")
    if test_arr.ndim == 1:
        if test_arr.size == 0:
            raise ValueError("test_scores must contain at least one score.")
        if calib_arr.size == 0:
            raise ValueError("calib_scores must contain at least one score.")
        return test_arr.reshape(1, -1), calib_arr.reshape(1, -1)

    if test_arr.shape[0] == 0:
        raise ValueError("test_scores must contain at least one repetition.")
    if test_arr.shape[1] == 0:
        raise ValueError("test_scores must contain at least one score.")
    if calib_arr.shape[1] == 0:
        raise ValueError("calib_scores must contain at least one score.")
    if test_arr.shape[0] != calib_arr.shape[0]:
        raise ValueError(
            "test_scores and calib_scores must have the same number of repetitions."
        )
    return test_arr, calib_arr


def normalize_e_values(e_values: np.ndarray) -> np.ndarray:
    """Return e-values as a non-empty validated one-dimensional array."""
    values = as_1d_numeric("e_values", e_values).astype(float, copy=True)
    if values.size == 0:
        raise ValueError("e_values must contain at least one e-value.")
    validate_finite("e_values", values)
    if np.any(values < 0):
        raise ValueError("e_values must be non-negative.")
    return values


def scores_from_results(
    results: Sequence[ConformalResult],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract compatible score matrices from native result snapshots."""
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError("results must be a sequence of ConformalResult objects.")
    if not results:
        raise ValueError("results must contain at least one ConformalResult.")

    first_test, first_calib, first_signature = _scores_from_result(results[0], 0)
    test_rows = [first_test]
    calib_rows = [first_calib]
    for index, result in enumerate(results[1:], start=1):
        test_row, calib_row, signature = _scores_from_result(result, index)
        _validate_result_compatibility(
            test_row,
            calib_row,
            signature,
            n_test=first_test.size,
            n_calibration=first_calib.size,
            test_batch_signature=first_signature,
        )
        test_rows.append(test_row)
        calib_rows.append(calib_row)
    return np.vstack(test_rows), np.vstack(calib_rows)


def compute_conformal_e_values(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
    tie_seed: int | None,
) -> np.ndarray:
    """Validate inputs and compute uniformly aggregated conformal e-values."""
    alpha_bh_value = validate_probability("alpha_bh", alpha_bh)
    tie_seed_value = validate_optional_seed("tie_seed", tie_seed)
    test_arr, calib_arr = normalize_repeated_scores(test_scores, calib_scores)
    n_repetitions, n_test = test_arr.shape

    generators = _tie_generators(tie_seed_value, n_repetitions)
    e_value_sum = np.zeros(n_test, dtype=float)
    for repetition, generator in enumerate(generators):
        e_value_sum += _single_split_e_values(
            test_arr[repetition],
            calib_arr[repetition],
            alpha_bh=alpha_bh_value,
            rng=generator,
            repetition=repetition,
        )
    return e_value_sum / n_repetitions


def e_bh_selection(e_values: np.ndarray, *, alpha: float) -> tuple[np.ndarray, float]:
    """Return the e-BH selection mask and its observed e-value cutoff."""
    n_hypotheses = e_values.size
    order = np.argsort(-e_values, kind="stable")
    sorted_e_values = e_values[order]
    thresholds = n_hypotheses / (alpha * np.arange(1, n_hypotheses + 1, dtype=float))
    passing = np.flatnonzero(sorted_e_values >= thresholds)
    if passing.size == 0:
        return np.zeros(n_hypotheses, dtype=bool), float("inf")

    n_selected = int(passing[-1] + 1)
    e_threshold = float(sorted_e_values[n_selected - 1])
    selected = e_values >= e_threshold
    return selected, e_threshold


def _scores_from_result(
    result: ConformalResult,
    index: int,
) -> tuple[np.ndarray, np.ndarray, BatchSignature]:
    """Extract one score pair and its trusted batch signature."""
    label = f"results[{index}]"
    _validate_result_fields(result, label)
    signature = _native_test_batch_signature(result, label)
    test_row = _validated_score_row(f"{label}.test_scores", result.test_scores)
    calib_row = _validated_score_row(f"{label}.calib_scores", result.calib_scores)
    _validate_signature_score_size(signature, test_row.size, label)
    return test_row, calib_row, signature


def _validate_result_fields(result: ConformalResult, label: str) -> None:
    """Validate the result container and required score fields."""
    if not isinstance(result, ConformalResult):
        raise TypeError(f"{label} must be a ConformalResult.")
    if result.test_scores is None or result.calib_scores is None:
        raise ValueError(
            f"{label} is missing test_scores or calib_scores. Run "
            "score_samples(...) before collecting detector.last_result."
        )
    if result.test_weights is not None or result.calib_weights is not None:
        raise ValueError(
            f"{label} contains weighted scores, which are unsupported."
            f"{_RAW_ARRAY_GUIDANCE}"
        )


def _native_test_batch_signature(
    result: ConformalResult,
    label: str,
) -> BatchSignature:
    """Validate native scope facts and return the stamped batch signature."""
    provenance = parse_result_provenance(result, allow_legacy_metadata=False)
    if provenance is None:
        raise ValueError(
            f"{label} has no native detector provenance.{_RAW_ARRAY_GUIDANCE}"
        )
    if provenance.weighted:
        raise ValueError(
            f"{label} is weighted; only unweighted results are supported."
            f"{_RAW_ARRAY_GUIDANCE}"
        )
    if provenance.strategy_family is not StrategyFamily.SPLIT:
        raise ValueError(
            f"{label} is not an integrated Split result.{_RAW_ARRAY_GUIDANCE}"
        )
    if provenance.calibration_mode is not CalibrationMode.INTEGRATED:
        raise ValueError(
            f"{label} is not an integrated Split result.{_RAW_ARRAY_GUIDANCE}"
        )
    if provenance.test_batch_signature is None:
        raise ValueError(
            f"{label} has no verified test-batch identity.{_RAW_ARRAY_GUIDANCE}"
        )
    return provenance.test_batch_signature


def _validated_score_row(name: str, values: np.ndarray) -> np.ndarray:
    """Return one finite, non-empty score row."""
    row = as_1d_numeric(name, values)
    validate_finite(name, row)
    if row.size == 0:
        raise ValueError(f"{name} must contain at least one score.")
    return row


def _validate_signature_score_size(
    signature: BatchSignature,
    n_scores: int,
    label: str,
) -> None:
    """Check that a result's score count matches its stamped input batch."""
    if not signature.shape:
        raise ValueError(
            f"{label} has an invalid test-batch identity.{_RAW_ARRAY_GUIDANCE}"
        )
    if signature.shape[0] != n_scores:
        raise ValueError(
            f"{label}.test_scores does not match its verified test batch."
            f"{_RAW_ARRAY_GUIDANCE}"
        )


def _validate_result_compatibility(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    signature: BatchSignature,
    *,
    n_test: int,
    n_calibration: int,
    test_batch_signature: BatchSignature,
) -> None:
    """Validate dimensions and exact test-batch identity across repetitions."""
    if test_scores.size != n_test:
        raise ValueError(
            "Every result must contain the same number of test scores in the "
            "same observation order."
        )
    if calib_scores.size != n_calibration:
        raise ValueError(
            "Every result must contain the same number of calibration scores."
        )
    if signature != test_batch_signature:
        raise ValueError(
            "Every result must describe the identical test batch in the same row order."
        )


def _as_1d_or_2d_numeric(name: str, values: np.ndarray) -> np.ndarray:
    """Normalize array-like input into a copied 1D or 2D float array."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if arr.ndim not in {1, 2}:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape!r}.")
    return arr.astype(float, copy=True)


def _tie_generators(
    tie_seed: int | None,
    n_repetitions: int,
) -> tuple[np.random.Generator | None, ...]:
    """Return one reproducible tie generator per repetition."""
    if tie_seed is None:
        return (None,) * n_repetitions
    seed_sequences = np.random.SeedSequence(tie_seed).spawn(n_repetitions)
    return tuple(np.random.default_rng(item) for item in seed_sequences)


def _single_split_e_values(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    *,
    alpha_bh: float,
    rng: np.random.Generator | None,
    repetition: int,
) -> np.ndarray:
    """Compute one split's e-values using one sort and cumulative counts."""
    n_test = test_scores.size
    n_calib = calib_scores.size
    combined_scores = np.concatenate((test_scores, calib_scores))
    is_test = np.concatenate(
        (np.ones(n_test, dtype=bool), np.zeros(n_calib, dtype=bool))
    )

    score_order = np.argsort(-combined_scores, kind="stable")
    ordered_scores = combined_scores[score_order]
    tied_locations = np.flatnonzero(ordered_scores[1:] == ordered_scores[:-1])
    has_ties = tied_locations.size > 0

    if has_ties and rng is None:
        tied_score = float(ordered_scores[tied_locations[0]])
        raise ValueError(
            "test_scores and calib_scores contain tied scores in repetition "
            f"{repetition + 1} (for example, {tied_score!r}). The conformal "
            "e-value validity argument requires a strict ordering; pass "
            "tie_seed=<non-negative integer> to randomize ties reproducibly."
        )

    if has_ties:
        # A random permutation provides unique secondary ranks, so finite-precision
        # RNG collisions cannot leave any score pair tied.
        tie_keys = rng.permutation(combined_scores.size)
        order = np.lexsort((-tie_keys, -combined_scores))
    else:
        order = score_order

    ordered_is_test = is_test[order]
    test_counts = np.cumsum(ordered_is_test)
    calib_counts = np.cumsum(~ordered_is_test)
    # Preserve ratio-then-scale evaluation at exact alpha_bh cutoffs.
    estimated_fdp = (n_test / n_calib) * np.divide(
        calib_counts,
        test_counts,
        out=np.full(combined_scores.size, np.inf, dtype=float),
        where=test_counts > 0,
    )
    valid = np.flatnonzero((test_counts > 0) & (estimated_fdp <= alpha_bh))
    if valid.size == 0:
        return np.zeros(n_test, dtype=float)

    cutoff = int(valid[-1])
    selected_test_indices = order[: cutoff + 1]
    selected_test_indices = selected_test_indices[selected_test_indices < n_test]
    denominator = 1.0 + float(calib_counts[cutoff])
    evidence = (1.0 + n_calib) / denominator

    e_values = np.zeros(n_test, dtype=float)
    e_values[selected_test_indices] = evidence
    return e_values
