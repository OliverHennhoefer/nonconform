import inspect
from itertools import permutations

import numpy as np
import pytest

from nonconform._internal.provenance import (
    BatchSignature,
    CalibrationMode,
    EstimationFamily,
    ResultProvenance,
    StrategyFamily,
    batch_signature,
)
from nonconform.fdr import (
    EValueSelectionResult,
    conformal_e_values,
    e_value_false_discovery_control,
    select_conformal_e_values,
)
from nonconform.structures import ConformalResult


def _metadata(**overrides):
    scope = {
        "strategy": "Split",
        "estimation": "Empirical",
        "weighted": False,
    }
    scope.update(overrides)
    return {"nonconform": scope}


def _result(
    test_scores=(3.0, 0.5),
    calib_scores=(0.0, 1.0, 2.0),
    *,
    metadata=None,
    test_weights=None,
    calib_weights=None,
    strategy_family=StrategyFamily.SPLIT,
    estimation_family=EstimationFamily.EMPIRICAL,
    calibration_mode=CalibrationMode.INTEGRATED,
    weighted=False,
    batch_digest="test-batch",
    native=True,
):
    test_scores_array = np.asarray(test_scores, dtype=float)
    result = ConformalResult(
        test_scores=test_scores_array,
        calib_scores=np.asarray(calib_scores, dtype=float),
        test_weights=test_weights,
        calib_weights=calib_weights,
        metadata=_metadata() if metadata is None else metadata,
    )
    if native:
        result._provenance = ResultProvenance(
            strategy_family=strategy_family,
            estimation_family=estimation_family,
            weighted=weighted,
            calibration_mode=calibration_mode,
            test_batch_signature=(
                None
                if batch_digest is None
                else BatchSignature(
                    shape=(test_scores_array.size, 1),
                    dtype="float64",
                    digest=batch_digest,
                )
            ),
        )
    return result


def _slow_conformal_e_values(test_scores, calib_scores, *, alpha_bh):
    rows = []
    for test_row, calib_row in zip(test_scores, calib_scores, strict=True):
        threshold = float("inf")
        for candidate in np.sort(np.concatenate((test_row, calib_row))):
            n_test_above = np.count_nonzero(test_row >= candidate)
            if n_test_above == 0:
                continue
            n_calib_above = np.count_nonzero(calib_row >= candidate)
            estimated_fdp = (test_row.size / calib_row.size) * (
                n_calib_above / n_test_above
            )
            if estimated_fdp <= alpha_bh:
                threshold = float(candidate)
                break
        n_calib_above = np.count_nonzero(calib_row >= threshold)
        evidence = (calib_row.size + 1) / (n_calib_above + 1)
        rows.append(evidence * (test_row >= threshold))
    return np.mean(rows, axis=0)


def test_e_bh_selects_largest_valid_prefix():
    e_values = np.array([30.0, 12.0, 4.0, 1.0])

    selected = e_value_false_discovery_control(e_values, alpha=0.2)

    np.testing.assert_array_equal(selected, np.array([True, True, False, False]))


def test_e_bh_returns_no_discoveries_when_no_prefix_passes():
    selected = e_value_false_discovery_control(np.array([5.0, 4.0]), alpha=0.1)

    np.testing.assert_array_equal(selected, np.array([False, False]))


def test_e_bh_is_monotone_in_alpha():
    e_values = np.array([30.0, 12.0, 4.0, 1.0])

    low_alpha = e_value_false_discovery_control(e_values, alpha=0.05)
    high_alpha = e_value_false_discovery_control(e_values, alpha=0.2)

    assert np.count_nonzero(low_alpha) <= np.count_nonzero(high_alpha)


def test_e_bh_includes_every_value_tied_at_cutoff():
    e_values = np.array([100.0, 10.0, 10.0, 10.0, 1.0])

    selected = e_value_false_discovery_control(e_values, alpha=0.2)

    np.testing.assert_array_equal(
        selected,
        np.array([True, True, True, True, False]),
    )


def test_e_bh_is_permutation_equivariant():
    e_values = np.array([2.0, 30.0, 12.0, 1.0])
    permutation = np.array([2, 0, 3, 1])

    expected = e_value_false_discovery_control(e_values, alpha=0.2)
    permuted = e_value_false_discovery_control(
        e_values[permutation],
        alpha=0.2,
    )

    np.testing.assert_array_equal(permuted[np.argsort(permutation)], expected)


@pytest.mark.parametrize(
    ("e_values", "match"),
    [
        (np.array([]), "at least one"),
        (np.array([[1.0, 2.0]]), "1D"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, -0.1]), "non-negative"),
    ],
)
def test_e_bh_validates_e_values(e_values, match):
    with pytest.raises(ValueError, match=match):
        e_value_false_discovery_control(e_values, alpha=0.2)


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan])
def test_e_bh_validates_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        e_value_false_discovery_control(np.array([1.0]), alpha=alpha)


def test_conformal_e_values_match_hand_computed_single_split():
    test_scores = np.array([3.0, 0.5])
    calib_scores = np.array([0.0, 1.0, 2.0])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)

    np.testing.assert_allclose(e_values, np.array([4.0, 0.0]))


@pytest.mark.parametrize(("n_calib", "alpha_bh"), [(15, 0.2), (30, 0.1), (60, 0.05)])
@pytest.mark.parametrize("below_boundary", [False, True])
def test_conformal_e_values_preserve_inner_cutoff_boundary(
    n_calib, alpha_bh, below_boundary
):
    # Three calibration scores give FDP exactly alpha_bh when all tests pass.
    if below_boundary:
        alpha_bh = np.nextafter(alpha_bh, 0.0)
    e_values = conformal_e_values(
        n_calib + np.arange(1.0, 4.0),
        np.arange(n_calib, dtype=float),
        alpha_bh=alpha_bh,
    )

    n_calib_above = 2 if below_boundary else 3
    expected = np.full(3, (n_calib + 1) / (n_calib_above + 1))
    np.testing.assert_array_equal(e_values, expected)


def test_inner_cutoff_boundary_does_not_add_e_bh_discoveries():
    e_values = conformal_e_values(
        np.array([20.0, 21.0, 22.0]),
        np.arange(15, dtype=float),
        alpha_bh=0.2,
    )

    selected = e_value_false_discovery_control(e_values, alpha=0.2)

    np.testing.assert_array_equal(selected, np.zeros(3, dtype=bool))


def test_batch_identity_preserves_large_object_integers():
    first = np.array([[2**53]], dtype=object)
    changed = np.array([[2**53 + 1]], dtype=object)

    assert batch_signature(first) != batch_signature(changed)


def test_conformal_e_values_average_repeated_splits_uniformly():
    test_scores = np.array([[3.0, 0.5], [3.0, 2.0]])
    calib_scores = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)

    np.testing.assert_allclose(e_values, np.array([4.0, 2.0]))


def test_optimized_conformal_e_values_match_slow_reference():
    rng = np.random.default_rng(2026)
    test_scores = rng.normal(size=(8, 31))
    calib_scores = rng.normal(size=(8, 47))

    actual = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)
    expected = _slow_conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=0.2,
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("alpha_bh", [0.1, 0.5, 0.9])
def test_all_null_enumeration_satisfies_aggregate_e_value_condition(alpha_bh):
    total_evidence = []
    for assignment in permutations(np.arange(4, dtype=float)):
        total_evidence.append(
            conformal_e_values(
                np.asarray(assignment[:2]),
                np.asarray(assignment[2:]),
                alpha_bh=alpha_bh,
            ).sum()
        )

    assert np.mean(total_evidence) <= 2.0 + 1e-12


def test_conformal_e_values_without_valid_threshold_are_zero():
    test_scores = np.array([1.0, 2.0])
    calib_scores = np.array([3.0, 4.0])

    e_values = conformal_e_values(test_scores, calib_scores, alpha_bh=0.1)

    np.testing.assert_allclose(e_values, np.zeros(2))


@pytest.mark.parametrize(
    ("test_scores", "calib_scores", "match"),
    [
        (np.array([]), np.array([1.0]), "test_scores"),
        (np.array([1.0]), np.array([]), "calib_scores"),
        (np.array([[1.0]]), np.array([2.0]), "same dimension"),
        (np.ones((2, 3)), np.ones((3, 4)), "same number of repetitions"),
        (np.array([1.0, np.inf]), np.array([2.0]), "finite"),
    ],
)
def test_conformal_e_values_validate_shapes_and_values(
    test_scores,
    calib_scores,
    match,
):
    with pytest.raises(ValueError, match=match):
        conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)


@pytest.mark.parametrize("alpha_bh", [0.0, 1.0, np.nan])
def test_conformal_e_values_validate_alpha_bh(alpha_bh):
    with pytest.raises(ValueError, match="alpha_bh"):
        conformal_e_values(np.array([1.0]), np.array([0.0]), alpha_bh=alpha_bh)


@pytest.mark.parametrize(
    ("test_scores", "calib_scores"),
    [
        ([3.0, 1.0], [0.0, 0.0]),
        ([3.0, 3.0], [0.0, 1.0]),
        ([3.0, 1.0], [0.0, 1.0]),
    ],
)
def test_conformal_e_values_reject_all_score_tie_locations_by_default(
    test_scores,
    calib_scores,
):
    with pytest.raises(ValueError, match="tied scores in repetition 1") as error:
        conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)

    assert "tie_seed=" in str(error.value)


def test_tie_error_identifies_repetition():
    test_scores = np.array([[4.0, 3.0], [2.0, 2.0]])
    calib_scores = np.array([[0.0, 1.0], [-1.0, 1.0]])

    with pytest.raises(ValueError, match="repetition 2"):
        conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)


def test_randomized_tie_breaking_is_seeded_and_sensitive_to_seed():
    test_scores = np.ones(4)
    calib_scores = np.ones(4)

    first = conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=0.2,
        tie_seed=0,
    )
    replay = conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=0.2,
        tie_seed=0,
    )
    different_seed = conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=0.2,
        tie_seed=1,
    )

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, different_seed)


def test_randomized_policy_does_not_change_strict_score_order_result():
    test_scores = np.array([3.0, 0.5])
    calib_scores = np.array([0.0, 1.0, 2.0])

    strict = conformal_e_values(test_scores, calib_scores, alpha_bh=0.2)
    randomized = conformal_e_values(
        test_scores,
        calib_scores,
        alpha_bh=0.2,
        tie_seed=7,
    )

    np.testing.assert_array_equal(randomized, strict)


@pytest.mark.parametrize(
    ("tie_seed", "error", "match"),
    [
        (-1, ValueError, "non-negative"),
        (True, TypeError, "non-negative"),
        (1.5, TypeError, "non-negative"),
        ("1", TypeError, "non-negative"),
    ],
)
def test_conformal_e_values_validates_tie_seed(tie_seed, error, match):
    with pytest.raises(error, match=match):
        conformal_e_values(
            np.array([2.0]),
            np.array([1.0]),
            alpha_bh=0.2,
            tie_seed=tie_seed,
        )


def test_select_conformal_e_values_defaults_and_matches_raw_primitives():
    results = [
        _result((3.0, 0.5), (0.0, 1.0, 2.0)),
        _result((3.0, 2.0), (-1.0, 0.0, 1.0)),
    ]

    result = select_conformal_e_values(results, alpha=0.2)
    raw_e_values = conformal_e_values(
        np.vstack([item.test_scores for item in results]),
        np.vstack([item.calib_scores for item in results]),
        alpha_bh=0.02,
    )

    assert isinstance(result, EValueSelectionResult)
    assert result.alpha == pytest.approx(0.2)
    assert result.alpha_bh == pytest.approx(0.02)
    assert result.e_threshold == float("inf")
    assert result.n_repetitions == 2
    assert result.n_calibration == 3
    assert result.tie_seed is None
    np.testing.assert_array_equal(result.e_values, raw_e_values)
    np.testing.assert_array_equal(
        result.selected,
        e_value_false_discovery_control(raw_e_values, alpha=0.2),
    )


def test_selection_result_arrays_are_copied_and_read_only():
    source = _result()
    result = select_conformal_e_values([source], alpha=0.2)
    source.test_scores[0] = -100.0

    np.testing.assert_array_equal(result.e_values, np.array([4.0, 0.0]))
    assert not result.e_values.flags.writeable
    assert not result.selected.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.e_values[0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        result.selected[0] = False


def test_selection_result_records_finite_e_bh_cutoff():
    result = select_conformal_e_values(
        [_result((10.0,), tuple(np.arange(9, dtype=float)))],
        alpha=0.2,
    )

    np.testing.assert_array_equal(result.selected, np.array([True]))
    assert result.e_threshold == pytest.approx(10.0)


def test_selection_result_records_tie_seed():
    result = select_conformal_e_values(
        [_result((1.0, 1.0), (0.0, 1.0, 2.0))],
        alpha=0.2,
        tie_seed=42,
    )

    assert result.tie_seed == 42
    assert not hasattr(result, "score_ties")
    assert not hasattr(result, "seed")


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("alpha", np.nan),
        ("alpha_bh", 0.0),
        ("alpha_bh", 1.0),
        ("alpha_bh", np.nan),
    ],
)
def test_select_conformal_e_values_validates_probabilities(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        select_conformal_e_values([_result()], **{parameter: value})


@pytest.mark.parametrize(
    ("results", "error", "match"),
    [
        ([], ValueError, "at least one"),
        (None, TypeError, "sequence"),
        ([object()], TypeError, "ConformalResult"),
        ([ConformalResult()], ValueError, "missing"),
        ([_result(native=False)], ValueError, "no native detector provenance"),
        ([_result(batch_digest=None)], ValueError, "test-batch identity"),
        ([_result(weighted=True)], ValueError, "weighted"),
        (
            [_result(strategy_family=StrategyFamily.OTHER)],
            ValueError,
            "integrated Split",
        ),
        (
            [_result(calibration_mode=CalibrationMode.DETACHED)],
            ValueError,
            "integrated",
        ),
        ([_result(test_weights=np.ones(2))], ValueError, "weighted scores"),
        ([_result(test_scores=(np.nan, 1.0))], ValueError, "finite"),
    ],
)
def test_select_conformal_e_values_rejects_unverifiable_results(
    results,
    error,
    match,
):
    with pytest.raises(error, match=match):
        select_conformal_e_values(results)


def test_select_conformal_e_values_ignores_mutable_metadata_claims():
    valid = _result(metadata={"nonconform": {"strategy": "NotSplit"}})
    selection = select_conformal_e_values([valid], alpha=0.2)
    assert selection.n_repetitions == 1

    invalid = _result(
        metadata=_metadata(),
        strategy_family=StrategyFamily.OTHER,
    )
    with pytest.raises(ValueError, match="integrated Split"):
        select_conformal_e_values([invalid])


def test_select_conformal_e_values_is_estimator_independent():
    empirical = select_conformal_e_values([_result()], alpha=0.2)
    custom = select_conformal_e_values(
        [_result(estimation_family=EstimationFamily.OTHER)],
        alpha=0.2,
    )

    np.testing.assert_array_equal(custom.e_values, empirical.e_values)


def test_conformal_result_copy_preserves_native_provenance():
    source = _result()

    copied = source.copy()
    selection = select_conformal_e_values([copied], alpha=0.2)

    assert copied._provenance is source._provenance
    np.testing.assert_array_equal(selection.e_values, np.array([4.0, 0.0]))


def test_select_conformal_e_values_requires_identical_test_batch():
    first = _result(batch_digest="batch-a")
    changed_or_reordered = _result(batch_digest="batch-b")

    with pytest.raises(ValueError, match=r"identical test batch.*same row order"):
        select_conformal_e_values([first, changed_or_reordered])


@pytest.mark.parametrize(
    ("first", "second", "match"),
    [
        (_result(), _result(test_scores=(10.0,)), "same number of test scores"),
        (_result(), _result(calib_scores=(10.0,)), "same number of calibration scores"),
    ],
)
def test_select_conformal_e_values_requires_consistent_sizes(first, second, match):
    with pytest.raises(ValueError, match=match):
        select_conformal_e_values([first, second])


def test_result_score_mutation_cannot_outgrow_stamped_batch():
    result = _result()
    result.test_scores = np.array([3.0])

    with pytest.raises(ValueError, match="does not match its verified test batch"):
        select_conformal_e_values([result])


def test_old_tie_parameters_and_result_fields_are_absent():
    raw_parameters = inspect.signature(conformal_e_values).parameters
    result_parameters = inspect.signature(select_conformal_e_values).parameters

    assert list(raw_parameters) == [
        "test_scores",
        "calib_scores",
        "alpha_bh",
        "tie_seed",
    ]
    assert list(result_parameters) == ["results", "alpha", "alpha_bh", "tie_seed"]
    assert raw_parameters["tie_seed"].default is None
    assert result_parameters["tie_seed"].default is None
    assert "score_ties" not in raw_parameters
    assert "score_ties" not in result_parameters
    assert "seed" not in raw_parameters
    assert "seed" not in result_parameters
