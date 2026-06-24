import numpy as np
import pytest

from nonconform.fdr import (
    FDPBoundResult,
    conformal_fdp_upper_bound,
    conformal_fdp_upper_bound_from_result,
)
from nonconform.structures import ConformalResult

SUPPORTED_METHODS = ["mc_thc", "mc_hc", "mc_ks", "ks", "mc_bj"]


def _bounds(
    p_values=np.array([0.05, 0.2, 0.2, 0.8]),
    **kwargs,
) -> FDPBoundResult:
    params = {
        "n_calibration": 20,
        "confidence": 0.8,
        "n_resamples": 25,
        "seed": 7,
    }
    params.update(kwargs)
    return conformal_fdp_upper_bound(p_values, **params)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"p_values": np.array([[0.1, 0.2]])}, ValueError, "1D"),
        ({"p_values": np.array([0.1, np.nan])}, ValueError, "finite"),
        ({"p_values": np.array([0.1, 1.2])}, ValueError, "within"),
        ({"p_values": np.array([])}, ValueError, "at least one"),
        ({"n_calibration": 0}, ValueError, "positive"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"n_resamples": 0}, ValueError, "positive"),
        ({"lower": 0.2, "upper": 0.1}, ValueError, "lower"),
        ({"beta": 0.0}, ValueError, "beta"),
        ({"precision": 0.0}, ValueError, "precision"),
        ({"method": "marginal_mc"}, ValueError, "method"),
    ],
)
def test_conformal_fdp_upper_bound_validates_inputs(kwargs, error_type, match):
    params = {
        "p_values": np.array([0.1, 0.2, 0.6]),
        "n_calibration": 10,
        "confidence": 0.8,
        "n_resamples": 5,
        "seed": 1,
    }
    params.update(kwargs)
    with pytest.raises(error_type, match=match):
        conformal_fdp_upper_bound(**params)


def test_conformal_fdp_upper_bound_rejects_non_string_method():
    with pytest.raises(TypeError, match="method"):
        _bounds(method=1)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("MC-THC", "mc_thc"),
        ("MC HC", "mc_hc"),
        ("mc-ks", "mc_ks"),
        ("KS", "ks"),
        ("MC-BJ", "mc_bj"),
    ],
)
def test_method_aliases_normalize(alias, expected):
    result = _bounds(method=alias)

    assert result.method == expected


def test_default_thresholds_are_sorted_unique_p_values():
    result = _bounds()

    np.testing.assert_allclose(result.thresholds, np.array([0.05, 0.2, 0.8]))
    np.testing.assert_array_equal(result.rejection_counts, np.array([1, 3, 4]))


def test_threshold_override_is_preserved_and_zero_discovery_bound_is_zero():
    thresholds = np.array([0.0, 0.5, 0.1])
    result = _bounds(thresholds=thresholds)

    np.testing.assert_allclose(result.thresholds, thresholds)
    np.testing.assert_array_equal(result.rejection_counts, np.array([0, 3, 1]))
    assert result.fdp_upper_bounds[0] == 0.0
    assert np.all((0.0 <= result.fdp_upper_bounds) & (result.fdp_upper_bounds <= 1.0))


def test_result_copies_inputs_and_does_not_mutate_them():
    p_values = np.array([0.4, 0.1, 0.8])
    thresholds = np.array([0.2, 0.7])

    result = _bounds(p_values=p_values, thresholds=thresholds)
    p_values[:] = 0.99
    thresholds[:] = 0.99

    np.testing.assert_allclose(result.p_values, np.array([0.4, 0.1, 0.8]))
    np.testing.assert_allclose(result.thresholds, np.array([0.2, 0.7]))


def test_bound_at_supports_scalar_and_vector_thresholds():
    result = _bounds(thresholds=np.array([0.1, 0.5]))

    scalar_bound = result.bound_at(0.5)
    vector_bounds = result.bound_at(np.array([0.1, 0.5]))

    assert isinstance(scalar_bound, float)
    np.testing.assert_allclose(vector_bounds, result.fdp_upper_bounds)


def test_precision_lower_bounds_are_fdp_complements():
    result = _bounds(thresholds=np.array([0.1, 0.5]))

    np.testing.assert_allclose(
        result.precision_lower_bounds,
        1.0 - result.fdp_upper_bounds,
    )
    np.testing.assert_allclose(
        result.precision_at(np.array([0.1, 0.5])),
        result.precision_lower_bounds,
    )
    assert isinstance(result.precision_at(0.5), float)


def test_to_frame_returns_threshold_level_certificate_table():
    result = _bounds(thresholds=np.array([0.1, 0.5]))

    table = result.to_frame()

    assert list(table.columns) == [
        "threshold",
        "discoveries",
        "fdp_upper_bound",
        "precision_lower_bound",
    ]
    np.testing.assert_allclose(table["threshold"].to_numpy(), result.thresholds)
    np.testing.assert_array_equal(
        table["discoveries"].to_numpy(),
        result.rejection_counts,
    )
    np.testing.assert_allclose(
        table["fdp_upper_bound"].to_numpy(),
        result.fdp_upper_bounds,
    )
    np.testing.assert_allclose(
        table["precision_lower_bound"].to_numpy(),
        result.precision_lower_bounds,
    )


def test_to_frame_accepts_custom_threshold_grid():
    result = _bounds(thresholds=np.array([0.1, 0.5]))
    grid = np.array([0.0, 0.2, 0.6])

    table = result.to_frame(thresholds=grid)

    np.testing.assert_allclose(table["threshold"].to_numpy(), grid)
    np.testing.assert_array_equal(table["discoveries"].to_numpy(), np.array([0, 3, 3]))
    np.testing.assert_allclose(
        table["fdp_upper_bound"].to_numpy(),
        result.bound_at(grid),
    )
    np.testing.assert_allclose(
        table["precision_lower_bound"].to_numpy(),
        result.precision_at(grid),
    )


def test_select_returns_original_order_threshold_mask():
    p_values = np.array([0.4, 0.1, 0.8, 0.2])
    result = _bounds(p_values=p_values)

    np.testing.assert_array_equal(
        result.select(0.2),
        np.array([False, True, False, True]),
    )


def test_select_rejects_vector_thresholds():
    result = _bounds()

    with pytest.raises(ValueError, match="scalar"):
        result.select(np.array([0.1, 0.2]))


def test_same_seed_gives_identical_bounds():
    first = _bounds(seed=11, method="mc_bj")
    second = _bounds(seed=11, method="mc_bj")

    np.testing.assert_allclose(first.fdp_upper_bounds, second.fdp_upper_bounds)


def test_different_seed_can_change_bounds():
    first = _bounds(
        p_values=np.array([0.01, 0.03, 0.08, 0.25, 0.6]),
        thresholds=np.array([0.03, 0.08, 0.25]),
        confidence=0.5,
        n_resamples=50,
        seed=1,
    )
    alternatives = [
        _bounds(
            p_values=np.array([0.01, 0.03, 0.08, 0.25, 0.6]),
            thresholds=np.array([0.03, 0.08, 0.25]),
            confidence=0.5,
            n_resamples=50,
            seed=seed,
        )
        for seed in range(2, 8)
    ]

    assert any(
        not np.allclose(first.fdp_upper_bounds, other.fdp_upper_bounds)
        for other in alternatives
    )


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_supported_methods_return_valid_bounds(method):
    result = _bounds(
        p_values=np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6]),
        thresholds=np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6]),
        n_calibration=50,
        confidence=0.5,
        n_resamples=20,
        seed=3,
        method=method,
    )

    assert result.method == method
    assert np.all((0.0 <= result.fdp_upper_bounds) & (result.fdp_upper_bounds <= 1.0))
    np.testing.assert_allclose(
        result.precision_lower_bounds,
        1.0 - result.fdp_upper_bounds,
    )


def test_supported_methods_match_fixed_reference_values():
    p_values = np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6])
    thresholds = np.array([0.001, 0.01, 0.03, 0.08, 0.25, 0.6])
    expected = {
        "mc_thc": np.array(
            [
                0.406804935815,
                0.203402467907,
                0.258195099608,
                0.356399518904,
                0.485119615123,
                0.570933012602,
            ]
        ),
        "mc_hc": np.array(
            [
                0.116166719315,
                0.203402467907,
                0.258195099608,
                0.356399518904,
                0.485119615123,
                0.570933012602,
            ]
        ),
        "mc_ks": np.array(
            [
                1.0,
                0.748581325403,
                0.539054216936,
                0.479290662702,
                0.583432530161,
                0.652860441801,
            ]
        ),
        "ks": np.array(
            [
                1.0,
                1.0,
                0.821888027249,
                0.691416020437,
                0.753132816349,
                0.794277346958,
            ]
        ),
        "mc_bj": np.array(
            [
                0.0,
                0.0,
                0.333333333333,
                0.25,
                0.4,
                0.5,
            ]
        ),
    }

    for method, expected_bounds in expected.items():
        result = conformal_fdp_upper_bound(
            p_values,
            n_calibration=50,
            confidence=0.5,
            n_resamples=20,
            seed=3,
            method=method,
            thresholds=thresholds,
        )

        np.testing.assert_allclose(result.fdp_upper_bounds, expected_bounds, rtol=1e-10)


def test_global_numpy_rng_state_is_not_mutated():
    np.random.seed(123)  # noqa: NPY002
    _bounds()
    observed = np.random.random(4)  # noqa: NPY002

    np.random.seed(123)  # noqa: NPY002
    expected = np.random.random(4)  # noqa: NPY002
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_boosted_bounds_are_no_larger_than_unboosted_bounds(method):
    p_values = np.array([0.01, 0.03, 0.06, 0.4, 0.8])
    thresholds = np.array([0.03, 0.06, 0.4, 0.8])

    boosted = _bounds(
        p_values=p_values,
        thresholds=thresholds,
        seed=9,
        boost=True,
        method=method,
    )
    unboosted = _bounds(
        p_values=p_values,
        thresholds=thresholds,
        seed=9,
        boost=False,
        method=method,
    )

    assert np.all(boosted.fdp_upper_bounds <= unboosted.fdp_upper_bounds + 1e-12)


def test_conformal_fdp_upper_bound_from_result_uses_calibration_size():
    result = ConformalResult(
        p_values=np.array([0.05, 0.2, 0.4]),
        calib_scores=np.arange(12, dtype=float),
    )

    bounds = conformal_fdp_upper_bound_from_result(
        result,
        confidence=0.8,
        n_resamples=10,
        seed=1,
    )

    assert bounds.n_calibration == 12
    assert bounds.n_test == 3


def test_conformal_fdp_upper_bound_from_result_requires_p_values():
    result = ConformalResult(calib_scores=np.arange(5, dtype=float))

    with pytest.raises(ValueError, match="p_values"):
        conformal_fdp_upper_bound_from_result(
            result,
            confidence=0.8,
            n_resamples=5,
        )


def test_conformal_fdp_upper_bound_from_result_requires_calibration_scores():
    result = ConformalResult(p_values=np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match="calib_scores"):
        conformal_fdp_upper_bound_from_result(
            result,
            confidence=0.8,
            n_resamples=5,
        )


def test_conformal_fdp_upper_bound_from_result_rejects_weighted_result():
    result = ConformalResult(
        p_values=np.array([0.1, 0.2]),
        calib_scores=np.array([0.2, 0.4, 0.6]),
        test_weights=np.ones(2),
        calib_weights=np.ones(3),
    )

    with pytest.raises(ValueError, match="unweighted"):
        conformal_fdp_upper_bound_from_result(
            result,
            confidence=0.8,
            n_resamples=5,
        )


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ({"kde": {}}, "empirical"),
        (
            {"nonconform": {"strategy": "CrossValidation", "estimation": "Empirical"}},
            "split",
        ),
        (
            {"nonconform": {"strategy": "Split", "estimation": "Probabilistic"}},
            "empirical",
        ),
        (
            {
                "nonconform": {
                    "strategy": "Split",
                    "estimation": "Empirical",
                    "weighted": True,
                }
            },
            "unweighted",
        ),
    ],
)
def test_conformal_fdp_upper_bound_from_result_rejects_known_unsupported_scopes(
    metadata, match
):
    result = ConformalResult(
        p_values=np.array([0.1, 0.2]),
        calib_scores=np.array([0.2, 0.4, 0.6]),
        metadata=metadata,
    )

    with pytest.raises(ValueError, match=match):
        conformal_fdp_upper_bound_from_result(
            result,
            confidence=0.8,
            n_resamples=5,
        )
