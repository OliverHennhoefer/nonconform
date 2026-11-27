"""Tests for the conditional calibration maps behind ``ConditionalEmpirical``."""

import numpy as np
import pytest

from nonconform._internal.conditional_calibration import (
    betainv_mc,
    betainv_simes,
    calibrate_conditional_p_values,
    compute_aseq,
    compute_cn,
    compute_hybrid_bound,
    estimate_fs_correction,
)

P_VALUES = np.array([0.2, 0.8])


def _calibrate(**overrides):
    kwargs = {"n_calibration": 20, "delta": 0.1, "method": "dkwm"} | overrides
    return calibrate_conditional_p_values(kwargs.pop("p_values", P_VALUES), **kwargs)


def test_numpy_integer_n_calibration_is_accepted() -> None:
    calibrated, _ = _calibrate(n_calibration=np.int64(10))

    assert calibrated.shape == P_VALUES.shape
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"n_calibration": True}, TypeError, "n_calibration must be an integer"),
        ({"n_calibration": 2.5}, TypeError, "n_calibration must be an integer"),
        ({"n_calibration": 0}, ValueError, "n_calibration must be at least 1"),
        ({"delta": "bad"}, ValueError, r"delta must be a float in \(0, 1\)"),
        ({"delta": np.nan}, ValueError, r"delta must be in \(0, 1\)"),
        ({"p_values": np.array([0.2, np.nan])}, ValueError, "p_values must be finite"),
        (
            {"p_values": np.array([0.2, 1.5])},
            ValueError,
            r"p_values must lie in \[0, 1\]",
        ),
        (
            {"method": "simes", "simes_kden": 0},
            ValueError,
            "simes_kden must be a positive integer",
        ),
        (
            {"method": "mc", "fs_correction": np.inf},
            ValueError,
            "fs_correction must be finite",
        ),
    ],
    ids=[
        "boolean-n-calibration",
        "float-n-calibration",
        "zero-n-calibration",
        "non-numeric-delta",
        "nan-delta",
        "nan-p-value",
        "p-value-above-one",
        "zero-simes-kden",
        "infinite-fs-correction",
    ],
)
def test_invalid_calibration_inputs_are_rejected(overrides, error, match) -> None:
    with pytest.raises(error, match=match):
        _calibrate(**overrides)


@pytest.mark.parametrize(
    "transform",
    [
        lambda n_cal: compute_cn(0.1, n_cal),
        lambda n_cal: compute_hybrid_bound(0.1, n_cal, 0.0),
        lambda n_cal: betainv_mc(P_VALUES, n_cal, 0.1, fs_correction=0.0),
    ],
    ids=["cn", "hybrid-bound", "mc"],
)
def test_iterated_log_maps_require_17_calibration_points(transform) -> None:
    with pytest.raises(ValueError, match="n_calibration >= 17, got 16"):
        transform(16)


def test_simes_sequence_clamps_k_to_calibration_size() -> None:
    np.testing.assert_array_equal(compute_aseq(5, 10, 0.1), compute_aseq(5, 5, 0.1))


@pytest.mark.parametrize(
    ("k", "error", "match"),
    [(2.0, TypeError, "k must be an integer"), (0, ValueError, "k must be at least 1")],
)
def test_simes_sequence_rejects_invalid_k(k, error, match) -> None:
    with pytest.raises(error, match=match):
        compute_aseq(10, k, 0.1)


def test_fs_correction_is_zero_below_17_calibration_points() -> None:
    assert estimate_fs_correction(0.1, 16) == 0.0


@pytest.mark.parametrize("n_mc", [99, 1000.0])
def test_fs_correction_rejects_invalid_simulation_count(n_mc) -> None:
    with pytest.raises(ValueError, match="n_mc must be an integer >= 100"):
        estimate_fs_correction(0.1, 20, n_mc=n_mc)


def test_fs_correction_locates_seeded_crossing_probability_root() -> None:
    delta, n_cal, n_mc = 0.1, 20, 100
    estimates = [
        estimate_fs_correction(delta, n_cal, n_mc=n_mc, rng=np.random.default_rng(0))
        for _ in range(2)
    ]
    uniforms = np.sort(np.random.default_rng(0).uniform(size=(n_mc, n_cal)), axis=1)

    def crossing_probability(gamma: float) -> float:
        bound = compute_hybrid_bound(delta, n_cal, gamma)
        return float(np.mean(np.any(uniforms > bound, axis=1)))

    assert estimates[0] == estimates[1]
    assert -(1.0 - delta) < estimates[0] < delta
    assert crossing_probability(estimates[0] + 1e-6) <= delta
    assert crossing_probability(estimates[0] - 1e-6) > delta


def test_betainv_simes_maps_p_above_last_grid_step_to_one() -> None:
    np.testing.assert_allclose(
        betainv_simes(np.array([0.5, 1.0]), n_calibration=1, k=1, delta=0.1),
        np.array([0.9, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )

    n_cal, delta = 50, 0.05
    p_vals = np.array([n_cal / (n_cal + 1), (n_cal + 0.5) / (n_cal + 1), 1.0])
    np.testing.assert_allclose(
        betainv_simes(p_vals, n_calibration=n_cal, k=1, delta=delta),
        np.array([1.0 - delta / n_cal, 1.0, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_betainv_preserves_strict_step_boundaries_for_randomized_p_values() -> None:
    n_cal, k_simes, delta = 48, 1, 0.05
    aseq = compute_aseq(n_cal, k_simes, delta)
    grid = np.arange(1, n_cal + 1, dtype=float) / (n_cal + 1)

    just_above = np.nextafter(grid, 1.0)
    expected_above = np.concatenate([1.0 - aseq[:-1][::-1], [1.0]])
    np.testing.assert_allclose(
        betainv_simes(just_above, n_cal, k_simes, delta),
        expected_above,
        rtol=0.0,
        atol=1e-12,
    )

    just_below = np.nextafter(grid, 0.0)
    expected_below = 1.0 - aseq[::-1]
    np.testing.assert_allclose(
        betainv_simes(just_below, n_cal, k_simes, delta),
        expected_below,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        betainv_simes(np.array([0.0]), n_cal, k_simes, delta),
        np.array([1.0 - aseq[-1]]),
        rtol=0.0,
        atol=1e-12,
    )
