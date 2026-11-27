"""Tests for conditionally calibrated empirical conformal p-values."""

import numpy as np
import pytest

from nonconform._internal import conditional_calibration
from nonconform.scoring import ConditionalEmpirical, Empirical

FS_CORRECTION = 0.03
SCORES = np.array([2.5, 1.7, 0.3, -0.4, -1.1])
CALIBRATION_20 = np.linspace(-1.8, 2.2, 20)


@pytest.fixture
def fs_correction_calls(monkeypatch):
    """Replace the Monte Carlo correction by a constant and record its calls."""
    calls: list[tuple[float, int]] = []

    def _fixed_fs_correction(
        delta: float,
        n_calibration: int,
        *,
        n_mc: int = 10_000,
        rng=None,
    ) -> float:
        del n_mc, rng
        calls.append((delta, n_calibration))
        return FS_CORRECTION

    monkeypatch.setattr(
        conditional_calibration,
        "estimate_fs_correction",
        _fixed_fs_correction,
    )
    return calls


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    weights = np.repeat(1.0, window) / float(window)
    return np.convolve(values, weights, mode="valid")


def _reference_compute_aseq(n_cal: int, k: int, delta: float) -> np.ndarray:
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n_cal - k + 1, n_cal + 1)))
    fac2 = _moving_average(np.log(np.arange(1, n_cal + 1)), k)
    return np.concatenate([np.zeros((k - 1,)), np.exp(fac2 + fac1)])


def _reference_betainv_generic(p_values: np.ndarray, aseq: np.ndarray) -> np.ndarray:
    """Give each p-value b_i for the first grid rank i with p <= i / (n_cal + 1)."""
    n_cal = len(aseq)
    envelope = [*(1.0 - aseq[::-1]), 1.0]  # b_1, ..., b_n_cal, b_{n_cal + 1} = 1
    ranks = [
        next(i for i in range(1, n_cal + 2) if p <= i / (n_cal + 1)) for p in p_values
    ]
    return np.array([envelope[i - 1] for i in ranks])


def _reference_betainv_simes(
    p_values: np.ndarray,
    n_cal: int,
    k: int,
    delta: float,
) -> np.ndarray:
    return _reference_betainv_generic(
        p_values,
        _reference_compute_aseq(n_cal, k, delta),
    )


def _reference_compute_cn(delta: float, n_cal: int) -> float:
    cn = (
        -np.log(-np.log(1.0 - delta))
        + 2.0 * np.log(np.log(n_cal))
        + 0.5 * np.log(np.log(np.log(n_cal)))
        - 0.5 * np.log(np.pi)
    )
    cn /= np.sqrt(2.0 * np.log(np.log(n_cal)))
    return float(cn)


def _reference_betainv_asymptotic(
    p_values: np.ndarray,
    n_cal: int,
    delta: float,
) -> np.ndarray:
    i = np.arange(1, n_cal + 1)
    cn = _reference_compute_cn(delta, n_cal)
    aseq = i / n_cal + cn * np.sqrt(i * (n_cal - i)) / (n_cal * np.sqrt(n_cal))
    aseq = 1.0 - np.minimum(1.0, aseq[::-1])
    return _reference_betainv_generic(p_values, aseq)


def _reference_compute_hybrid_bound(
    delta: float,
    n_cal: int,
    fs_correction: float,
) -> np.ndarray:
    i = np.arange(1, n_cal + 1)
    cn = _reference_compute_cn(delta - fs_correction, n_cal)
    bound = i / n_cal + cn * np.sqrt(i * (n_cal - i)) / (n_cal * np.sqrt(n_cal))

    k_linear = max(2, n_cal // 2)
    slope = float(bound[k_linear - 1] - bound[k_linear - 2])
    bound[k_linear:] = bound[k_linear - 1] + slope * (i[k_linear:] - k_linear)

    k_simes = max(1, n_cal // 2)
    bound_simes = 1.0 - _reference_compute_aseq(n_cal, k_simes, delta)[::-1]
    return np.minimum(bound_simes, bound)


def _reference_betainv_mc(
    p_values: np.ndarray,
    n_cal: int,
    delta: float,
    fs_correction: float,
) -> np.ndarray:
    aseq = 1.0 - np.minimum(
        1.0,
        _reference_compute_hybrid_bound(delta, n_cal, fs_correction)[::-1],
    )
    return _reference_betainv_generic(p_values, aseq)


def test_conditional_empirical_respects_unit_interval_and_monotonicity() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=32)
    calibration = rng.normal(size=64)

    estimation = ConditionalEmpirical(method="dkwm", delta=0.1)
    p_values = estimation.compute_p_values(scores, calibration)

    assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    order = np.argsort(scores)
    p_sorted = p_values[order]
    assert np.all(np.diff(p_sorted) <= 1e-12)


def test_conditional_empirical_deterministic_with_fixed_seed() -> None:
    scores = np.array([0.1, 0.2, 0.2, 0.5, 0.5])
    calibration = np.array([0.2, 0.2, 0.3, 0.4, 0.5, 0.5, 0.1, 0.0])

    p_values = []
    for _ in range(2):
        estimation = ConditionalEmpirical(
            method="simes",
            tie_break="randomized",
            delta=0.1,
        )
        estimation.set_seed(13)
        p_values.append(estimation.compute_p_values(scores, calibration))

    np.testing.assert_array_equal(*p_values)


def test_conditional_empirical_n_calib_one() -> None:
    scores = np.array([-1.0, 0.0, 1.0])
    calibration = np.array([0.5])

    estimation = ConditionalEmpirical()
    p_values = estimation.compute_p_values(scores, calibration)

    assert p_values.shape == scores.shape
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))

    # Simes uses k = 1 here; a score at or below the calibration score must get 1.
    simes = ConditionalEmpirical(method="simes", delta=0.1)
    np.testing.assert_allclose(
        simes.compute_p_values(scores, calibration),
        np.array([1.0, 1.0, 0.9]),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("method", "calibration", "delta", "reference"),
    [
        (
            "simes",
            np.array([3.1, 2.0, 1.5, 1.1, 0.6, 0.2, -0.1, -0.3, -0.9, -1.5]),
            0.2,
            lambda p, n_cal, delta: _reference_betainv_simes(
                p, n_cal, n_cal // 2, delta
            ),
        ),
        ("asymptotic", CALIBRATION_20, 0.1, _reference_betainv_asymptotic),
        (
            "mc",
            CALIBRATION_20,
            0.1,
            lambda p, n_cal, delta: _reference_betainv_mc(
                p, n_cal, delta, FS_CORRECTION
            ),
        ),
    ],
    ids=["simes", "asymptotic", "mc-fixed-fs-correction"],
)
def test_conditional_empirical_matches_reference_transform(
    fs_correction_calls,
    method: str,
    calibration: np.ndarray,
    delta: float,
    reference,
) -> None:
    base_p = Empirical(tie_break="classical").compute_p_values(SCORES, calibration)
    expected = reference(base_p, len(calibration), delta)

    estimation = ConditionalEmpirical(
        method=method,
        tie_break="classical",
        delta=delta,
        simes_kden=2,
    )
    actual = estimation.compute_p_values(SCORES, calibration)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("simes", lambda p: _reference_betainv_simes(p, 48, 24, 0.05)),
        ("asymptotic", lambda p: _reference_betainv_asymptotic(p, 48, 0.05)),
        ("mc", lambda p: _reference_betainv_mc(p, 48, 0.05, FS_CORRECTION)),
    ],
    ids=["simes", "asymptotic", "mc-fixed-fs-correction"],
)
def test_conditional_empirical_matches_reference_at_every_rank(
    fs_correction_calls,
    method: str,
    reference,
) -> None:
    # n_cal = 48 has ranks, such as 22 / 49, where a floor of the rescaled
    # p-value lands one step off; scores between calibration points hit all ranks.
    calibration = np.arange(48.0)
    scores = np.arange(49.0) - 0.5
    base_p = Empirical(tie_break="classical").compute_p_values(scores, calibration)
    assert 22 / 49 in base_p

    estimation = ConditionalEmpirical(method=method, tie_break="classical", delta=0.05)
    actual = estimation.compute_p_values(scores, calibration)
    np.testing.assert_allclose(actual, reference(base_p), rtol=0.0, atol=1e-12)
    assert actual[0] == 1.0


def test_conditional_empirical_rejects_weighted_inputs() -> None:
    estimation = ConditionalEmpirical(method="dkwm")
    with pytest.raises(ValueError, match="does not support weighted p-values"):
        estimation.compute_p_values(
            np.array([0.1, 0.2]),
            np.array([0.0, 0.3, 0.7]),
            weights=(np.ones(3), np.ones(2)),
        )


def test_conditional_empirical_accepts_empty_scores() -> None:
    estimation = ConditionalEmpirical(method="simes")
    output = estimation.compute_p_values(np.array([]), np.array([0.1, 0.2, 0.3]))
    assert output.shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"delta": -0.1}, ValueError, "delta must be in"),
        ({"delta": 1.0}, ValueError, "delta must be in"),
        ({"delta": "bad"}, ValueError, r"delta must be a float in \(0, 1\)"),
        ({"method": "unknown"}, ValueError, "must be one of"),
        ({"method": None}, TypeError, "method must be a string"),
        ({"simes_kden": 0}, ValueError, "simes_kden must be a positive integer"),
        ({"simes_kden": True}, ValueError, "simes_kden must be a positive integer"),
        (
            {"mc_num_simulations": 10},
            ValueError,
            "mc_num_simulations must be an integer >= 100",
        ),
        (
            {"mc_num_simulations": False},
            ValueError,
            "mc_num_simulations must be an integer >= 100",
        ),
    ],
)
def test_conditional_empirical_invalid_init_raises(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ConditionalEmpirical(**kwargs)


@pytest.mark.parametrize("method", ["mc", "asymptotic"])
@pytest.mark.parametrize(
    ("n_calib", "falls_back"),
    [(5, True), (16, True), (17, False)],
)
def test_iterated_log_methods_fall_back_to_dkwm_below_17_calibration_points(
    fs_correction_calls,
    method: str,
    n_calib: int,
    falls_back: bool,
) -> None:
    scores = np.array([-1.5, -0.1, 0.7])
    calibration = np.linspace(-2.0, 2.0, n_calib)
    estimation = ConditionalEmpirical(method=method, delta=0.1)
    dkwm = ConditionalEmpirical(method="dkwm", delta=0.1)

    matches_dkwm = np.allclose(
        estimation.compute_p_values(scores, calibration),
        dkwm.compute_p_values(scores, calibration),
        rtol=0.0,
        atol=1e-12,
    )

    assert matches_dkwm == falls_back


def test_conditional_empirical_mc_cache_key_and_seed_reset(
    fs_correction_calls,
) -> None:
    scores = np.array([0.7, -0.2, -1.1], dtype=float)
    calibration_20 = np.linspace(-2.0, 2.0, 20, dtype=float)
    calibration_21 = np.linspace(-2.0, 2.0, 21, dtype=float)

    estimation = ConditionalEmpirical(method="mc", tie_break="classical", delta=0.1)

    estimation.compute_p_values(scores, calibration_20)
    estimation.compute_p_values(scores, calibration_20)
    assert fs_correction_calls == [(0.1, 20)]

    estimation.compute_p_values(scores, calibration_21)
    assert fs_correction_calls == [(0.1, 20), (0.1, 21)]

    estimation.set_seed(123)
    estimation.compute_p_values(scores, calibration_20)
    assert fs_correction_calls == [(0.1, 20), (0.1, 21), (0.1, 20)]
