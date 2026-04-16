import numpy as np

from nonconform.scoring import ConditionalEmpirical, Empirical


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    weights = np.repeat(1.0, window) / float(window)
    return np.convolve(values, weights, mode="valid")


def _reference_compute_aseq(n_cal: int, k: int, delta: float) -> np.ndarray:
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n_cal - k + 1, n_cal + 1)))
    fac2 = _moving_average(np.log(np.arange(1, n_cal + 1)), k)
    return np.concatenate([np.zeros((k - 1,)), np.exp(fac2 + fac1)])


def _reference_betainv_generic(p_values: np.ndarray, aseq: np.ndarray) -> np.ndarray:
    n_cal = len(aseq)
    idx = np.floor((n_cal + 1) * (1.0 - p_values)).astype(int)
    idx = np.clip(idx, 1, n_cal)
    return 1.0 - aseq[idx - 1]


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

    first = ConditionalEmpirical(
        method="simes",
        tie_break="randomized",
        delta=0.1,
    )
    first.set_seed(13)

    second = ConditionalEmpirical(
        method="simes",
        tie_break="randomized",
        delta=0.1,
    )
    second.set_seed(13)

    p_first = first.compute_p_values(scores, calibration)
    p_second = second.compute_p_values(scores, calibration)
    np.testing.assert_array_equal(p_first, p_second)


def test_conditional_empirical_n_calib_one() -> None:
    scores = np.array([-1.0, 0.0, 1.0])
    calibration = np.array([0.5])

    estimation = ConditionalEmpirical()
    p_values = estimation.compute_p_values(scores, calibration)

    assert p_values.shape == scores.shape
    assert np.all((0.0 <= p_values) & (p_values <= 1.0))


def test_conditional_empirical_matches_reference_vector_for_simes() -> None:
    scores = np.array([2.5, 1.7, 0.3, -0.4, -1.1], dtype=float)
    calibration = np.array(
        [3.1, 2.0, 1.5, 1.1, 0.6, 0.2, -0.1, -0.3, -0.9, -1.5],
        dtype=float,
    )
    delta = 0.2
    simes_kden = 2

    base_p = Empirical(tie_break="classical").compute_p_values(scores, calibration)
    k = max(1, int(len(calibration) / simes_kden))
    expected = _reference_betainv_simes(base_p, len(calibration), k, delta)

    estimation = ConditionalEmpirical(
        method="simes",
        tie_break="classical",
        delta=delta,
        simes_kden=simes_kden,
    )
    actual = estimation.compute_p_values(scores, calibration)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_conditional_empirical_matches_reference_vector_for_asymptotic() -> None:
    scores = np.array([2.5, 1.7, 0.3, -0.4, -1.1], dtype=float)
    calibration = np.linspace(-1.8, 2.2, 20, dtype=float)
    delta = 0.1

    base_p = Empirical(tie_break="classical").compute_p_values(scores, calibration)
    expected = _reference_betainv_asymptotic(base_p, len(calibration), delta)

    estimation = ConditionalEmpirical(
        method="asymptotic",
        tie_break="classical",
        delta=delta,
    )
    actual = estimation.compute_p_values(scores, calibration)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_conditional_empirical_matches_reference_vector_for_mc_fixed_fs_correction(
    monkeypatch,
) -> None:
    scores = np.array([2.5, 1.7, 0.3, -0.4, -1.1], dtype=float)
    calibration = np.linspace(-1.8, 2.2, 20, dtype=float)
    delta = 0.1
    fs_correction = 0.03

    from nonconform._internal import conditional_calibration

    def _fixed_fs_correction(
        delta: float,
        n_calibration: int,
        *,
        n_mc: int = 10_000,
        rng=None,
    ) -> float:
        del delta, n_calibration, n_mc, rng
        return fs_correction

    monkeypatch.setattr(
        conditional_calibration,
        "estimate_fs_correction",
        _fixed_fs_correction,
    )

    base_p = Empirical(tie_break="classical").compute_p_values(scores, calibration)
    expected = _reference_betainv_mc(base_p, len(calibration), delta, fs_correction)

    estimation = ConditionalEmpirical(
        method="mc",
        tie_break="classical",
        delta=delta,
    )
    actual = estimation.compute_p_values(scores, calibration)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
