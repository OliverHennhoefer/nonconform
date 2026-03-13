"""Conditional calibration transforms for conformal p-values.

This module provides finite-sample calibration mappings used by
``ConditionalEmpirical`` in :mod:`nonconform.scoring`.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

ConditionalCalibrationMethod = Literal["mc", "simes", "dkwm", "asymptotic"]

_MIN_ASYMPTOTIC_N_CAL = 17
_BINARY_SEARCH_TOL = 1e-6
_GAMMA_MARGIN = 1e-6


def normalize_conditional_calibration_method(
    method: str | ConditionalCalibrationMethod,
) -> ConditionalCalibrationMethod:
    """Normalize conditional calibration method names."""
    if not isinstance(method, str):
        raise TypeError(
            "method must be a string in {'mc', 'simes', 'dkwm', 'asymptotic'}."
        )
    normalized = method.strip().lower()
    valid = {"mc", "simes", "dkwm", "asymptotic"}
    if normalized not in valid:
        raise ValueError(
            "method must be one of {'mc', 'simes', 'dkwm', 'asymptotic'}. "
            f"Got {method!r}."
        )
    return cast(ConditionalCalibrationMethod, normalized)


def _validate_delta(delta: float) -> float:
    """Validate calibration confidence level."""
    try:
        delta_value = float(delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta must be a float in (0, 1).") from exc
    if not np.isfinite(delta_value) or not (0.0 < delta_value < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}.")
    return delta_value


def _validate_n_calibration(n_calibration: int) -> int:
    """Validate calibration set size."""
    if isinstance(n_calibration, (bool, np.bool_)) or not isinstance(
        n_calibration, (int, np.integer)
    ):
        raise TypeError(
            "n_calibration must be an integer with at least one calibration point."
        )
    n_cal = int(n_calibration)
    if n_cal < 1:
        raise ValueError(
            "n_calibration must be at least 1 for conditional calibration."
        )
    return n_cal


def _validate_p_values(p_values: np.ndarray) -> np.ndarray:
    """Validate p-value vector for conditional transforms."""
    arr = np.asarray(p_values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("p_values must be finite.")
    eps = 1e-10
    if np.any((arr < -eps) | (arr > 1.0 + eps)):
        raise ValueError("p_values must lie in [0, 1].")
    return np.clip(arr, 0.0, 1.0)


def compute_cn(delta: float, n_calibration: int) -> float:
    """Compute iterated-log correction constant from CCCPV reference code."""
    delta_value = _validate_delta(delta)
    n_cal = _validate_n_calibration(n_calibration)
    if n_cal < _MIN_ASYMPTOTIC_N_CAL:
        raise ValueError(
            "MC/Asymptotic conditional calibration requires "
            f"n_calibration >= {_MIN_ASYMPTOTIC_N_CAL}, got {n_cal}."
        )

    cn = (
        -np.log(-np.log(1.0 - delta_value))
        + 2.0 * np.log(np.log(n_cal))
        + 0.5 * np.log(np.log(np.log(n_cal)))
        - 0.5 * np.log(np.pi)
    )
    cn /= np.sqrt(2.0 * np.log(np.log(n_cal)))
    return float(cn)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average with valid-window semantics."""
    weights = np.repeat(1.0, window) / float(window)
    return np.convolve(values, weights, mode="valid")


def compute_aseq(n_calibration: int, k: int, delta: float) -> np.ndarray:
    """Compute the Simes calibration sequence."""
    n_cal = _validate_n_calibration(n_calibration)
    delta_value = _validate_delta(delta)
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if k > n_cal:
        k = n_cal

    fac1 = np.log(delta_value) / k - np.mean(
        np.log(np.arange(n_cal - k + 1, n_cal + 1))
    )
    fac2 = _moving_average(np.log(np.arange(1, n_cal + 1)), k)
    aseq = np.concatenate([np.zeros((k - 1,)), np.exp(fac2 + fac1)])
    return aseq


def _betainv_generic(p_values: np.ndarray, aseq: np.ndarray) -> np.ndarray:
    """Apply inverse beta-sequence transform."""
    p_vals = _validate_p_values(p_values)
    n_cal = len(aseq)
    idx = np.floor((n_cal + 1) * (1.0 - p_vals)).astype(int)
    idx = np.clip(idx, 1, n_cal)
    return 1.0 - aseq[idx - 1]


def betainv_simes(
    p_values: np.ndarray,
    n_calibration: int,
    k: int,
    delta: float,
) -> np.ndarray:
    """Simes-based conditional calibration map."""
    aseq = compute_aseq(n_calibration, k, delta)
    return _betainv_generic(p_values, aseq)


def betainv_asymptotic(
    p_values: np.ndarray,
    n_calibration: int,
    delta: float,
) -> np.ndarray:
    """Asymptotic conditional calibration map."""
    n_cal = _validate_n_calibration(n_calibration)
    cn = compute_cn(delta, n_cal)
    i = np.arange(1, n_cal + 1)
    aseq = i / n_cal + cn * np.sqrt(i * (n_cal - i)) / (n_cal * np.sqrt(n_cal))
    aseq = 1.0 - np.minimum(1.0, aseq[::-1])
    return _betainv_generic(p_values, aseq)


def compute_hybrid_bound(
    delta: float,
    n_calibration: int,
    gamma: float,
) -> np.ndarray:
    """Compute hybrid linear/Simes boundary used by MC calibration."""
    n_cal = _validate_n_calibration(n_calibration)
    if n_cal < _MIN_ASYMPTOTIC_N_CAL:
        raise ValueError(
            "Hybrid bound requires n_calibration >= "
            f"{_MIN_ASYMPTOTIC_N_CAL}, got {n_cal}."
        )

    i = np.arange(1, n_cal + 1)
    cna = compute_cn(delta - gamma, n_cal)
    bound = i / n_cal + cna * np.sqrt(i * (n_cal - i)) / (n_cal * np.sqrt(n_cal))

    k_linear = max(2, n_cal // 2)
    slope = float(bound[k_linear - 1] - bound[k_linear - 2])
    bound[k_linear:] = bound[k_linear - 1] + slope * (i[k_linear:] - k_linear)

    k_simes = max(1, n_cal // 2)
    bound_simes = 1.0 - compute_aseq(n_cal, k_simes, delta)[::-1]
    return np.minimum(bound_simes, bound)


def estimate_fs_correction(
    delta: float,
    n_calibration: int,
    *,
    n_mc: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate finite-sample correction used by MC conditional calibration."""
    delta_value = _validate_delta(delta)
    n_cal = _validate_n_calibration(n_calibration)
    if n_cal < _MIN_ASYMPTOTIC_N_CAL:
        return 0.0
    if not isinstance(n_mc, int) or n_mc < 100:
        raise ValueError("n_mc must be an integer >= 100.")

    generator = np.random.default_rng() if rng is None else rng
    uniforms = np.sort(generator.uniform(size=(n_mc, n_cal)), axis=1)

    def estimate_prob_crossing(gamma: float) -> float:
        bound = compute_hybrid_bound(delta_value, n_cal, gamma)
        crossings = np.sum(uniforms > bound, axis=1)
        return float(np.mean(crossings > 0))

    gamma_low = -(1.0 - _GAMMA_MARGIN - delta_value)
    gamma_high = delta_value - _GAMMA_MARGIN

    while abs(gamma_high - gamma_low) > _BINARY_SEARCH_TOL:
        gamma_mid = 0.5 * (gamma_low + gamma_high)
        crossing_prob = estimate_prob_crossing(gamma_mid)
        if crossing_prob > delta_value:
            gamma_low = gamma_mid
        else:
            gamma_high = gamma_mid

    return float(0.5 * (gamma_low + gamma_high))


def betainv_mc(
    p_values: np.ndarray,
    n_calibration: int,
    delta: float,
    *,
    fs_correction: float,
) -> np.ndarray:
    """Monte-Carlo conditional calibration map."""
    n_cal = _validate_n_calibration(n_calibration)
    if n_cal < _MIN_ASYMPTOTIC_N_CAL:
        raise ValueError(
            "MC conditional calibration requires "
            f"n_calibration >= {_MIN_ASYMPTOTIC_N_CAL}, got {n_cal}."
        )
    aseq = 1.0 - np.minimum(
        1.0,
        compute_hybrid_bound(delta, n_cal, fs_correction)[::-1],
    )
    return _betainv_generic(p_values, aseq)


def calibrate_conditional_p_values(
    p_values: np.ndarray,
    *,
    n_calibration: int,
    delta: float,
    method: str | ConditionalCalibrationMethod,
    simes_kden: int = 2,
    fs_correction: float | None = None,
    rng: np.random.Generator | None = None,
    mc_num_simulations: int = 10_000,
) -> tuple[np.ndarray, float | None]:
    """Calibrate conformal p-values with a conditional validity transform.

    For small calibration sets, ``"mc"`` and ``"asymptotic"`` automatically
    fall back to ``"dkwm"`` because iterated-log constants are undefined.
    """
    p_vals = _validate_p_values(p_values)
    if p_vals.size == 0:
        return p_vals.copy(), fs_correction

    n_cal = _validate_n_calibration(n_calibration)
    delta_value = _validate_delta(delta)
    normalized_method = normalize_conditional_calibration_method(method)

    if normalized_method in {"mc", "asymptotic"} and n_cal < _MIN_ASYMPTOTIC_N_CAL:
        normalized_method = "dkwm"

    if normalized_method == "dkwm":
        epsilon = np.sqrt(np.log(2.0 / delta_value) / (2.0 * n_cal))
        return np.minimum(1.0, p_vals + epsilon), fs_correction

    if not isinstance(simes_kden, int) or simes_kden < 1:
        raise ValueError("simes_kden must be a positive integer.")
    k = max(1, int(n_cal / simes_kden))

    if normalized_method == "simes":
        return np.clip(
            betainv_simes(p_vals, n_cal, k, delta_value),
            0.0,
            1.0,
        ), fs_correction

    if normalized_method == "asymptotic":
        return np.clip(
            betainv_asymptotic(p_vals, n_cal, delta_value),
            0.0,
            1.0,
        ), fs_correction

    if fs_correction is None:
        fs_correction = estimate_fs_correction(
            delta_value,
            n_cal,
            n_mc=mc_num_simulations,
            rng=rng,
        )
    else:
        fs_correction = float(fs_correction)
        if not np.isfinite(fs_correction):
            raise ValueError("fs_correction must be finite.")

    return np.clip(
        betainv_mc(
            p_vals,
            n_cal,
            delta_value,
            fs_correction=fs_correction,
        ),
        0.0,
        1.0,
    ), fs_correction


__all__ = [
    "ConditionalCalibrationMethod",
    "betainv_asymptotic",
    "betainv_mc",
    "betainv_simes",
    "calibrate_conditional_p_values",
    "compute_aseq",
    "compute_cn",
    "compute_hybrid_bound",
    "estimate_fs_correction",
    "normalize_conditional_calibration_method",
]
