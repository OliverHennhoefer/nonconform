"""KDE tuning and bandwidth estimation utilities.

This module provides hyperparameter tuning for kernel density estimation
and rule-of-thumb bandwidth selectors.
"""

import logging
import math
import sys
from collections.abc import Sequence

import numpy as np
import optuna
from KDEpy import FFTKDE
from sklearn.model_selection import KFold, LeaveOneOut

from .constants import Kernel

# Statistical constants for bandwidth estimation
# Scott (1992): h = 1.06 * sigma * n^(-1/5)
SCOTT_CONSTANT = 1.06

# Silverman (1986): h = 0.9 * min(sigma, IQR/1.349) * n^(-1/5)
SILVERMAN_CONSTANT = 0.9

# IQR to standard deviation conversion factor for normal distribution
# For N(0,1): IQR = Q3 - Q1 = 0.6745 - (-0.6745) = 1.349 * sigma
IQR_TO_STD_FACTOR = 1.349

# Maximum ratio between bandwidth bounds to prevent extreme search ranges
MAX_BANDWIDTH_RATIO = 1000


def _scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule-of-thumb bandwidth estimation.

    Reference: Scott, D.W. (1992). Multivariate Density Estimation.
    """
    n = len(data)
    sigma = np.std(data, ddof=1)
    return SCOTT_CONSTANT * sigma * n ** (-0.2)


def _silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth estimation.

    Reference: Silverman, B.W. (1986). Density Estimation for Statistics
    and Data Analysis.
    """
    n = len(data)
    sigma = np.std(data, ddof=1)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    sigma_hat = min(sigma, iqr / IQR_TO_STD_FACTOR) if iqr > 0 else sigma
    return SILVERMAN_CONSTANT * sigma_hat * n ** (-0.2)


def _sheather_jones_bandwidth(data: np.ndarray) -> float:
    """Sheather-Jones bandwidth selector with fallback to Silverman."""
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(data, bw_method="scott")
        return kde.factor * np.std(data, ddof=1)
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # Fallback to Silverman if scipy unavailable or numerical issues occur
        return _silverman_bandwidth(data)


def compute_bandwidth_range(data: np.ndarray) -> tuple[float, float]:
    """Compute bandwidth search range using robust statistics.

    Uses percentile-based range and IQR-based spread to be robust against
    outliers that can cause extreme bandwidth ranges.
    """
    # Robust range: use percentiles instead of min/max to ignore outliers
    q1, q99 = np.percentile(data, [1, 99])
    robust_range = q99 - q1

    # Robust spread: IQR-based (same approach as Silverman's rule)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    robust_std = iqr / IQR_TO_STD_FACTOR if iqr > 0 else float(np.std(data))

    bw_min = min(robust_range * 0.001, robust_std * 0.01)
    bw_max = max(robust_range * 0.5, robust_std * 2)

    # Cap the ratio to prevent extreme search ranges
    if bw_max / max(bw_min, 1e-10) > MAX_BANDWIDTH_RATIO:
        bw_max = bw_min * MAX_BANDWIDTH_RATIO

    return bw_min, bw_max


def tune_kde_hyperparameters(
    calibration_set: np.ndarray,
    kernel_options: Sequence[Kernel] | Kernel,
    n_trials: int = 100,
    cv_folds: int = -1,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """Tune KDE hyperparameters using Optuna with cross-validated log-likelihood.

    The bandwidth search range is derived automatically from the calibration data,
    so callers only need to specify which kernels to consider.

    Args:
        calibration_set: Calibration scores for tuning.
        kernel_options: Kernel enum or iterable of kernels for the search space.
        n_trials: Number of Optuna trials; if <= 0, returns heuristic defaults.
        cv_folds: Cross-validation folds (-1 for leave-one-out).
        weights: Optional calibration weights for weighted KDE.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'bandwidth', 'kernel', 'best_score', 'study'.
    """
    calibration_set = calibration_set.ravel()

    # Sort data and weights together to maintain correspondence
    if weights is not None:
        sort_idx = np.argsort(calibration_set)
        calibration_set = calibration_set[sort_idx]
        weights = weights[sort_idx]
    else:
        calibration_set = np.sort(calibration_set)

    kernels = _normalise_kernels(kernel_options)

    bw_min, bw_max = compute_bandwidth_range(calibration_set)
    bw_min = float(max(bw_min, 1e-6))
    bw_max = float(max(bw_max, bw_min * 1.01))

    heuristic_bandwidths = _collect_heuristic_bandwidths(
        calibration_set, bw_min, bw_max
    )
    default_kernel = kernels[0]
    default_bandwidth = heuristic_bandwidths[0]

    if n_trials <= 0:
        return {
            "bandwidth": default_bandwidth,
            "kernel": default_kernel,
            "best_score": None,
            "study": None,
        }

    # Suppress Optuna's default output - route through project logger
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger = logging.getLogger("nonconform")

    warmup_steps = int(0.3 * n_trials)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=warmup_steps)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    for kernel in kernels:
        for bandwidth in heuristic_bandwidths:
            params = {"bandwidth": float(np.clip(bandwidth, bw_min, bw_max))}
            if len(kernels) > 1:
                params["kernel"] = kernel.value
            study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        if len(kernels) > 1:
            kernel_value = trial.suggest_categorical(
                "kernel", [k.value for k in kernels]
            )
            kernel_enum = next(k for k in kernels if k.value == kernel_value)
        else:
            kernel_enum = kernels[0]

        bandwidth = trial.suggest_float(
            "bandwidth",
            bw_min,
            bw_max,
            log=True,
        )

        return _compute_cv_log_likelihood(
            calibration_set, kernel_enum, bandwidth, cv_folds, weights, seed
        )

    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if logger.isEnabledFor(logging.INFO):
            kernel_val = trial.params.get("kernel", kernels[0].value)
            logger.info(
                f"Trial {trial.number}: bandwidth={trial.params['bandwidth']:.6f}, "
                f"kernel={kernel_val}, score={trial.value:.4f}"
            )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[trial_callback] if logger.isEnabledFor(logging.INFO) else None,
    )

    best_kernel = (
        next(
            k
            for k in kernels
            if k.value == study.best_params.get("kernel", kernels[0].value)
        )
        if len(kernels) > 1
        else kernels[0]
    )
    best_bandwidth = float(np.clip(study.best_params["bandwidth"], bw_min, bw_max))

    return {
        "bandwidth": best_bandwidth,
        "kernel": best_kernel,
        "best_score": study.best_value,
        "study": study,
    }


def _normalise_kernels(kernel_options: Sequence[Kernel] | Kernel) -> list[Kernel]:
    """Convert kernel input into a non-empty list of Kernel enums."""
    if isinstance(kernel_options, Kernel):
        return [kernel_options]

    kernels = [kernel for kernel in kernel_options]
    if not kernels:
        kernels = [Kernel.GAUSSIAN]
    return kernels


def _collect_heuristic_bandwidths(
    data: np.ndarray, bw_min: float, bw_max: float
) -> list[float]:
    """Gather rule-of-thumb bandwidths clipped to the admissible search range."""
    candidates = [
        _sheather_jones_bandwidth(data),
        _scott_bandwidth(data),
        _silverman_bandwidth(data),
    ]

    heuristics: list[float] = []
    for bw in candidates:
        if not np.isfinite(bw) or bw <= 0:
            continue
        clipped = float(np.clip(bw, bw_min, bw_max))
        if any(
            math.isclose(clipped, existing, rel_tol=1e-6) for existing in heuristics
        ):
            continue
        heuristics.append(clipped)

    if not heuristics:
        heuristics.append(float(max(np.std(data), bw_min)))

    return heuristics


def _compute_cv_log_likelihood(
    data: np.ndarray,
    kernel: Kernel,
    bandwidth: float,
    cv_folds: int,
    weights: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    """Compute cross-validated log-likelihood for KDE using sklearn CV."""
    if cv_folds == -1:
        cv_splitter = LeaveOneOut()
    else:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    iterator = cv_splitter.split(data)

    log_likelihoods = []
    for train_idx, val_idx in iterator:
        train_data = data[train_idx]
        val_data = np.ravel(data[val_idx])
        train_weights = weights[train_idx] if weights is not None else None

        try:
            kde = _fit_kde(train_data, bandwidth, kernel, train_weights)
            grid, pdf_values = kde.evaluate(2**14)
            density_floor = sys.float_info.min
            densities = np.interp(
                val_data,
                grid,
                pdf_values,
                left=density_floor,
                right=density_floor,
            )
            densities = np.maximum(densities, density_floor)
            log_likelihoods.append(np.mean(np.log(densities)))
        except (ValueError, np.linalg.LinAlgError):
            # KDE fitting can fail with singular data or numerical issues
            return -np.inf

    return np.mean(log_likelihoods)


def _fit_kde(
    data: np.ndarray,
    bandwidth: float,
    kernel: Kernel,
    weights: np.ndarray | None = None,
) -> FFTKDE:
    """Fit FFTKDE model."""
    data = data.ravel()

    # Sort data and weights together to maintain correspondence
    if weights is not None:
        sort_idx = np.argsort(data)
        data = data[sort_idx]
        weights = weights[sort_idx]
    else:
        data = np.sort(data)

    kde = FFTKDE(kernel=kernel.value, bw=bandwidth)
    if weights is not None:
        kde.fit(data, weights=weights)
    else:
        kde.fit(data)
    return kde


__all__ = [
    "IQR_TO_STD_FACTOR",
    "MAX_BANDWIDTH_RATIO",
    "SCOTT_CONSTANT",
    "SILVERMAN_CONSTANT",
    "_collect_heuristic_bandwidths",
    "_compute_cv_log_likelihood",
    "_fit_kde",
    "_normalise_kernels",
    "_scott_bandwidth",
    "_sheather_jones_bandwidth",
    "_silverman_bandwidth",
    "compute_bandwidth_range",
    "tune_kde_hyperparameters",
]
