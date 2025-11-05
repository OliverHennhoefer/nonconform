import math
from collections.abc import Sequence

import numpy as np
import optuna
from KDEpy import FFTKDE
from sklearn.model_selection import KFold, LeaveOneOut

from nonconform.utils.func.enums import Kernel
from nonconform.utils.tune.bandwidth import (
    _scott_bandwidth,
    _sheather_jones_bandwidth,
    _silverman_bandwidth,
    compute_bandwidth_range,
)


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

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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
            grid, pdf_values = kde.evaluate()
            density_floor = np.finfo(pdf_values.dtype).tiny
            densities = np.interp(
                val_data,
                grid,
                pdf_values,
                left=density_floor,
                right=density_floor,
            )
            densities = np.maximum(densities, density_floor)
            log_likelihoods.append(np.mean(np.log(densities)))
        except Exception:
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
