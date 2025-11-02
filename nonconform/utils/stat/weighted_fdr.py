"""False Discovery Rate control for conformal prediction.

This module implements Weighted Conformalized Selection (WCS) for FDR control
under covariate shift. For standard BH/BY procedures, use
scipy.stats.false_discovery_control.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from nonconform.utils.func.enums import Pruning
from nonconform.utils.func.logger import get_logger
from nonconform.utils.stat.statistical import calculate_weighted_p_val


def _bh_rejection_indices(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return indices of BH rejection set for given p-values.

    This helper mimics the Benjamini-Hochberg procedure: sort p-values,
    find the largest k such that p_(k) ≤ q*k/m, and return the first k
    indices in the sorted order.

    Args:
        p_values: Array of p-values to apply BH procedure on.
        q: Target false discovery rate threshold.

    Returns:
        Array of indices in the rejection set. Returns empty array if no
        p-value meets the criterion.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=int)
    # Sort indices by p-value
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    # Thresholds q * (1:m) / m
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return np.array([], dtype=int)
    k = below[-1]
    return sorted_idx[: k + 1]


def _bh_rejection_count(p_values: np.ndarray, thresholds: np.ndarray) -> int:
    """Return size of BH rejection set for given p-values."""
    if p_values.size == 0:
        return 0
    sorted_p = np.sort(p_values)
    below = np.nonzero(sorted_p <= thresholds)[0]
    return 0 if len(below) == 0 else int(below[-1] + 1)


def _calib_weight_mass_below(
    calib_scores: np.ndarray, w_calib: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Compute weighted calibration mass strictly below each target score."""
    if len(calib_scores) == 0:
        return np.zeros_like(targets, dtype=float)
    order = np.argsort(calib_scores)
    sorted_scores = calib_scores[order]
    sorted_weights = w_calib[order]
    cum_weights = np.concatenate(([0.0], np.cumsum(sorted_weights)))
    positions = np.searchsorted(sorted_scores, targets, side="left")
    return cum_weights[positions]


def _prune_heterogeneous(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Heterogeneous pruning with independent random variables.

    Uses independent ξ_j ∈ [0,1] for each candidate, ordering by ξ_j * |R_j^{(0)}|.

    Args:
        first_sel_idx: Indices of first selection set R^{(1)}.
        sizes_sel: Sizes |R_j^{(0)}| for each j in first_sel_idx.
        rng: Random number generator for ξ_j.

    Returns:
        Sorted array of final selected indices.
    """
    xi = rng.uniform(size=len(first_sel_idx))
    order = np.argsort(xi * sizes_sel)
    return np.sort(first_sel_idx[order])


def _prune_homogeneous(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Homogeneous pruning with shared random variable.

    Uses a single shared ξ ∈ [0,1] across all candidates, ordering by ξ * |R_j^{(0)}|.

    Args:
        first_sel_idx: Indices of first selection set R^{(1)}.
        sizes_sel: Sizes |R_j^{(0)}| for each j in first_sel_idx.
        rng: Random number generator for ξ.

    Returns:
        Sorted array of final selected indices.
    """
    xi = rng.uniform()
    order = np.argsort(xi * sizes_sel)
    return np.sort(first_sel_idx[order])


def _prune_deterministic(
    first_sel_idx: np.ndarray, sizes_sel: np.ndarray
) -> np.ndarray:
    """Deterministic pruning based on rejection set sizes.

    Orders candidates by |R_j^{(0)}| without randomization.

    Args:
        first_sel_idx: Indices of first selection set R^{(1)}.
        sizes_sel: Sizes |R_j^{(0)}| for each j in first_sel_idx.

    Returns:
        Sorted array of final selected indices.
    """
    order = np.argsort(sizes_sel)
    return np.sort(first_sel_idx[order])


def _compute_rejection_set_size_for_instance(
    j: int,
    test_scores: np.ndarray,
    w_test: np.ndarray,
    sum_calib_weight: float,
    bh_thresholds: np.ndarray,
    calib_mass_below: np.ndarray,
    scratch: np.ndarray,
) -> int:
    """Compute rejection set size |R_j^{(0)}| for test instance j.

    For a given test instance j, computes auxiliary p-values for all other
    test instances and applies the Benjamini-Hochberg procedure to determine
    the rejection set size.

    Args:
        j: Index of the test instance to compute rejection set size for.
        test_scores: Non-conformity scores for all test instances.
        w_test: Importance weights for all test instances.
        sum_calib_weight: Sum of all calibration weights (precomputed).
        bh_thresholds: Precomputed BH thresholds q * (1:m) / m.
        calib_mass_below: Weighted calibration mass strictly below each test score.
        scratch: Workspace array for computing auxiliary p-values.

    Returns:
        Size of the rejection set |R_j^{(0)}| for instance j.
    """
    np.copyto(scratch, calib_mass_below)
    scratch += w_test[j] * (test_scores[j] < test_scores)
    scratch[j] = 0.0
    denominator = sum_calib_weight + w_test[j]
    scratch /= denominator
    scratch[j] = 0.0
    return _bh_rejection_count(scratch, bh_thresholds)


def weighted_false_discovery_control(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    w_test: np.ndarray,
    w_calib: np.ndarray,
    q: float,
    pruning: Pruning = Pruning.DETERMINISTIC,
    seed: int | None = None,
) -> np.ndarray:
    """Perform Weighted Conformalized Selection (WCS).

    Args:
        test_scores: Non-conformity scores for the test data (length m).
        calib_scores: Non-conformity scores for the calibration data (length n).
        w_test: Importance weights for the test data (length m).
        w_calib: Importance weights for the calibration data (length n).
        q: Target false discovery rate (0 < q < 1).
        pruning: Pruning method. ``'hete'`` (heterogeneous pruning) uses
            independent random variables l_j; ``'homo'`` (homogeneous
            pruning) uses a single random variable l shared across
            candidates; ``'dtm'`` (deterministic) performs deterministic
            pruning based on |R_j^{(0)}|. Defaults to ``'dtm'``.
        seed: Random seed for reproducibility. Defaults to None
            (non-deterministic).

    Returns:
        Boolean mask of test points retained after pruning (final selection).
        For deterministic pruning (``'dtm'``), this may coincide with the
        first selection step.

    Note:
        The procedure follows Algorithm 1 in Jin & Candes (2023):

        1. Compute weighted conformal p-values ``p_vals`` for the test
           points.
        2. For each j, compute auxiliary p-values p^{(j)}_l (l ≠ j) and
           form the BH rejection set R_j^{(0)} on these auxiliary
           p-values; set s_j = q * |R_j^{(0)}| / m.
        3. Form the first selection set R^{(1)} = {j: p_j ≤ s_j}.
        4. Prune R^{(1)} using the specified method:
           * ``'hete'``: heterogeneous pruning with independent ξ_j.
           * ``'homo'``: homogeneous pruning with a shared ξ.
           * ``'dtm'``: deterministic pruning based on |R_j^{(0)}|.
        5. Return boolean mask for final selected test points.

        Computational cost is O(m^2) in the number of test points.

    References:
        Jin, Y., & Candes, E. (2023). Model-free selective inference under
        covariate shift via weighted conformal p-values. arXiv preprint
        arXiv:2307.09291.
    """
    # Convert inputs to numpy arrays
    test_scores = np.asarray(test_scores)
    calib_scores = np.asarray(calib_scores)
    w_test = np.asarray(w_test)
    w_calib = np.asarray(w_calib)
    m = len(test_scores)
    if seed is None:
        # Draw entropy from OS to seed the generator explicitly for lint compliance.
        seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)

    # Step 1: weighted conformal p-values using package utility
    p_vals = calculate_weighted_p_val(test_scores, calib_scores, w_test, w_calib)

    # Precompute constants
    sum_calib_weight = np.sum(w_calib)

    # Step 2: compute R_j^{(0)} sizes and thresholds s_j
    r_sizes = np.zeros(m, dtype=float)
    bh_thresholds = q * (np.arange(1, m + 1) / m)
    calib_mass_below = _calib_weight_mass_below(calib_scores, w_calib, test_scores)
    scratch = np.empty(m, dtype=float)
    logger = get_logger("utils.stat.weighted_fdr")
    j_iterator = (
        tqdm(
            range(m),
            desc="Weighted FDR Control",
        )
        if logger.isEnabledFor(logging.INFO)
        else range(m)
    )
    for j in j_iterator:
        r_sizes[j] = _compute_rejection_set_size_for_instance(
            j,
            test_scores,
            w_test,
            sum_calib_weight,
            bh_thresholds,
            calib_mass_below,
            scratch,
        )

    # Compute thresholds s_j = q * |R_j^{(0)}| / m
    thresholds = q * r_sizes / m

    # Step 3: first selection set R^{(1)}
    first_sel_idx = np.flatnonzero(p_vals <= thresholds)

    # If no points selected, return early with empty boolean mask
    if len(first_sel_idx) == 0:
        final_sel_mask = np.zeros(m, dtype=bool)
        return final_sel_mask

    # Step 4: pruning
    # For pruning, we need |R_j^{(0)}| for each j in first_sel_idx
    sizes_sel = r_sizes[first_sel_idx]
    if pruning == Pruning.HETEROGENEOUS:
        final_sel_idx = _prune_heterogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == Pruning.HOMOGENEOUS:
        final_sel_idx = _prune_homogeneous(first_sel_idx, sizes_sel, rng)
    elif pruning == pruning.DETERMINISTIC:
        final_sel_idx = _prune_deterministic(first_sel_idx, sizes_sel)
    else:
        raise ValueError(
            f"Unknown pruning method '{pruning}'. "
            f"Use 'heterogeneous', 'homogeneous' or 'deterministic'."
        )

    # Convert indices to boolean mask
    final_sel_mask = np.zeros(m, dtype=bool)
    final_sel_mask[final_sel_idx] = True

    return final_sel_mask
