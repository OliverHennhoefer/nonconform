"""False Discovery Rate control for conformal prediction.

This module implements Weighted Conformalized Selection (WCS) for FDR control
under covariate shift. For standard BH/BY procedures, use
scipy.stats.false_discovery_control.
"""

from __future__ import annotations

import numpy as np

from nonconform.utils.stat.statistical import calculate_weighted_p_val


def _bh_rejection_indices(p_values: np.ndarray, q: float) -> np.ndarray:
    """Return indices of BH rejection set for given p-values.

    This helper mimics the Benjamini-Hochberg procedure: sort p-values,
    find the largest k such that p_(k) ≤ q*k/m, and return the first k
    indices in the sorted order.  If no p-value meets the criterion,
    returns an empty array.
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


def weighted_false_discovery_control(
    test_scores: np.ndarray,
    calib_scores: np.ndarray,
    w_test: np.ndarray,
    w_calib: np.ndarray,
    q: float,
    rand: str = "dtm",
    seed: int | None = None,
) -> np.ndarray:
    """Perform Weighted Conformalized Selection (WCS).

    Parameters
    ----------
    test_scores : np.ndarray
        Non-conformity scores for the test data (length m).
    calib_scores : np.ndarray
        Non-conformity scores for the calibration data (length n).
    w_test : np.ndarray
        Importance weights for the test data (length m).
    w_calib : np.ndarray
        Importance weights for the calibration data (length n).
    q : float
        Target false discovery rate (0 < q < 1).
    rand : {"hete", "homo", "dtm"}, optional
        Pruning method.  ``'hete'`` (heterogeneous pruning) uses
        independent random variables l_j; ``'homo'`` (homogeneous
        pruning) uses a single random variable l shared across
        candidates; ``'dtm'`` (deterministic) performs deterministic
        pruning based on |R_j^{(0)}|.  Defaults to ``'dtm'``.
    seed : int | None, optional
        Random seed for reproducibility.
        Defaults to ``None`` (non-deterministic).

    Returns:
    -------
    np.ndarray
        Boolean mask of test points retained after pruning (final selection).
        For deterministic pruning (``'dtm'``), this may coincide with the
        first selection step.

    Notes:
    -----
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
    ----------
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
    rng = np.random.default_rng(seed)

    # Step 1: weighted conformal p-values using package utility
    p_vals = calculate_weighted_p_val(test_scores, calib_scores, w_test, w_calib)

    # Precompute constants
    sum_calib_weight = np.sum(w_calib)

    # Step 2: compute R_j^{(0)} sizes and thresholds s_j
    r_sizes = np.zeros(m, dtype=float)
    for j in range(m):
        # Compute auxiliary p-values for test instance j
        p_aux = np.zeros(m, dtype=float)
        for ell in range(m):
            if ell == j:
                # p_j^{(j)} is ignored (set to 0 so BH includes j)
                p_aux[ell] = 0.0
                continue
            v_ell = test_scores[ell]
            # Weighted count of calibration scores strictly less than v_ell
            w_less = np.sum(w_calib[calib_scores < v_ell])
            # Weighted indicator for test_j's score strictly less than v_ell
            add = w_test[j] if test_scores[j] < v_ell else 0.0
            numerator = w_less + add
            denominator = sum_calib_weight + w_test[j]
            p_aux[ell] = numerator / denominator
        # Apply BH to auxiliary p-values
        rej_idx = _bh_rejection_indices(p_aux, q)
        r_sizes[j] = len(rej_idx)

    # Compute thresholds s_j = q * |R_j^{(0)}| / m
    thresholds = q * r_sizes / m

    # Step 3: first selection set R^{(1)}
    first_sel_idx = np.nonzero(p_vals <= thresholds)[0]

    # If no points selected, return early with empty boolean mask
    if len(first_sel_idx) == 0:
        final_sel_mask = np.zeros(m, dtype=bool)
        return final_sel_mask

    # Step 4: pruning
    # For pruning, we need |R_j^{(0)}| for each j in first_sel_idx
    sizes_sel = r_sizes[first_sel_idx]
    # Initialize empty final selection
    final_sel_idx: np.ndarray
    if rand == "hete":
        # Heterogeneous pruning: independent ξ_j ∈ [0,1] for each j
        xi = rng.uniform(size=len(first_sel_idx))
        # Order candidates by increasing ξ_j * |R_j^{(0)}|
        order = np.argsort(xi * sizes_sel)
        # Determine the largest r such that at least r points have
        # rank ≤ r (self-consistency condition)
        # We select in that order until count ≥ r
        selected = []
        for r, idx in enumerate(order, start=1):
            selected.append(first_sel_idx[idx])
            if len(selected) < r:
                continue
        final_sel_idx = np.sort(np.array(selected, dtype=int))
    elif rand == "homo":
        # Homogeneous pruning: shared ξ ∈ [0,1]
        xi = rng.uniform()
        order = np.argsort(xi * sizes_sel)
        selected = []
        for r, idx in enumerate(order, start=1):
            selected.append(first_sel_idx[idx])
            if len(selected) < r:
                continue
        final_sel_idx = np.sort(np.array(selected, dtype=int))
    elif rand == "dtm":
        # Deterministic pruning: order by |R_j^{(0)}|
        order = np.argsort(sizes_sel)
        selected = []
        for r, idx in enumerate(order, start=1):
            selected.append(first_sel_idx[idx])
            if len(selected) < r:
                continue
        final_sel_idx = np.sort(np.array(selected, dtype=int))
    else:
        raise ValueError(
            f"Unknown pruning method '{rand}'. Use 'hete', 'homo' or 'dtm'."
        )

    # Convert indices to boolean mask
    final_sel_mask = np.zeros(m, dtype=bool)
    final_sel_mask[final_sel_idx] = True

    return final_sel_mask
