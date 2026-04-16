"""Internal helpers for integrative conformal detection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split

from nonconform._internal import ScorePolarity, set_params
from nonconform.adapters import (
    adapt,
    resolve_implicit_score_polarity,
    resolve_score_polarity,
)
from nonconform.integrative.models import IntegrativeModel

EPSILON = 1e-12


def _clone_estimator(estimator: Any) -> Any:
    """Clone an estimator with sklearn-first fallback to deepcopy."""
    try:
        return clone(estimator)
    except Exception:
        return deepcopy(estimator)


def _as_numpy_with_index(
    x: pd.DataFrame | pd.Series | np.ndarray,
) -> tuple[np.ndarray, pd.Index | None]:
    """Return numpy view of input and optional pandas index."""
    if isinstance(x, pd.Series):
        return x.to_numpy(copy=False).reshape(-1, 1), x.index
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(copy=False), x.index
    return np.asarray(x), None


def _ensure_2d(name: str, x: np.ndarray) -> np.ndarray:
    """Validate and normalize tabular arrays."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if len(arr) == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return arr


def self_inclusive_ranks(scores: np.ndarray) -> np.ndarray:
    """Return self-inclusive conformal p-values for a score set.

    Higher scores are considered more conforming.
    """
    values = np.asarray(scores, dtype=float).ravel()
    if len(values) == 0:
        return np.array([], dtype=float)
    sorted_values = np.sort(values)
    return np.searchsorted(sorted_values, values, side="right") / len(values)


def conformalize_against_calibration(
    scores: np.ndarray,
    calibration_scores: np.ndarray,
) -> np.ndarray:
    """Return conformal p-values against an external calibration set."""
    score_arr = np.asarray(scores, dtype=float).ravel()
    calib_arr = np.asarray(calibration_scores, dtype=float).ravel()
    sorted_cal = np.sort(calib_arr)
    return (1 + np.searchsorted(sorted_cal, score_arr, side="right")) / (
        1 + len(sorted_cal)
    )


def bh_rejection_count(p_values: np.ndarray, alpha: float) -> int:
    """Return the size of the BH rejection set."""
    adjusted = false_discovery_control(np.asarray(p_values, dtype=float), method="bh")
    return int(np.sum(adjusted <= alpha))


def prune_integrative_selection(
    selected_indices: np.ndarray,
    r_tilde: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the paper's pruning rule for split conditional FDR."""
    if len(selected_indices) == 0:
        return np.array([], dtype=int)

    eps = rng.uniform(size=len(selected_indices))
    selected_r = r_tilde[selected_indices].astype(float)
    selected_r = np.maximum(selected_r, 1.0)

    r_final = 0
    for r in range(len(selected_indices), 0, -1):
        if np.sum(eps <= (r / selected_r)) >= r:
            r_final = r
            break

    if r_final == 0:
        return np.array([], dtype=int)
    keep = eps <= (r_final / selected_r)
    return np.sort(selected_indices[keep])[:r_final]


def _resolve_one_class_polarity(
    spec: IntegrativeModel,
    estimator: Any,
) -> ScorePolarity:
    """Resolve one-class score polarity."""
    if spec.score_polarity is None:
        return resolve_implicit_score_polarity(estimator)
    return resolve_score_polarity(estimator, spec.score_polarity)


def _resolve_binary_outlier_label(
    inlier_label: int | str | bool,
) -> int | str | bool:
    """Return a binary outlier label that does not collide with the inlier label."""
    if isinstance(inlier_label, (bool, np.bool_)):
        return not bool(inlier_label)
    if isinstance(inlier_label, (int, np.integer)):
        return 1 if int(inlier_label) == 0 else 0

    candidate = "__outlier__"
    suffix = 0
    while candidate == inlier_label:
        suffix += 1
        candidate = f"__outlier__{suffix}"
    return candidate


@dataclass(slots=True)
class FittedModel:
    """Internal fitted model wrapper."""

    spec: IntegrativeModel
    estimator: Any
    name: str
    reference: str | None
    polarity: ScorePolarity | None = None
    score_source: str | None = None
    inlier_label: int | str | bool | None = None

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return conformity scores."""
        if self.spec.kind == "one_class":
            raw = np.asarray(self.estimator.decision_function(x), dtype=float).ravel()
            if self.polarity is ScorePolarity.HIGHER_IS_NORMAL:
                return raw
            return -raw

        if self.score_source == "predict_proba":
            proba = np.asarray(self.estimator.predict_proba(x), dtype=float)
            class_idx = int(
                np.where(self.estimator.classes_ == self.inlier_label)[0][0]
            )
            return proba[:, class_idx].ravel()

        decision = np.asarray(self.estimator.decision_function(x), dtype=float)
        if decision.ndim == 2:
            class_idx = int(
                np.where(self.estimator.classes_ == self.inlier_label)[0][0]
            )
            return decision[:, class_idx].ravel()

        positive_class = self.estimator.classes_[1]
        if positive_class == self.inlier_label:
            return decision.ravel()
        return -decision.ravel()


def fit_integrative_model(
    spec: IntegrativeModel,
    *,
    x_in_train: np.ndarray,
    x_out_train: np.ndarray,
    seed: int | None,
) -> FittedModel:
    """Fit an integrative model specification."""
    estimator = _clone_estimator(spec.estimator)

    if spec.kind == "one_class":
        adapted = adapt(estimator)
        polarity = _resolve_one_class_polarity(spec, adapted)
        configured = set_params(deepcopy(adapted), seed)
        if spec.reference == "inlier":
            configured.fit(x_in_train)
        else:
            configured.fit(x_out_train)
        name = spec.name or (
            f"one_class:{spec.reference}:{type(spec.estimator).__name__}"
        )
        return FittedModel(
            spec=spec,
            estimator=configured,
            name=name,
            reference=spec.reference,
            polarity=polarity,
        )

    configured = set_params(estimator, seed)
    y_in = np.repeat(spec.inlier_label, len(x_in_train))
    out_label = _resolve_binary_outlier_label(spec.inlier_label)
    y_out = np.repeat(out_label, len(x_out_train))
    x_train = np.vstack([x_in_train, x_out_train])
    y_train = np.concatenate([y_in, y_out])
    configured.fit(x_train, y_train)

    score_source = spec.score_source
    if score_source == "auto":
        if hasattr(configured, "predict_proba"):
            score_source = "predict_proba"
        elif hasattr(configured, "decision_function"):
            score_source = "decision_function"
        else:
            raise ValueError(
                f"Binary estimator {type(spec.estimator).__name__} must implement "
                "predict_proba or decision_function."
            )
    elif not hasattr(configured, score_source):
        raise ValueError(
            f"Binary estimator {type(spec.estimator).__name__} does not implement "
            f"{score_source}."
        )

    if (
        not hasattr(configured, "classes_")
        or spec.inlier_label not in configured.classes_
    ):
        raise ValueError(
            "Binary estimator classes_ do not contain the configured inlier_label."
        )

    name = spec.name or f"binary:{type(spec.estimator).__name__}"
    return FittedModel(
        spec=spec,
        estimator=configured,
        name=name,
        reference=None,
        score_source=score_source,
        inlier_label=spec.inlier_label,
    )


@dataclass(slots=True)
class SplitState:
    """Cached state for the split integrative strategy."""

    x_in_train: np.ndarray
    x_in_calib: np.ndarray
    x_out_train: np.ndarray
    x_out_calib: np.ndarray
    u0_models: list[FittedModel]
    u1_models: list[FittedModel]
    u0_inlier_calib_scores: np.ndarray
    u0_outlier_calib_scores: np.ndarray
    u1_inlier_calib_scores: np.ndarray
    u1_outlier_calib_scores: np.ndarray


def build_split_state(
    *,
    models: list[IntegrativeModel],
    x_inliers: np.ndarray,
    x_outliers: np.ndarray,
    n_calib_in: int,
    n_calib_out: int,
    seed: int | None,
) -> SplitState:
    """Fit all split models and cache calibration score matrices."""
    in_idx = np.arange(len(x_inliers))
    out_idx = np.arange(len(x_outliers))
    in_train_idx, in_calib_idx = train_test_split(
        in_idx,
        test_size=n_calib_in,
        shuffle=True,
        random_state=seed,
    )
    out_train_idx, out_calib_idx = train_test_split(
        out_idx,
        test_size=n_calib_out,
        shuffle=True,
        random_state=seed,
    )

    x_in_train = x_inliers[in_train_idx]
    x_in_calib = x_inliers[in_calib_idx]
    x_out_train = x_outliers[out_train_idx]
    x_out_calib = x_outliers[out_calib_idx]

    u0_specs = [m for m in models if m.kind == "binary" or m.reference == "inlier"]
    u1_specs = [m for m in models if m.kind == "binary" or m.reference == "outlier"]
    if not u0_specs:
        raise ValueError(
            "At least one inlier-side model is required. Add an IntegrativeModel "
            "with reference='inlier' or a binary model."
        )
    if not u1_specs:
        raise ValueError(
            "At least one outlier-side model is required. Add an IntegrativeModel "
            "with reference='outlier' or a binary model."
        )

    u0_models = [
        fit_integrative_model(
            spec,
            x_in_train=x_in_train,
            x_out_train=x_out_train,
            seed=seed,
        )
        for spec in u0_specs
    ]
    u1_models = [
        fit_integrative_model(
            spec,
            x_in_train=x_in_train,
            x_out_train=x_out_train,
            seed=seed,
        )
        for spec in u1_specs
    ]

    return SplitState(
        x_in_train=x_in_train,
        x_in_calib=x_in_calib,
        x_out_train=x_out_train,
        x_out_calib=x_out_calib,
        u0_models=u0_models,
        u1_models=u1_models,
        u0_inlier_calib_scores=np.vstack([m.score(x_in_calib) for m in u0_models]),
        u0_outlier_calib_scores=np.vstack([m.score(x_out_calib) for m in u0_models]),
        u1_inlier_calib_scores=np.vstack([m.score(x_in_calib) for m in u1_models]),
        u1_outlier_calib_scores=np.vstack([m.score(x_out_calib) for m in u1_models]),
    )


def _signed_scores(scores: np.ndarray, sign: int) -> np.ndarray:
    """Apply a sign flip to score arrays."""
    return scores if sign > 0 else -scores


def _choose_split_candidate(
    inlier_group_scores: np.ndarray,
    outlier_group_scores: np.ndarray,
    *,
    maximize_inliers: bool,
) -> tuple[int, int]:
    """Choose best candidate/sign for split model selection."""
    best_idx = 0
    best_sign = 1
    best_utility = -np.inf
    for idx in range(inlier_group_scores.shape[0]):
        for sign in (1, -1):
            inlier_scores = _signed_scores(inlier_group_scores[idx], sign)
            outlier_scores = _signed_scores(outlier_group_scores[idx], sign)
            if maximize_inliers:
                utility = np.median(inlier_scores) - np.median(outlier_scores)
            else:
                utility = np.median(outlier_scores) - np.median(inlier_scores)
            if utility > best_utility:
                best_utility = float(utility)
                best_idx = idx
                best_sign = sign
    return best_idx, best_sign


def compute_split_single(
    state: SplitState,
    *,
    u0_test_scores: np.ndarray,
    u1_test_scores: np.ndarray,
    target_idx: int,
    extra_inlier_test_indices: tuple[int, ...] = (),
) -> tuple[
    float,
    float,
    np.ndarray,
    dict[str, Any],
]:
    """Compute one split integrative p-value from cached score matrices."""
    if target_idx in extra_inlier_test_indices:
        raise ValueError(
            "target_idx must not be included in extra_inlier_test_indices."
        )

    extra_indices = list(extra_inlier_test_indices)

    def _gather(matrix: np.ndarray) -> np.ndarray:
        pieces = [matrix[:, : state.x_in_calib.shape[0]]]
        if extra_indices:
            pieces.append(
                matrix[:, state.x_in_calib.shape[0] + np.array(extra_indices)]
            )
        pieces.append(matrix[:, state.x_in_calib.shape[0] + np.array([target_idx])])
        return np.concatenate(pieces, axis=1)

    u0_full = np.concatenate([state.u0_inlier_calib_scores, u0_test_scores], axis=1)
    u1_full = np.concatenate([state.u1_inlier_calib_scores, u1_test_scores], axis=1)

    u0_group = _gather(u0_full)
    u1_group = _gather(u1_full)

    u0_idx, u0_sign = _choose_split_candidate(
        u0_group,
        state.u0_outlier_calib_scores,
        maximize_inliers=True,
    )
    u1_idx, u1_sign = _choose_split_candidate(
        u1_group,
        state.u1_outlier_calib_scores,
        maximize_inliers=False,
    )

    selected_u0 = _signed_scores(u0_group[u0_idx], u0_sign)
    selected_u1_input = _signed_scores(u1_group[u1_idx], u1_sign)
    selected_u1_out = _signed_scores(state.u1_outlier_calib_scores[u1_idx], u1_sign)

    prelim_u0 = self_inclusive_ranks(selected_u0)
    prelim_u1 = conformalize_against_calibration(selected_u1_input, selected_u1_out)
    ratio = prelim_u0 / np.maximum(prelim_u1, EPSILON)
    final_p = self_inclusive_ranks(ratio)[-1]

    metadata = {
        "u0_model_index": u0_idx,
        "u0_sign": u0_sign,
        "u0_model_name": state.u0_models[u0_idx].name,
        "u1_model_index": u1_idx,
        "u1_sign": u1_sign,
        "u1_model_name": state.u1_models[u1_idx].name,
        "u0_preliminary": prelim_u0[-1],
        "u1_preliminary": prelim_u1[-1],
    }
    return final_p, ratio[-1], ratio[:-1], metadata


def compute_split_batch(
    state: SplitState,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute split integrative p-values for a batch."""
    u0_test_scores = np.vstack([m.score(x_test) for m in state.u0_models])
    u1_test_scores = np.vstack([m.score(x_test) for m in state.u1_models])

    p_values = np.empty(len(x_test), dtype=float)
    ratios = np.empty(len(x_test), dtype=float)
    ratio_cal = np.empty((len(x_test), state.x_in_calib.shape[0]), dtype=float)
    selected_u0_models = np.empty(len(x_test), dtype=int)
    selected_u1_models = np.empty(len(x_test), dtype=int)
    selected_u0_signs = np.empty(len(x_test), dtype=int)
    selected_u1_signs = np.empty(len(x_test), dtype=int)
    prelim_u0 = np.empty(len(x_test), dtype=float)
    prelim_u1 = np.empty(len(x_test), dtype=float)

    for idx in range(len(x_test)):
        p_val, ratio, calib_ratios, meta = compute_split_single(
            state,
            u0_test_scores=u0_test_scores,
            u1_test_scores=u1_test_scores,
            target_idx=idx,
        )
        p_values[idx] = p_val
        ratios[idx] = ratio
        ratio_cal[idx] = calib_ratios
        selected_u0_models[idx] = meta["u0_model_index"]
        selected_u1_models[idx] = meta["u1_model_index"]
        selected_u0_signs[idx] = meta["u0_sign"]
        selected_u1_signs[idx] = meta["u1_sign"]
        prelim_u0[idx] = meta["u0_preliminary"]
        prelim_u1[idx] = meta["u1_preliminary"]

    metadata = {
        "strategy": "split",
        "u0_test_scores": u0_test_scores.copy(),
        "u1_test_scores": u1_test_scores.copy(),
        "u0_inlier_calib_scores": state.u0_inlier_calib_scores.copy(),
        "u0_outlier_calib_scores": state.u0_outlier_calib_scores.copy(),
        "u1_inlier_calib_scores": state.u1_inlier_calib_scores.copy(),
        "u1_outlier_calib_scores": state.u1_outlier_calib_scores.copy(),
        "u0_model_names": [m.name for m in state.u0_models],
        "u1_model_names": [m.name for m in state.u1_models],
        "selected_u0_models": selected_u0_models,
        "selected_u1_models": selected_u1_models,
        "selected_u0_signs": selected_u0_signs,
        "selected_u1_signs": selected_u1_signs,
        "preliminary_u0": prelim_u0,
        "preliminary_u1": prelim_u1,
    }
    return p_values, ratios, ratio_cal, metadata


@dataclass(slots=True)
class TCVPlusState:
    """Cached state for transductive CV+."""

    x_inliers: np.ndarray
    x_outliers: np.ndarray
    u0_specs: list[IntegrativeModel]
    u1_specs: list[IntegrativeModel]
    outlier_fold_indices: np.ndarray
    u1_fold_models: list[list[FittedModel]]
    u1_holdout_scores: list[np.ndarray]
    strategy_k_in: int
    strategy_k_out: int
    seed: int | None
    shuffle: bool


def _make_kfold_assignments(
    n_samples: int,
    k: int,
    *,
    seed: int | None,
    shuffle: bool,
) -> np.ndarray:
    """Return fold assignments for all samples."""
    assignments = np.empty(n_samples, dtype=int)
    splitter = KFold(
        n_splits=k, shuffle=shuffle, random_state=seed if shuffle else None
    )
    dummy = np.zeros((n_samples, 1))
    for fold_idx, (_, holdout_idx) in enumerate(splitter.split(dummy), start=1):
        assignments[holdout_idx] = fold_idx
    return assignments


def build_tcv_plus_state(
    *,
    models: list[IntegrativeModel],
    x_inliers: np.ndarray,
    x_outliers: np.ndarray,
    k_in: int,
    k_out: int,
    seed: int | None,
    shuffle: bool,
) -> TCVPlusState:
    """Prepare cached outlier-fold models for TCV+."""
    u0_specs = [m for m in models if m.kind == "binary" or m.reference == "inlier"]
    u1_specs = [m for m in models if m.kind == "binary" or m.reference == "outlier"]
    if not u0_specs or not u1_specs:
        raise ValueError(
            "TCV+ requires at least one inlier-side and one outlier-side model."
        )

    out_assignments = _make_kfold_assignments(
        len(x_outliers),
        k_out,
        seed=seed,
        shuffle=shuffle,
    )

    u1_fold_models: list[list[FittedModel]] = []
    u1_holdout_scores: list[np.ndarray] = []
    for spec in u1_specs:
        per_fold_models: list[FittedModel] = []
        holdout_scores = np.empty(len(x_outliers), dtype=float)
        for fold_idx in range(1, k_out + 1):
            train_mask = out_assignments != fold_idx
            fitted = fit_integrative_model(
                spec,
                x_in_train=x_inliers,
                x_out_train=x_outliers[train_mask],
                seed=seed,
            )
            per_fold_models.append(fitted)
            holdout_mask = out_assignments == fold_idx
            holdout_scores[holdout_mask] = fitted.score(x_outliers[holdout_mask])
        u1_fold_models.append(per_fold_models)
        u1_holdout_scores.append(holdout_scores)

    return TCVPlusState(
        x_inliers=x_inliers,
        x_outliers=x_outliers,
        u0_specs=u0_specs,
        u1_specs=u1_specs,
        outlier_fold_indices=out_assignments,
        u1_fold_models=u1_fold_models,
        u1_holdout_scores=u1_holdout_scores,
        strategy_k_in=k_in,
        strategy_k_out=k_out,
        seed=seed,
        shuffle=shuffle,
    )


def _score_external_points_average(
    models: list[FittedModel], x: np.ndarray
) -> np.ndarray:
    """Score external points by averaging across fold models."""
    fold_scores = np.vstack([model.score(x) for model in models])
    return np.mean(fold_scores, axis=0)


def _compute_tcv_u1_values(
    models: list[FittedModel],
    holdout_scores: np.ndarray,
    outlier_fold_indices: np.ndarray,
    x_points: np.ndarray,
    sign: int,
) -> np.ndarray:
    """Compute TCV+ preliminary u1 values for arbitrary points."""
    point_scores = np.vstack([sign * model.score(x_points) for model in models])
    aligned_scores = point_scores[outlier_fold_indices - 1]
    comparisons = (sign * holdout_scores)[:, None] <= aligned_scores
    return (1 + np.sum(comparisons, axis=0)) / (1 + len(holdout_scores))


def _fit_tcv_u0_fold_models(
    spec: IntegrativeModel,
    *,
    x_inliers: np.ndarray,
    x_outliers: np.ndarray,
    fold_assignments: np.ndarray,
    k_in: int,
    seed: int | None,
) -> tuple[list[FittedModel], np.ndarray]:
    """Fit inlier-side fold models and return held-out inlier scores."""
    models: list[FittedModel] = []
    held_out = np.empty(len(x_inliers), dtype=float)
    for fold_idx in range(1, k_in + 1):
        train_mask = fold_assignments != fold_idx
        fitted = fit_integrative_model(
            spec,
            x_in_train=x_inliers[train_mask],
            x_out_train=x_outliers,
            seed=seed,
        )
        models.append(fitted)
        holdout_mask = fold_assignments == fold_idx
        held_out[holdout_mask] = fitted.score(x_inliers[holdout_mask])
    return models, held_out


def _choose_tcv_u0_candidate(
    candidate_inlier_scores: list[np.ndarray],
    candidate_outlier_scores: list[np.ndarray],
) -> tuple[int, int]:
    """Choose TCV+ u0 candidate/sign using held-out conformity scores."""
    best_idx = 0
    best_sign = 1
    best_utility = -np.inf
    for idx, (inlier_scores, outlier_scores) in enumerate(
        zip(candidate_inlier_scores, candidate_outlier_scores, strict=True)
    ):
        for sign in (1, -1):
            utility = np.median(sign * inlier_scores) - np.median(sign * outlier_scores)
            if utility > best_utility:
                best_utility = float(utility)
                best_idx = idx
                best_sign = sign
    return best_idx, best_sign


def _choose_tcv_u1_candidate(
    candidate_inlier_scores: list[np.ndarray],
    candidate_outlier_scores: list[np.ndarray],
) -> tuple[int, int]:
    """Choose TCV+ u1 candidate/sign using held-out conformity scores."""
    best_idx = 0
    best_sign = 1
    best_utility = -np.inf
    for idx, (inlier_scores, outlier_scores) in enumerate(
        zip(candidate_inlier_scores, candidate_outlier_scores, strict=True)
    ):
        for sign in (1, -1):
            utility = np.median(sign * outlier_scores) - np.median(sign * inlier_scores)
            if utility > best_utility:
                best_utility = float(utility)
                best_idx = idx
                best_sign = sign
    return best_idx, best_sign


def compute_tcv_plus_batch(
    state: TCVPlusState,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute TCV+ integrative p-values for a batch."""
    p_values = np.empty(len(x_test), dtype=float)
    ratios = np.empty(len(x_test), dtype=float)
    ratio_cal = np.empty((len(x_test), len(state.x_inliers)), dtype=float)
    selected_u0_models = np.empty(len(x_test), dtype=int)
    selected_u1_models = np.empty(len(x_test), dtype=int)
    selected_u0_signs = np.empty(len(x_test), dtype=int)
    selected_u1_signs = np.empty(len(x_test), dtype=int)
    augmented_fold_assignments = _make_kfold_assignments(
        len(state.x_inliers) + 1,
        state.strategy_k_in,
        seed=state.seed,
        shuffle=state.shuffle,
    )
    inlier_assignments = augmented_fold_assignments[:-1]
    target_fold = int(augmented_fold_assignments[-1])

    for idx, x_target in enumerate(x_test):
        x_single = x_target.reshape(1, -1)

        u0_fold_models_all: list[list[FittedModel]] = []
        u0_inlier_scores_all: list[np.ndarray] = []
        u0_outlier_scores_all: list[np.ndarray] = []
        for spec in state.u0_specs:
            fold_models, held_out_scores = _fit_tcv_u0_fold_models(
                spec,
                x_inliers=state.x_inliers,
                x_outliers=state.x_outliers,
                fold_assignments=inlier_assignments,
                k_in=state.strategy_k_in,
                seed=state.seed,
            )
            target_score = fold_models[target_fold - 1].score(x_single)
            outlier_scores = _score_external_points_average(
                fold_models, state.x_outliers
            )
            u0_fold_models_all.append(fold_models)
            u0_inlier_scores_all.append(np.concatenate([held_out_scores, target_score]))
            u0_outlier_scores_all.append(outlier_scores)

        u1_inlier_scores_all: list[np.ndarray] = []
        u1_outlier_scores_all: list[np.ndarray] = []
        for models, holdout_scores in zip(
            state.u1_fold_models, state.u1_holdout_scores, strict=True
        ):
            inlier_scores = _score_external_points_average(models, state.x_inliers)
            target_scores = _score_external_points_average(models, x_single)
            outlier_scores = holdout_scores.copy()
            u1_inlier_scores_all.append(np.concatenate([inlier_scores, target_scores]))
            u1_outlier_scores_all.append(outlier_scores)

        u0_idx, u0_sign = _choose_tcv_u0_candidate(
            u0_inlier_scores_all,
            u0_outlier_scores_all,
        )
        u1_idx, u1_sign = _choose_tcv_u1_candidate(
            u1_inlier_scores_all,
            u1_outlier_scores_all,
        )

        selected_u0_scores = u0_sign * u0_inlier_scores_all[u0_idx]
        prelim_u0 = self_inclusive_ranks(selected_u0_scores)

        u1_models = state.u1_fold_models[u1_idx]
        inlier_plus_target = np.vstack([state.x_inliers, x_single])
        prelim_u1 = _compute_tcv_u1_values(
            u1_models,
            state.u1_holdout_scores[u1_idx],
            state.outlier_fold_indices,
            inlier_plus_target,
            u1_sign,
        )

        ratio_all = prelim_u0 / np.maximum(prelim_u1, EPSILON)
        p_values[idx] = self_inclusive_ranks(ratio_all)[-1]
        ratios[idx] = ratio_all[-1]
        ratio_cal[idx] = ratio_all[:-1]
        selected_u0_models[idx] = u0_idx
        selected_u1_models[idx] = u1_idx
        selected_u0_signs[idx] = u0_sign
        selected_u1_signs[idx] = u1_sign

    metadata = {
        "strategy": "tcv_plus",
        "selected_u0_models": selected_u0_models,
        "selected_u1_models": selected_u1_models,
        "selected_u0_signs": selected_u0_signs,
        "selected_u1_signs": selected_u1_signs,
        "u0_model_names": [
            spec.name or f"{spec.kind}:{spec.reference}:{type(spec.estimator).__name__}"
            for spec in state.u0_specs
        ],
        "u1_model_names": [
            spec.name or f"{spec.kind}:{spec.reference}:{type(spec.estimator).__name__}"
            for spec in state.u1_specs
        ],
        "k_in": state.strategy_k_in,
        "k_out": state.strategy_k_out,
    }
    return p_values, ratios, ratio_cal, metadata


__all__ = [
    "EPSILON",
    "SplitState",
    "TCVPlusState",
    "bh_rejection_count",
    "build_split_state",
    "build_tcv_plus_state",
    "compute_split_batch",
    "compute_split_single",
    "compute_tcv_plus_batch",
    "conformalize_against_calibration",
    "prune_integrative_selection",
    "self_inclusive_ranks",
]
