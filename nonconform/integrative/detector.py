"""Public detector API for integrative conformal detection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from nonconform.integrative._core import (
    SplitState,
    TCVPlusState,
    _as_numpy_with_index,
    _ensure_2d,
    build_split_state,
    build_tcv_plus_state,
    compute_split_batch,
    compute_tcv_plus_batch,
)
from nonconform.integrative.models import IntegrativeModel
from nonconform.integrative.strategies import IntegrativeSplit, TransductiveCVPlus
from nonconform.structures import ConformalResult


def _snapshot_param(value: Any) -> Any:
    """Return an immutable constructor-parameter snapshot."""
    return deepcopy(value)


def _normalize_models(
    models: IntegrativeModel | list[IntegrativeModel],
) -> list[IntegrativeModel]:
    """Normalize the models input into a list."""
    if isinstance(models, IntegrativeModel):
        return [models]
    return list(models)


def _get_component_params(component: Any) -> dict[str, Any]:
    """Return nested parameters for sklearn-style inspection."""
    if hasattr(component, "get_params"):
        try:
            params = component.get_params(deep=True)
        except TypeError:
            params = component.get_params()
        return {key: _snapshot_param(value) for key, value in params.items()}

    if is_dataclass(component):
        return {
            field.name: _snapshot_param(getattr(component, field.name))
            for field in fields(component)
        }

    return {}


class IntegrativeConformalDetector:
    """Integrative conformal detector with labeled inliers and outliers.

    This detector implements the split and TCV+ integrative conformal workflows
    described in the Liang-Sesia-Sun reference paper.

    Args:
        models: Collection of typed model specifications.
        strategy: Integrative strategy instance.
        seed: Optional random seed.
        verbose: Reserved for parity with the main detector API.
    """

    def __init__(
        self,
        *,
        models: IntegrativeModel | list[IntegrativeModel],
        strategy: IntegrativeSplit | TransductiveCVPlus,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._configure(
            models=models,
            strategy=strategy,
            seed=seed,
            verbose=verbose,
        )

    def _reset_fit_state(self) -> None:
        """Clear all learned state derived from fit()."""
        self._state: SplitState | TCVPlusState | None = None
        self._last_result: ConformalResult | None = None

    def _configure(
        self,
        *,
        models: IntegrativeModel | list[IntegrativeModel],
        strategy: IntegrativeSplit | TransductiveCVPlus,
        seed: int | None,
        verbose: bool,
    ) -> None:
        """Apply constructor parameters and reset learned state."""
        normalized_models = _normalize_models(models)
        if not normalized_models:
            raise ValueError("models must contain at least one IntegrativeModel.")

        self._init_models = _snapshot_param(models)
        self._init_strategy = _snapshot_param(strategy)
        self._init_seed = seed
        self._init_verbose = verbose

        self.models = [deepcopy(model) for model in normalized_models]
        self.strategy = deepcopy(strategy)
        self.seed = seed
        self.verbose = verbose
        self._reset_fit_state()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters following sklearn conventions."""
        params: dict[str, Any] = {
            "models": _snapshot_param(self._init_models),
            "strategy": _snapshot_param(self._init_strategy),
            "seed": self._init_seed,
            "verbose": self._init_verbose,
        }
        if not deep:
            return params

        for key, value in _get_component_params(self.strategy).items():
            params[f"strategy__{key}"] = value

        for idx, model in enumerate(self.models):
            params[f"models__{idx}"] = _snapshot_param(model)
            for key, value in _get_component_params(model).items():
                params[f"models__{idx}__{key}"] = value
                if key != "estimator":
                    continue
                for estimator_key, estimator_value in _get_component_params(
                    value
                ).items():
                    params[f"models__{idx}__estimator__{estimator_key}"] = (
                        estimator_value
                    )

        return params

    def set_params(self, **params: Any) -> Self:
        """Set estimator parameters following sklearn conventions."""
        if not params:
            return self

        updated_params = self.get_params(deep=False)
        strategy_updates: dict[str, Any] = {}
        model_updates: dict[int, dict[str, Any]] = {}

        for key, value in params.items():
            if "__" not in key:
                if key not in updated_params:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator "
                        f"{type(self).__name__}."
                    )
                updated_params[key] = value
                continue

            component_name, nested_key = key.split("__", 1)
            if component_name == "strategy":
                strategy_updates[nested_key] = value
                continue

            if component_name == "models":
                index_str, separator, remainder = nested_key.partition("__")
                if not index_str.isdigit():
                    raise ValueError(f"Invalid parameter {key!r}.")
                idx = int(index_str)
                if not separator:
                    models = _normalize_models(updated_params["models"])
                    if idx >= len(models):
                        raise ValueError(f"Invalid parameter {key!r}.")
                    models[idx] = value
                    updated_params["models"] = models
                    continue
                model_updates.setdefault(idx, {})[remainder] = value
                continue

            raise ValueError(
                f"Invalid parameter {component_name!r} for estimator "
                f"{type(self).__name__}."
            )

        strategy = deepcopy(updated_params["strategy"])
        for nested_key, value in strategy_updates.items():
            if "__" in nested_key or not hasattr(strategy, nested_key):
                raise ValueError(f"Invalid parameter 'strategy__{nested_key}'.")
            setattr(strategy, nested_key, value)
        updated_params["strategy"] = strategy

        models_param = updated_params["models"]
        models_are_scalar = isinstance(models_param, IntegrativeModel)
        models = [deepcopy(model) for model in _normalize_models(models_param)]
        for idx, updates in model_updates.items():
            if idx >= len(models):
                raise ValueError(f"Invalid parameter 'models__{idx}'.")
            model = models[idx]
            for nested_key, value in updates.items():
                field_name, separator, remainder = nested_key.partition("__")
                if field_name == "estimator":
                    if not separator:
                        model.estimator = value
                        continue
                    if not hasattr(model.estimator, "set_params"):
                        raise ValueError(
                            f"Cannot set nested parameters for 'models__{idx}__"
                            "estimator': component does not implement "
                            "set_params()."
                        )
                    model.estimator.set_params(**{remainder: value})
                    continue

                if separator or not hasattr(model, field_name):
                    raise ValueError(
                        f"Invalid parameter 'models__{idx}__{nested_key}'."
                    )
                setattr(model, field_name, value)

        updated_params["models"] = models[0] if models_are_scalar else models
        self._configure(**updated_params)
        return self

    def __sklearn_clone__(self) -> Self:
        """Return sklearn-compatible unfitted clone from constructor snapshots."""
        params = self.get_params(deep=False)
        cloned_params = {key: _snapshot_param(value) for key, value in params.items()}
        return type(self)(**cloned_params)

    def fit(
        self,
        x_inliers: pd.DataFrame | np.ndarray,
        x_outliers: pd.DataFrame | np.ndarray,
    ) -> Self:
        """Fit the detector on labeled inliers and labeled outliers."""
        x_in = _ensure_2d("x_inliers", np.asarray(x_inliers))
        x_out = _ensure_2d("x_outliers", np.asarray(x_outliers))
        if x_in.shape[1] != x_out.shape[1]:
            raise ValueError(
                "x_inliers and x_outliers must have the same feature count."
            )

        if isinstance(self.strategy, IntegrativeSplit):
            n_calib_in, n_calib_out = self.strategy.resolve_sizes(len(x_in), len(x_out))
            self._state = build_split_state(
                models=self.models,
                x_inliers=x_in,
                x_outliers=x_out,
                n_calib_in=n_calib_in,
                n_calib_out=n_calib_out,
                seed=self.seed,
            )
        else:
            self.strategy.validate_dataset_sizes(len(x_in), len(x_out))
            self._state = build_tcv_plus_state(
                models=self.models,
                x_inliers=x_in,
                x_outliers=x_out,
                k_in=self.strategy.k_in,
                k_out=self.strategy.k_out,
                seed=self.seed,
                shuffle=self.strategy.shuffle,
            )

        self._last_result = None
        return self

    def compute_p_values(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
    ) -> np.ndarray | pd.Series:
        """Return integrative conformal p-values for a test batch."""
        if self._state is None:
            raise NotFittedError(
                "This IntegrativeConformalDetector instance is not fitted yet."
            )
        x_array, index = _as_numpy_with_index(x)
        x_array = _ensure_2d("x", x_array)

        if isinstance(self._state, SplitState):
            (
                p_values,
                ratios,
                ratio_cal,
                metadata,
            ) = compute_split_batch(self._state, x_array)
        else:
            (
                p_values,
                ratios,
                ratio_cal,
                metadata,
            ) = compute_tcv_plus_batch(self._state, x_array)

        self._last_result = ConformalResult(
            p_values=p_values.copy(),
            test_scores=ratios.copy(),
            calib_scores=ratio_cal.copy(),
            metadata={"integrative": metadata},
        )
        if index is not None:
            return pd.Series(p_values, index=index, name="p_value")
        return p_values

    def score_samples(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
    ) -> np.ndarray | pd.Series:
        """Return the integrative ranking statistic ``r = u0 / u1``."""
        p_values = self.compute_p_values(x)
        result = self.last_result
        if result is None or result.test_scores is None:
            raise RuntimeError(
                "Internal error: expected test_scores after compute_p_values()."
            )
        if isinstance(p_values, pd.Series):
            return pd.Series(result.test_scores, index=p_values.index, name="score")
        return result.test_scores.copy()

    def select(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> np.ndarray | pd.Series:
        """Return the split conditional-FDR selection mask."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        x_array, index = _as_numpy_with_index(x)
        x_array = _ensure_2d("x", x_array)
        _ = self.compute_p_values(x_array)

        if isinstance(self.strategy, TransductiveCVPlus):
            raise NotImplementedError(
                "select() is currently implemented only for IntegrativeSplit. "
                "TCV+ conditional-FDR selection is intentionally deferred."
            )

        from nonconform.fdr import integrative_false_discovery_control

        selection_seed = self.seed if seed is None else seed
        selected = integrative_false_discovery_control(
            result=self._last_result,
            alpha=alpha,
            seed=selection_seed,
        )
        if index is not None:
            return pd.Series(selected, index=index, name="selected")
        return selected

    @property
    def last_result(self) -> ConformalResult | None:
        """Return the most recent result bundle."""
        return None if self._last_result is None else self._last_result.copy()

    @property
    def is_fitted(self) -> bool:
        """Return whether fit() has been called successfully."""
        return self._state is not None


__all__ = [
    "IntegrativeConformalDetector",
]
