"""Temporal orchestration utilities for sequential conformal workflows.

This module provides a stateful session API that composes:

- conformal p-value generation via ``ConformalDetector``
- optional online FDR decisions via an ``OnlineFDRController``
- optional exchangeability monitoring via martingales

The goal is to keep statistical core components unchanged while removing manual
loop glue code in streaming/temporal applications.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from nonconform.fdr import (
    OnlineFDRController,
    online_false_discovery_control,
    weighted_false_discovery_control,
)
from nonconform.martingales import BaseMartingale, MartingaleState

if TYPE_CHECKING:
    from nonconform.detector import ConformalDetector


TemporalHook = Callable[["TemporalStepResult"], None]


def _validate_alpha(alpha: float) -> None:
    """Validate FDR target level."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def _require_online_controller(controller: object) -> OnlineFDRController:
    """Validate online controller interface contract."""
    if not hasattr(controller, "test_one") or not callable(controller.test_one):
        raise TypeError(
            "online_controller must implement callable "
            "test_one(p_value: float) -> bool."
        )
    return controller  # type: ignore[return-value]


def _as_1d_float_p_values(name: str, p_values: np.ndarray) -> np.ndarray:
    """Normalize and validate p-values as a finite 1D float array in [0, 1]."""
    try:
        values = np.asarray(p_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {values.shape!r}.")
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    eps = 1e-10
    if np.any((values < -eps) | (values > 1.0 + eps)):
        raise ValueError(f"{name} must be within [0, 1].")
    return np.clip(values, 0.0, 1.0)


@dataclass(slots=True)
class TemporalStepResult:
    """Result bundle returned by ``TemporalSession.step(...)``."""

    p_values: np.ndarray | pd.Series
    online_decisions: np.ndarray | pd.Series | None
    martingale_states: list[MartingaleState] | None
    triggered_alarms: tuple[str, ...]


class TemporalSession:
    """Stateful orchestrator for sequential conformal inference workflows.

    This class composes:

    - ``ConformalDetector.compute_p_values(...)``
    - optional online FDR decisions via ``online_controller.test_one(...)``
    - optional martingale updates via ``martingale.update_many(...)``

    Optional hooks can be attached for event-driven lifecycle integration.
    """

    def __init__(
        self,
        *,
        detector: ConformalDetector,
        online_controller: OnlineFDRController | None = None,
        martingale: BaseMartingale | None = None,
        on_step_end: TemporalHook | None = None,
        on_alarm: TemporalHook | None = None,
    ) -> None:
        self.detector = detector
        self.online_controller = (
            None
            if online_controller is None
            else _require_online_controller(online_controller)
        )
        self.martingale = martingale
        self.on_step_end = on_step_end
        self.on_alarm = on_alarm
        self._last_batch_decisions: np.ndarray | pd.Series | None = None

    @property
    def last_batch_decisions(self) -> np.ndarray | pd.Series | None:
        """Return side-by-side batch decisions from the most recent step."""
        if self._last_batch_decisions is None:
            return None
        if isinstance(self._last_batch_decisions, pd.Series):
            return self._last_batch_decisions.copy()
        return self._last_batch_decisions.copy()

    def reset(
        self,
        *,
        reset_online_controller: bool = False,
        reset_martingale: bool = True,
    ) -> None:
        """Reset temporal session state.

        Args:
            reset_online_controller: If True and the online controller defines
                ``reset()``, call it.
            reset_martingale: If True and martingale is configured, call
                ``martingale.reset()``.
        """
        self._last_batch_decisions = None
        if reset_martingale and self.martingale is not None:
            self.martingale.reset()
        if (
            reset_online_controller
            and self.online_controller is not None
            and hasattr(self.online_controller, "reset")
            and callable(self.online_controller.reset)  # type: ignore[attr-defined]
        ):
            self.online_controller.reset()  # type: ignore[attr-defined]

    def _collect_triggered_alarms(
        self, states: list[MartingaleState] | None
    ) -> tuple[str, ...]:
        """Collect unique alarm names in first-seen order."""
        if not states:
            return ()
        seen: set[str] = set()
        alarms: list[str] = []
        for state in states:
            for alarm_name in state.triggered_alarms:
                if alarm_name not in seen:
                    seen.add(alarm_name)
                    alarms.append(alarm_name)
        return tuple(alarms)

    def _format_decisions_output(
        self,
        decisions: np.ndarray,
        *,
        index: pd.Index | None,
        name: str,
    ) -> np.ndarray | pd.Series:
        """Return ndarray or pandas Series decisions with preserved index."""
        if index is not None:
            return pd.Series(decisions, index=index, name=name)
        return decisions

    def _compute_batch_decisions(
        self,
        *,
        p_values: np.ndarray,
        index: pd.Index | None,
        alpha: float,
    ) -> np.ndarray | pd.Series:
        """Compute per-step batch decisions from already computed p-values."""
        last_result = self.detector.last_result
        if last_result is None or last_result.p_values is None:
            raise RuntimeError(
                "Internal error: missing detector.last_result after compute_p_values()."
            )

        if (
            last_result.calib_weights is not None
            and last_result.test_weights is not None
        ):
            decisions = weighted_false_discovery_control(
                result=last_result,
                alpha=alpha,
            )
        else:
            decisions = false_discovery_control(p_values, method="bh") <= alpha

        return self._format_decisions_output(
            np.asarray(decisions, dtype=bool),
            index=index,
            name="batch_selected",
        )

    def step(
        self,
        x: pd.DataFrame | pd.Series | np.ndarray,
        *,
        refit_weights: bool = True,
        apply_batch_select: bool = False,
        alpha: float | None = None,
    ) -> TemporalStepResult:
        """Run one temporal orchestration step over one event or mini-batch.

        Args:
            x: Input event(s) for conformal inference.
            refit_weights: Weighted-mode behavior passed through to detector.
            apply_batch_select: If True, compute side-by-side batch decisions
                (BH/WCS) from the same p-values used in this step.
            alpha: Batch-selection alpha for ``apply_batch_select=True``.
                Defaults to ``0.05`` when omitted.

        Returns:
            ``TemporalStepResult`` for this step.
        """
        p_values_out = self.detector.compute_p_values(x, refit_weights=refit_weights)

        if isinstance(p_values_out, pd.Series):
            index: pd.Index | None = p_values_out.index
            p_values = _as_1d_float_p_values("p_values", p_values_out.to_numpy())
            p_values_result: np.ndarray | pd.Series = p_values_out.copy()
        else:
            index = None
            p_values = _as_1d_float_p_values("p_values", np.asarray(p_values_out))
            p_values_result = p_values.copy()

        online_decisions_result: np.ndarray | pd.Series | None = None
        if self.online_controller is not None:
            online_decisions = online_false_discovery_control(
                p_values=p_values,
                controller=self.online_controller,
            )
            online_decisions_result = self._format_decisions_output(
                online_decisions,
                index=index,
                name="online_selected",
            )

        martingale_states: list[MartingaleState] | None = None
        if self.martingale is not None:
            martingale_states = self.martingale.update_many(p_values)

        triggered_alarms = self._collect_triggered_alarms(martingale_states)

        self._last_batch_decisions = None
        if apply_batch_select:
            batch_alpha = 0.05 if alpha is None else float(alpha)
            _validate_alpha(batch_alpha)
            self._last_batch_decisions = self._compute_batch_decisions(
                p_values=p_values,
                index=index,
                alpha=batch_alpha,
            )

        result = TemporalStepResult(
            p_values=p_values_result,
            online_decisions=online_decisions_result,
            martingale_states=martingale_states,
            triggered_alarms=triggered_alarms,
        )

        if self.on_step_end is not None:
            self.on_step_end(result)
        if triggered_alarms and self.on_alarm is not None:
            self.on_alarm(result)
        return result


__all__ = [
    "OnlineFDRController",
    "TemporalSession",
    "TemporalStepResult",
]
