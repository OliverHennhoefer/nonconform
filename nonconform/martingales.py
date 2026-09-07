"""Exchangeability martingales for sequential conformal evidence.

This module implements p-value-based martingales and alarm statistics for
streaming or temporal monitoring workflows. In practice, you feed one conformal
p-value at a time and read a running evidence state after each update.

Implemented martingales:
    - PowerMartingale
    - SimpleMixtureMartingale
    - SimpleJumperMartingale
    - MixtureMartingale

All classes consume conformal p-values in ``[0, 1]``. Alarm statistics are
computed from martingale ratio increments or mixed component statistics and
exposed together with the current martingale value in :class:`MartingaleState`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

_PROB_TOL = 1e-12
_FLOAT_INFO = np.finfo(float)
_LOG_FLOAT_MAX = float(np.log(_FLOAT_INFO.max))
_LOG_FLOAT_MIN = float(np.log(_FLOAT_INFO.smallest_subnormal))


def _logsumexp(values: np.ndarray) -> float:
    """Return numerically stable ``log(sum(exp(values)))``."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    max_val = float(np.max(arr))
    if np.isneginf(max_val):
        return float("-inf")
    if np.isposinf(max_val):
        return float("inf")
    return float(max_val + np.log(np.sum(np.exp(arr - max_val))))


def _validate_probability(value: float) -> float:
    """Validate and normalize a p-value scalar."""
    try:
        p_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("p_value must be a numeric scalar.") from exc
    if not np.isfinite(p_value):
        raise ValueError("p_value must be finite.")
    if p_value < -_PROB_TOL or p_value > 1.0 + _PROB_TOL:
        raise ValueError(f"p_value must be in [0, 1], got {p_value}.")
    if p_value < 0.0:
        return 0.0
    if p_value > 1.0:
        return 1.0
    return p_value


def _validate_positive_threshold(name: str, threshold: float | None) -> float | None:
    """Validate alarm threshold semantics."""
    if threshold is None:
        return None
    try:
        threshold_float = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite float or None.") from exc
    if not np.isfinite(threshold_float) or threshold_float <= 0.0:
        raise ValueError(f"{name} must be a positive finite float or None.")
    return threshold_float


def _linear_from_log(log_value: float) -> float:
    """Convert a log-statistic into linear scale."""
    if np.isposinf(log_value) or log_value >= _LOG_FLOAT_MAX:
        return float("inf")
    if np.isneginf(log_value) or log_value <= _LOG_FLOAT_MIN:
        return 0.0
    return float(np.exp(log_value))


def _power_log_increment(p_value: float, epsilon: float) -> float:
    """Return ``log(epsilon * p_value**(epsilon - 1))`` with boundary handling."""
    if p_value == 0.0:
        if epsilon < 1.0:
            return float("inf")
        return 0.0
    return float(np.log(epsilon) + (epsilon - 1.0) * np.log(p_value))


def _log_harmonic_restart_weight(step: int) -> float:
    """Return log pi_t for the harmonic restart prior."""
    return float(-np.log(step) - np.log(step + 1))


def _log_harmonic_restart_tail(step: int) -> float:
    """Return log tail mass after ``step`` for the harmonic restart prior."""
    return float(-np.log(step + 1))


@dataclass(slots=True, frozen=True)
class AlarmConfig:
    """Optional alarm thresholds for martingale evidence statistics.

    Thresholds are disabled when set to ``None``. Each threshold compares
    against a running statistic in :class:`MartingaleState`.

    ``ville_threshold`` and ``restarted_ville_threshold`` are Ville thresholds
    for e-processes. ``cusum_threshold`` and ``shiryaev_roberts_threshold`` are
    change-evidence thresholds and should not be interpreted as
    probability-of-ever-crossing Ville thresholds without a separate theorem for
    the exact statistic.

    Attributes:
        ville_threshold: Threshold for the original cumulative martingale.
        restarted_ville_threshold: Threshold for the harmonic restart-mixture
            e-process.
        cusum_threshold: Threshold for the multiplicative CUSUM statistic.
        shiryaev_roberts_threshold: Threshold for the Shiryaev-Roberts statistic.
    """

    ville_threshold: float | None = None
    restarted_ville_threshold: float | None = None
    cusum_threshold: float | None = None
    shiryaev_roberts_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate optional threshold values after dataclass initialization."""
        object.__setattr__(
            self,
            "ville_threshold",
            _validate_positive_threshold("ville_threshold", self.ville_threshold),
        )
        object.__setattr__(
            self,
            "restarted_ville_threshold",
            _validate_positive_threshold(
                "restarted_ville_threshold", self.restarted_ville_threshold
            ),
        )
        object.__setattr__(
            self,
            "cusum_threshold",
            _validate_positive_threshold("cusum_threshold", self.cusum_threshold),
        )
        object.__setattr__(
            self,
            "shiryaev_roberts_threshold",
            _validate_positive_threshold(
                "shiryaev_roberts_threshold",
                self.shiryaev_roberts_threshold,
            ),
        )


@dataclass(slots=True, frozen=True)
class MartingaleState:
    """Immutable snapshot of evidence statistics after one update.

    Linear-scale values may be ``0`` or ``inf`` after floating-point underflow or
    overflow; the corresponding ``log_*`` field preserves the log-scale state.
    ``triggered_alarms`` reports thresholds crossed at this step and is not a
    latched alarm history.

    ``e_value`` is the ordinary capital ratio, not necessarily an increment
    that reproduces the other statistics (see :class:`MixtureMartingale`).
    """

    step: int
    p_value: float
    log_martingale: float
    martingale: float
    log_restarted_martingale: float
    restarted_martingale: float
    log_cusum: float
    cusum: float
    log_shiryaev_roberts: float
    shiryaev_roberts: float
    triggered_alarms: tuple[str, ...]
    log_e_value: float = 0.0
    e_value: float = 1.0


class BaseMartingale(ABC):
    """Abstract base class for p-value-driven sequential evidence.

    The default update multiplies capital by a non-negative betting factor and
    updates alarm statistics from that factor. Composite subclasses may instead
    aggregate component statistics. Ville-threshold interpretation
    requires the resulting process to be an e-process under the null, which in
    turn depends on the conditional validity of the input p-values. Merely
    passing values in ``[0, 1]`` does not establish that property.
    """

    def __init__(self, alarm_config: AlarmConfig | None = None) -> None:
        self._alarm_config = alarm_config if alarm_config is not None else AlarmConfig()
        self.reset()

    @property
    def state(self) -> MartingaleState:
        """Return current state snapshot."""
        return self._current_state()

    def reset(self) -> None:
        """Reset martingale and alarm statistics to initial values."""
        self._step = 0
        self._last_p_value = float("nan")
        self._last_log_increment = 0.0
        self._log_martingale = 0.0
        self._log_active_restarted_mass = float("-inf")
        self._log_restarted_martingale = 0.0
        # CUSUM/SR start at 0 on linear scale -> -inf in log space.
        self._log_cusum = float("-inf")
        self._log_shiryaev_roberts = float("-inf")
        self._reset_method_state()

    def update_many(
        self, p_values: Sequence[float] | np.ndarray
    ) -> list[MartingaleState]:
        """Update state for each p-value in order and return all snapshots."""
        return [self.update(float(p_value)) for p_value in p_values]

    def update(self, p_value: float) -> MartingaleState:
        """Ingest one p-value in ``[0, 1]`` and return the updated state."""
        p_value_validated = _validate_probability(p_value)
        log_increment = self._compute_log_increment(p_value_validated)
        if np.isnan(log_increment):
            raise ValueError("Martingale increment is NaN.")

        self._step += 1
        self._last_p_value = p_value_validated
        self._last_log_increment = log_increment
        self._update_statistics(log_increment)
        return self._current_state()

    def _update_statistics(self, log_increment: float) -> None:
        """Advance evidence statistics after the step and increment are recorded."""
        self._log_martingale += log_increment
        self._log_active_restarted_mass = float(
            log_increment
            + np.logaddexp(
                self._log_active_restarted_mass,
                _log_harmonic_restart_weight(self._step),
            )
        )
        self._log_restarted_martingale = float(
            np.logaddexp(
                self._log_active_restarted_mass,
                _log_harmonic_restart_tail(self._step),
            )
        )
        self._log_cusum = float(log_increment + max(self._log_cusum, 0.0))
        self._log_shiryaev_roberts = float(
            log_increment + np.logaddexp(0.0, self._log_shiryaev_roberts)
        )

    @abstractmethod
    def _reset_method_state(self) -> None:
        """Reset method-specific state."""

    @abstractmethod
    def _compute_log_increment(self, p_value: float) -> float:
        """Return log martingale ratio increment for one p-value."""

    def _current_state(self) -> MartingaleState:
        """Create immutable state snapshot."""
        return MartingaleState(
            step=self._step,
            p_value=self._last_p_value,
            log_martingale=self._log_martingale,
            martingale=_linear_from_log(self._log_martingale),
            log_restarted_martingale=self._log_restarted_martingale,
            restarted_martingale=_linear_from_log(self._log_restarted_martingale),
            log_cusum=self._log_cusum,
            cusum=_linear_from_log(self._log_cusum),
            log_shiryaev_roberts=self._log_shiryaev_roberts,
            shiryaev_roberts=_linear_from_log(self._log_shiryaev_roberts),
            triggered_alarms=self._triggered_alarms(),
            log_e_value=self._last_log_increment,
            e_value=_linear_from_log(self._last_log_increment),
        )

    def _triggered_alarms(self) -> tuple[str, ...]:
        """Return tuple of currently triggered alarm names."""
        alarms: list[str] = []
        if (
            self._alarm_config.ville_threshold is not None
            and self._log_martingale >= np.log(self._alarm_config.ville_threshold)
        ):
            alarms.append("ville")
        if (
            self._alarm_config.restarted_ville_threshold is not None
            and self._log_restarted_martingale
            >= np.log(self._alarm_config.restarted_ville_threshold)
        ):
            alarms.append("restarted_ville")
        if self._alarm_config.cusum_threshold is not None and self._log_cusum >= np.log(
            self._alarm_config.cusum_threshold
        ):
            alarms.append("cusum")
        if (
            self._alarm_config.shiryaev_roberts_threshold is not None
            and self._log_shiryaev_roberts
            >= np.log(self._alarm_config.shiryaev_roberts_threshold)
        ):
            alarms.append("shiryaev_roberts")
        return tuple(alarms)


class PowerMartingale(BaseMartingale):
    r"""Power martingale with fixed ``epsilon`` in ``(0, 1]``.

    Each p-value contributes the betting factor
    ``epsilon * p_value ** (epsilon - 1)``. Values below one emphasize small
    p-values; ``epsilon=1`` produces the constant factor one.
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        alarm_config: AlarmConfig | None = None,
    ) -> None:
        self.epsilon = float(epsilon)
        if not (0.0 < self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in (0, 1], got {self.epsilon}.")
        super().__init__(alarm_config=alarm_config)

    def _reset_method_state(self) -> None:
        """Power martingale has no additional reset state."""

    def _compute_log_increment(self, p_value: float) -> float:
        """Return log increment for one p-value."""
        return _power_log_increment(p_value, self.epsilon)


class SimpleMixtureMartingale(BaseMartingale):
    """Equal-weight mixture of power martingales over an epsilon grid.

    When ``epsilons`` is omitted, the grid contains ``n_grid`` evenly spaced
    values from ``min_epsilon`` through one. The mixture tracks the arithmetic
    mean of component capitals, evaluated stably in log space.

    Its alarm statistics use ratios of that all-history mixture capital. For
    fixed-weight averages of independently maintained alarm statistics, use
    :class:`MixtureMartingale` with separate :class:`PowerMartingale` experts.
    """

    def __init__(
        self,
        epsilons: Sequence[float] | np.ndarray | None = None,
        *,
        n_grid: int = 100,
        min_epsilon: float = 0.01,
        alarm_config: AlarmConfig | None = None,
    ) -> None:
        if epsilons is None:
            if n_grid < 2:
                raise ValueError(f"n_grid must be at least 2, got {n_grid}.")
            if not (0.0 < min_epsilon <= 1.0):
                raise ValueError(f"min_epsilon must be in (0, 1], got {min_epsilon}.")
            self.epsilons = np.linspace(float(min_epsilon), 1.0, int(n_grid))
        else:
            self.epsilons = np.asarray(epsilons, dtype=float)
            if self.epsilons.ndim != 1 or self.epsilons.size == 0:
                raise ValueError("epsilons must be a non-empty 1D sequence.")

        if not np.all(np.isfinite(self.epsilons)):
            raise ValueError("epsilons must be finite.")
        if np.any((self.epsilons <= 0.0) | (self.epsilons > 1.0)):
            raise ValueError("All epsilons must be in (0, 1].")
        self._n_eps = int(self.epsilons.size)
        super().__init__(alarm_config=alarm_config)

    def _reset_method_state(self) -> None:
        """Initialize per-epsilon log capital state."""
        self._log_capitals = np.zeros(self._n_eps, dtype=float)
        self._log_mixture_value = 0.0

    def _compute_log_increment(self, p_value: float) -> float:
        """Return log increment from mixture capital ratio."""
        if p_value == 0.0:
            log_r_each = np.where(self.epsilons < 1.0, np.inf, 0.0)
        else:
            log_r_each = np.log(self.epsilons) + (self.epsilons - 1.0) * np.log(p_value)

        self._log_capitals = self._log_capitals + log_r_each
        log_mixture_new = _logsumexp(self._log_capitals) - np.log(self._n_eps)
        if np.isposinf(log_mixture_new):
            log_increment = (
                0.0 if np.isposinf(self._log_mixture_value) else float("inf")
            )
        else:
            log_increment = float(log_mixture_new - self._log_mixture_value)
        self._log_mixture_value = log_mixture_new
        return log_increment


class MixtureMartingale(BaseMartingale):
    """Fixed-weight averages of independently updated martingale evidence.

    Each component receives the same p-value. Ordinary capital, harmonic
    restart evidence, CUSUM, and Shiryaev-Roberts statistics are averaged
    separately; alarm statistics are not computed from mixture capital ratios.
    Fixed normalized mixtures inherit the corresponding validity guarantees
    only when every participating component satisfies those guarantees under
    the common null and filtration. Composition does not remove learning inertia
    inside an individual expert.

    Args:
        martingales: Nonempty sequence of reset ``BaseMartingale`` instances.
            Components are deep-copied and exclusively owned by this mixture.
            Custom martingales and nested mixtures are supported.
        weights: Finite nonnegative weights with positive total. Normalized
            once; omitted weights give equal allocation. Zero-weight components
            are excluded from copying and updates.
        alarm_config: Thresholds applied to the combined statistics. Component
            alarms are not propagated.

    Notes:
        ``state.e_value`` and ``state.log_e_value`` describe the ordinary
        mixture capital ratio only. They cannot reproduce the combined CUSUM,
        SR, or restart statistics through a shared-increment recursion. For
        consecutive zero capitals or consecutive infinite log-capitals, the
        undefined ratio is reported as the neutral diagnostic factor one.
        Finite log-capitals retain their ratio even when linear values overflow.

        A component failure blocks further updates until ``reset()``; the
        mixture state remains at its last completed update. When used inside
        an ``ExchangeabilityMonitor``, reset the whole monitor after failure.
        Aggregation adds O(J) work and storage beyond the J component costs.
    """

    def __init__(
        self,
        martingales: Sequence[BaseMartingale],
        *,
        weights: Sequence[float] | None = None,
        alarm_config: AlarmConfig | None = None,
    ) -> None:
        components = tuple(martingales)
        if not components:
            raise ValueError("martingales must be a nonempty sequence.")
        for component in components:
            if not isinstance(component, BaseMartingale):
                raise TypeError("Every component must be a BaseMartingale.")
            if component.state.step != 0:
                raise ValueError("Every component must be reset at construction.")

        try:
            raw_weights = (
                np.ones(len(components), dtype=float)
                if weights is None
                else np.asarray(weights, dtype=float)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weights must be a finite numeric sequence.") from exc
        if raw_weights.shape != (len(components),):
            raise ValueError("weights must have one entry per component.")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0):
            raise ValueError("weights must be finite and nonnegative.")
        active = raw_weights > 0
        if not np.any(active):
            raise ValueError("weights must have positive total.")
        # Normalize in log space to preserve tiny positive weights and avoid
        # overflow when several individually finite weights have a huge sum.
        log_weights = np.log(raw_weights[active])
        self._log_weights = log_weights - _logsumexp(log_weights)
        try:
            self._martingales = tuple(
                deepcopy(component)
                for component, participates in zip(components, active, strict=True)
                if participates
            )
        except Exception as exc:
            raise TypeError("martingales must support deep copying.") from exc
        super().__init__(alarm_config=alarm_config)

    def _reset_method_state(self) -> None:
        """Reset all owned components and clear the failure latch."""
        self._failed = True
        for component in self._martingales:
            component.reset()
        self._next_log_statistics = np.array([0.0, 0.0, -np.inf, -np.inf])
        self._failed = False

    def _compute_log_increment(self, p_value: float) -> float:
        """Advance experts and prepare their combined statistics for commit."""
        if self._failed:
            raise RuntimeError("A mixture component failed; reset() before updating.")
        try:
            states = [component.update(p_value) for component in self._martingales]
            if any(state.step != self._step + 1 for state in states):
                raise ValueError("Component steps must advance together.")
            log_statistics = np.array(
                [
                    [
                        state.log_martingale,
                        state.log_restarted_martingale,
                        state.log_cusum,
                        state.log_shiryaev_roberts,
                    ]
                    for state in states
                ]
            ).T
            if np.any(np.isnan(log_statistics)):
                raise ValueError("Component log statistics must not contain NaN.")
            self._next_log_statistics = np.array(
                [_logsumexp(row + self._log_weights) for row in log_statistics]
            )
        except Exception:
            self._failed = True
            raise

        log_capital = float(self._next_log_statistics[0])
        if np.isinf(log_capital) and log_capital == self._log_martingale:
            return 0.0
        return float(log_capital - self._log_martingale)

    def _update_statistics(self, log_increment: float) -> None:
        """Commit each averaged statistic independently of the capital ratio."""
        (
            self._log_martingale,
            self._log_restarted_martingale,
            self._log_cusum,
            self._log_shiryaev_roberts,
        ) = map(float, self._next_log_statistics)


class SimpleJumperMartingale(BaseMartingale):
    """Simple Jumper martingale (Algorithm 1 in Vovk et al.).

    This method mixes three betting components with parameters ``-1``, ``0``,
    and ``1``. Before each bet, ``jump`` redistributes capital equally among the
    components; the current p-value then multiplies each component by
    ``1 + epsilon * (p_value - 0.5)``.
    """

    def __init__(
        self,
        jump: float = 0.01,
        alarm_config: AlarmConfig | None = None,
    ) -> None:
        self.jump = float(jump)
        if not (0.0 < self.jump <= 1.0):
            raise ValueError(f"jump must be in (0, 1], got {self.jump}.")
        self._epsilons = np.array([-1.0, 0.0, 1.0], dtype=float)
        super().__init__(alarm_config=alarm_config)

    def _reset_method_state(self) -> None:
        """Reset Simple Jumper components C_{-1}, C_0, C_1."""
        self._log_components = np.full(3, np.log(1.0 / 3.0), dtype=float)

    def _compute_log_increment(self, p_value: float) -> float:
        """Return log increment by one Simple Jumper update."""
        log_capital_prev = _logsumexp(self._log_components)
        log_stay = np.log1p(-self.jump)
        log_jump = np.log(self.jump / 3.0)

        log_components_after_jump = np.empty_like(self._log_components)
        for i, log_component in enumerate(self._log_components):
            log_components_after_jump[i] = np.logaddexp(
                log_stay + log_component,
                log_jump + log_capital_prev,
            )

        betting_terms = 1.0 + self._epsilons * (p_value - 0.5)
        # Note: with epsilons=[-1, 0, 1] and p_value in [0, 1], betting_terms are
        # always in [0.5, 1.5], so all are strictly positive for valid inputs.

        self._log_components = log_components_after_jump + np.log(betting_terms)
        log_capital_new = _logsumexp(self._log_components)
        return float(log_capital_new - log_capital_prev)


__all__ = [
    "AlarmConfig",
    "BaseMartingale",
    "MartingaleState",
    "MixtureMartingale",
    "PowerMartingale",
    "SimpleJumperMartingale",
    "SimpleMixtureMartingale",
]
