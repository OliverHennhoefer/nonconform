"""Unit tests for martingale evidence utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nonconform.martingales import (
    AlarmConfig,
    BaseMartingale,
    PowerMartingale,
    SimpleJumperMartingale,
    SimpleMixtureMartingale,
)


class SequenceIncrementMartingale(BaseMartingale):
    """Test helper that emits a fixed sequence of log increments."""

    def __init__(self, log_increments: list[float]) -> None:
        self._initial_log_increments = log_increments.copy()
        self._log_increments = log_increments.copy()
        super().__init__()

    def _reset_method_state(self) -> None:
        self._log_increments = self._initial_log_increments.copy()

    def _compute_log_increment(self, p_value: float) -> float:
        _ = p_value
        if len(self._log_increments) == 0:
            raise RuntimeError("No increments left in SequenceIncrementMartingale.")
        return self._log_increments.pop(0)


def _assert_state_close(left, right):
    assert left.step == right.step
    assert left.triggered_alarms == right.triggered_alarms
    np.testing.assert_allclose(left.p_value, right.p_value)
    np.testing.assert_allclose(left.log_martingale, right.log_martingale)
    np.testing.assert_allclose(left.martingale, right.martingale)
    np.testing.assert_allclose(left.log_cusum, right.log_cusum)
    np.testing.assert_allclose(left.cusum, right.cusum)
    np.testing.assert_allclose(left.log_shiryaev_roberts, right.log_shiryaev_roberts)
    np.testing.assert_allclose(left.shiryaev_roberts, right.shiryaev_roberts)


def _simple_jumper_manual(
    p_values: np.ndarray, jump: float
) -> tuple[np.ndarray, np.ndarray]:
    epsilons = np.array([-1.0, 0.0, 1.0], dtype=float)
    c_eps = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    capitals = np.empty(len(p_values), dtype=float)
    log_capitals = np.empty(len(p_values), dtype=float)

    for i, p_value in enumerate(p_values):
        c_total = float(np.sum(c_eps))
        c_eps = (1.0 - jump) * c_eps + (jump / 3.0) * c_total
        c_eps = c_eps * (1.0 + epsilons * (p_value - 0.5))
        c_total = float(np.sum(c_eps))
        capitals[i] = c_total
        log_capitals[i] = np.log(c_total)

    return capitals, log_capitals


class TestValidation:
    def test_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            BaseMartingale()  # type: ignore[abstract]

    def test_alarm_config_rejects_non_positive(self):
        with pytest.raises(ValueError, match="ville_threshold"):
            AlarmConfig(ville_threshold=0.0)
        with pytest.raises(ValueError, match="cusum_threshold"):
            AlarmConfig(cusum_threshold=-1.0)
        with pytest.raises(ValueError, match="shiryaev_roberts_threshold"):
            AlarmConfig(shiryaev_roberts_threshold=float("inf"))

    def test_invalid_p_values_raise(self):
        martingale = PowerMartingale(epsilon=0.5)
        with pytest.raises(ValueError, match="p_value"):
            martingale.update(float("nan"))
        with pytest.raises(ValueError, match="p_value"):
            martingale.update(-0.1)
        with pytest.raises(ValueError, match="p_value"):
            martingale.update(1.1)

    def test_constructor_parameter_validation(self):
        with pytest.raises(ValueError, match="epsilon"):
            PowerMartingale(epsilon=0.0)
        with pytest.raises(ValueError, match="jump"):
            SimpleJumperMartingale(jump=0.0)
        with pytest.raises(ValueError, match="n_grid"):
            SimpleMixtureMartingale(n_grid=1)
        with pytest.raises(ValueError, match="epsilons"):
            SimpleMixtureMartingale(epsilons=[])


class TestInitialState:
    def test_initial_state_matches_defaults(self):
        state = PowerMartingale(epsilon=0.7).state
        assert state.step == 0
        assert np.isnan(state.p_value)
        assert state.martingale == pytest.approx(1.0)
        assert state.cusum == pytest.approx(0.0)
        assert state.shiryaev_roberts == pytest.approx(0.0)
        assert state.triggered_alarms == ()


class TestPowerMartingale:
    def test_exact_update_values(self):
        martingale = PowerMartingale(epsilon=0.5)
        states = martingale.update_many(np.array([0.25, 0.04], dtype=float))

        assert states[0].martingale == pytest.approx(1.0)
        assert states[1].martingale == pytest.approx(2.5)

        expected_log = np.log(0.5) + (0.5 - 1.0) * np.log(0.04)
        assert states[1].log_martingale == pytest.approx(expected_log)

    def test_zero_p_value_with_epsilon_below_one_yields_infinite_evidence(self):
        martingale = PowerMartingale(epsilon=0.5)
        state = martingale.update(0.0)
        assert math.isinf(state.martingale)
        assert state.log_martingale == float("inf")


class TestSimpleMixtureMartingale:
    def test_mixture_matches_manual_two_point_grid(self):
        martingale = SimpleMixtureMartingale(epsilons=[0.5, 1.0])
        states = martingale.update_many(np.array([0.25, 0.04], dtype=float))

        # epsilon=0.5 capital: 2.5, epsilon=1.0 capital: 1.0 -> mixture 1.75
        assert states[-1].martingale == pytest.approx(1.75)
        assert states[-1].log_martingale == pytest.approx(np.log(1.75))

    def test_mixture_remains_well_defined_after_zero_p_value(self):
        martingale = SimpleMixtureMartingale(epsilons=[0.5, 1.0])

        first = martingale.update(0.0)
        second = martingale.update(0.5)

        assert first.log_martingale == float("inf")
        assert second.log_martingale == float("inf")
        assert second.step == 2


class TestSimpleJumperMartingale:
    def test_exact_simple_jumper_updates(self):
        p_values = np.array([0.2, 0.8, 0.3], dtype=float)
        jump = 0.01
        manual_capitals, manual_logs = _simple_jumper_manual(p_values, jump=jump)

        martingale = SimpleJumperMartingale(jump=jump)
        states = martingale.update_many(p_values)
        for i, state in enumerate(states):
            assert state.martingale == pytest.approx(manual_capitals[i])
            assert state.log_martingale == pytest.approx(manual_logs[i])


class TestAlarmRecursions:
    def test_cusum_and_sr_match_linear_recursions(self):
        ratios = np.array([2.0, 0.5, 3.0, 0.2], dtype=float)
        log_ratios = list(np.log(ratios))
        martingale = SequenceIncrementMartingale(log_increments=log_ratios)

        gamma = 0.0
        sr = 0.0
        for ratio in ratios:
            gamma = ratio * max(gamma, 1.0)
            sr = ratio * (sr + 1.0)
            state = martingale.update(0.5)
            assert state.cusum == pytest.approx(gamma)
            assert state.shiryaev_roberts == pytest.approx(sr)

    def test_shiryaev_roberts_log_recursion_stays_finite_for_large_values(self):
        martingale = SequenceIncrementMartingale(log_increments=[400.0, 400.0, 400.0])
        states = martingale.update_many(np.array([0.5, 0.5, 0.5], dtype=float))

        assert np.isfinite(states[-1].log_shiryaev_roberts)
        assert states[-1].log_shiryaev_roberts == pytest.approx(1200.0)


class TestUtilityBehavior:
    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (PowerMartingale, {"epsilon": 0.7}),
            (SimpleMixtureMartingale, {"epsilons": [0.3, 0.6, 1.0]}),
            (SimpleJumperMartingale, {"jump": 0.03}),
        ],
    )
    def test_update_many_matches_iterative_updates(self, factory, kwargs):
        p_values = np.array([0.14, 0.72, 0.31, 0.56, 0.83], dtype=float)
        martingale_many = factory(**kwargs)
        many_states = martingale_many.update_many(p_values)

        martingale_iter = factory(**kwargs)
        iter_states = [martingale_iter.update(float(p)) for p in p_values]

        for state_many, state_iter in zip(many_states, iter_states, strict=True):
            _assert_state_close(state_many, state_iter)

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (PowerMartingale, {"epsilon": 0.4}),
            (SimpleMixtureMartingale, {"epsilons": [0.4, 0.8, 1.0]}),
            (SimpleJumperMartingale, {"jump": 0.02}),
        ],
    )
    def test_reset_reproducibility(self, factory, kwargs):
        p_values = np.array([0.17, 0.29, 0.61, 0.92], dtype=float)
        martingale = factory(**kwargs)
        first_pass = martingale.update_many(p_values)
        martingale.reset()
        second_pass = martingale.update_many(p_values)

        for first_state, second_state in zip(first_pass, second_pass, strict=True):
            _assert_state_close(first_state, second_state)

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (PowerMartingale, {"epsilon": 0.8}),
            (SimpleMixtureMartingale, {"epsilons": [0.25, 0.5, 0.75, 1.0]}),
            (SimpleJumperMartingale, {"jump": 0.01}),
        ],
    )
    def test_long_run_numerical_stability(self, factory, kwargs):
        rng = np.random.default_rng(42)
        p_values = rng.uniform(1e-6, 1.0 - 1e-6, size=2_000)
        martingale = factory(**kwargs)
        states = martingale.update_many(p_values)
        final = states[-1]
        assert np.isfinite(final.log_martingale)
        assert np.isfinite(final.log_cusum)
        assert np.isfinite(final.log_shiryaev_roberts)

    def test_alarm_triggering_is_opt_in(self):
        martingale = PowerMartingale(
            epsilon=0.5,
            alarm_config=AlarmConfig(
                ville_threshold=2.0,
                cusum_threshold=2.0,
                shiryaev_roberts_threshold=2.0,
            ),
        )
        state = martingale.update(0.04)  # increment = 2.5
        assert set(state.triggered_alarms) == {"ville", "cusum", "shiryaev_roberts"}
