"""Fixed-weight composition tests independent of the mixture implementation."""

from dataclasses import fields, replace

import numpy as np
import pytest
from scipy.special import logsumexp

from nonconform.martingales import (
    AlarmConfig,
    BaseMartingale,
    MixtureMartingale,
    PowerMartingale,
    SimpleJumperMartingale,
    SimpleMixtureMartingale,
)

STATISTICS = (
    "martingale",
    "restarted_martingale",
    "cusum",
    "shiryaev_roberts",
)


def assert_states_equal(actual, expected):
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "triggered_alarms":
            assert left == right
        else:
            np.testing.assert_allclose(left, right, atol=1e-12)


class ScriptedExpert(BaseMartingale):
    def __init__(self, increments):
        self.increments = tuple(increments)
        super().__init__()

    def _reset_method_state(self):
        pass

    def _compute_log_increment(self, p_value):
        return self.increments[self.state.step]


class FailingExpert(PowerMartingale):
    def update(self, p_value):
        state = super().update(p_value)
        if state.step == 2:
            raise RuntimeError("deliberate component failure")
        return state


class NaNExpert(PowerMartingale):
    def _current_state(self):
        state = super()._current_state()
        return replace(state, log_cusum=np.nan) if state.step else state


class NonCopyableExpert(PowerMartingale):
    def __deepcopy__(self, memo):
        raise TypeError("cannot copy")


@pytest.mark.parametrize("weights", [None, [2, 3, 5]])
def test_statistics_match_independently_updated_experts(weights):
    experts = [PowerMartingale(0.3), SimpleJumperMartingale(), PowerMartingale(0.8)]
    mixture = MixtureMartingale(experts, weights=weights)
    normalized = np.ones(3) / 3 if weights is None else np.asarray(weights) / 10
    previous_capital = 1.0
    for p_value in [0.8, 0.03, 0.7, 0.15, 1.0, 0.01]:
        expected = [expert.update(p_value) for expert in experts]
        state = mixture.update(p_value)
        assert state.step == expected[0].step
        assert state.p_value == p_value
        for name in STATISTICS:
            values = [getattr(component, name) for component in expected]
            assert getattr(state, name) == pytest.approx(np.dot(normalized, values))
            expected_log = logsumexp(
                [getattr(component, f"log_{name}") for component in expected],
                b=normalized,
            )
            assert getattr(state, f"log_{name}") == pytest.approx(expected_log)
        assert state.e_value == pytest.approx(state.martingale / previous_capital)
        previous_capital = state.martingale


@pytest.mark.parametrize("factory", [PowerMartingale, SimpleJumperMartingale])
def test_single_expert_equivalence_including_initial_state(factory):
    expert = factory()
    mixture = MixtureMartingale([expert], weights=[7])
    assert_states_equal(mixture.state, expert.state)
    for p_value in [0.9, 0.1, 1.0, 0.02]:
        assert_states_equal(mixture.update(p_value), expert.update(p_value))


def test_nested_mixture_matches_flattened_allocation():
    experts = [PowerMartingale(0.3), SimpleJumperMartingale(), PowerMartingale(0.9)]
    nested = MixtureMartingale(
        [MixtureMartingale(experts[:2], weights=[1, 3]), experts[2]], weights=[2, 3]
    )
    flat = MixtureMartingale(experts, weights=[0.1, 0.3, 0.6])
    for p_value in [0.7, 0.02, 0.9, 0.04]:
        assert_states_equal(nested.update(p_value), flat.update(p_value))


def test_mixture_owns_independent_components_and_weights():
    expert = SimpleJumperMartingale()
    weights = np.array([1.0, 2.0])
    inputs = [expert, expert]
    mixture = MixtureMartingale(inputs, weights=weights)
    reference = SimpleJumperMartingale()
    weights[:] = [0, 1]
    inputs.clear()
    expert.jump = 0.5
    expert.update(0.02)
    for p_value in [0.7, 0.05, 0.01]:
        assert_states_equal(mixture.update(p_value), reference.update(p_value))
    assert expert.state.step == 1
    mixture.reset()
    assert expert.state.step == 1
    reference.reset()
    assert_states_equal(mixture.state, reference.state)


def test_reset_and_batch_updates_match_scalar_updates():
    mixture = MixtureMartingale([PowerMartingale(0.5), SimpleJumperMartingale()])
    p_values = [0.01, 0.9, 0.2, 0.02]
    expected = mixture.update_many(p_values)
    mixture.reset()
    for actual, reference in zip(
        [mixture.update(p) for p in p_values], expected, strict=True
    ):
        assert_states_equal(actual, reference)


@pytest.mark.parametrize(
    "weights",
    [[0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [1], [[1, 2]], ["bad", 1]],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        MixtureMartingale([PowerMartingale(), PowerMartingale()], weights=weights)


def test_invalid_components():
    with pytest.raises(ValueError, match="nonempty"):
        MixtureMartingale([])
    with pytest.raises(TypeError, match="BaseMartingale"):
        MixtureMartingale([object()])
    expert = PowerMartingale()
    expert.update(0.5)
    with pytest.raises(ValueError, match="reset"):
        MixtureMartingale([expert])
    with pytest.raises(TypeError, match="deep copying"):
        MixtureMartingale([NonCopyableExpert()])


def test_zero_weight_component_does_not_participate():
    mixture = MixtureMartingale(
        [PowerMartingale(1), NonCopyableExpert()], weights=[1, 0]
    )
    assert mixture.update(0).martingale == 1
    assert mixture.update(0).e_value == 1


def test_weight_normalization_avoids_overflow_and_keeps_tiny_positive_weights():
    huge = np.finfo(float).max
    mixture = MixtureMartingale(
        [PowerMartingale(1), PowerMartingale(1)], weights=[huge, huge]
    )
    assert mixture.update(0.5).martingale == pytest.approx(1)
    tiny = np.nextafter(0.0, 1.0)
    mixture = MixtureMartingale(
        [PowerMartingale(1), PowerMartingale(0.5)], weights=[huge, tiny]
    )
    assert np.isposinf(mixture.update(0).log_martingale)


@pytest.mark.parametrize("p_value", [-0.1, 1.1, np.nan, np.inf, "bad"])
def test_invalid_p_value_does_not_advance_or_poison_mixture(p_value):
    mixture = MixtureMartingale([PowerMartingale(), SimpleJumperMartingale()])
    previous = mixture.update(0.7)
    with pytest.raises(ValueError):
        mixture.update(p_value)
    assert_states_equal(mixture.state, previous)
    reference = MixtureMartingale([PowerMartingale(), SimpleJumperMartingale()])
    reference.update(0.7)
    assert_states_equal(mixture.update(0.1), reference.update(0.1))


@pytest.mark.parametrize("expert", [FailingExpert(), NaNExpert()])
def test_component_failure_requires_reset(expert):
    mixture = MixtureMartingale([PowerMartingale(), expert])
    previous = mixture.state
    if isinstance(expert, FailingExpert):
        previous = mixture.update(0.7)
    with pytest.raises((RuntimeError, ValueError), match=r"failure|NaN"):
        mixture.update(0.1)
    assert_states_equal(mixture.state, previous)
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        mixture.update(0.1)
    mixture.reset()
    assert mixture.state.step == 0
    if isinstance(expert, FailingExpert):
        assert mixture.update(0.7).step == 1


def test_only_outer_alarms_are_reported():
    child = PowerMartingale(0.5, AlarmConfig(ville_threshold=1))
    mixture = MixtureMartingale([child])
    assert mixture.update(0.1).triggered_alarms == ()
    mixture = MixtureMartingale(
        [PowerMartingale(0.5)],
        alarm_config=AlarmConfig(1, 1, 1, 1),
    )
    assert mixture.update(0.1).triggered_alarms == (
        "ville",
        "restarted_ville",
        "cusum",
        "shiryaev_roberts",
    )


@pytest.mark.parametrize("log_factor", [-1000.0, 1000.0])
def test_finite_log_capital_keeps_ratio_despite_linear_underflow_or_overflow(
    log_factor,
):
    mixture = MixtureMartingale([ScriptedExpert([log_factor] * 3)])
    for step in range(1, 4):
        state = mixture.update(0.5)
        assert state.log_martingale == pytest.approx(step * log_factor)
        assert state.log_e_value == pytest.approx(log_factor)
        assert not any(np.isnan(getattr(state, f"log_{name}")) for name in STATISTICS)


@pytest.mark.parametrize("log_factor", [-np.inf, np.inf])
def test_undefined_capital_ratios_have_neutral_diagnostics(log_factor):
    mixture = MixtureMartingale([ScriptedExpert([log_factor, 0.0])])
    assert mixture.update(0.5).log_e_value == log_factor
    state = mixture.update(0.5)
    assert state.log_e_value == 0
    assert state.e_value == 1
    assert state.log_martingale == log_factor
    assert not any(np.isnan(getattr(state, f"log_{name}")) for name in STATISTICS)


def test_zero_p_values_preserve_infinite_evidence():
    mixture = MixtureMartingale([PowerMartingale(0.5), PowerMartingale(1)])
    for p_value in [0, 0, 1, 0.5]:
        state = mixture.update(p_value)
        assert all(np.isposinf(getattr(state, f"log_{name}")) for name in STATISTICS)


def test_late_change_uses_per_expert_statistics_without_changing_legacy_mixture():
    mixture = MixtureMartingale([PowerMartingale(0.5), PowerMartingale(1)])
    legacy = SimpleMixtureMartingale(epsilons=[0.5, 1])
    for p_value in np.random.default_rng(42).uniform(size=10_000):
        mixture.update(p_value)
        legacy.update(p_value)
    for _ in range(100):
        state = mixture.update(0.01)
        legacy_state = legacy.update(0.01)
        assert state.log_martingale == pytest.approx(legacy_state.log_martingale)
        assert legacy_state.e_value == pytest.approx(1)
    assert legacy_state.log_shiryaev_roberts < np.log(1e8)
    assert state.log_shiryaev_roberts > np.log(1e8)
    assert state.log_restarted_martingale > np.log(100)
    assert legacy_state.log_restarted_martingale < np.log(100)
