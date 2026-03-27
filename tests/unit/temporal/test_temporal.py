from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from nonconform import ConformalDetector, Split
from nonconform.martingales import AlarmConfig, PowerMartingale
from nonconform.temporal import TemporalSession
from tests.conftest import MockDetector


class ThresholdController:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold
        self.calls: list[float] = []
        self.reset_calls = 0

    def test_one(self, p_value: float) -> bool:
        self.calls.append(float(p_value))
        return p_value <= self.threshold

    def reset(self) -> None:
        self.reset_calls += 1


class BadControllerMissingMethod:
    pass


class BadControllerReturnType:
    def test_one(self, p_value: float) -> str:
        _ = p_value
        return "yes"


def _build_detector(seed: int = 42) -> ConformalDetector:
    scores = np.linspace(-2.0, 2.0, 300)
    detector = ConformalDetector(
        detector=MockDetector(scores=scores),
        strategy=Split(n_calib=0.2),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    x_train = rng.standard_normal((100, 4))
    detector.fit(x_train)
    return detector


def test_step_supports_single_instance_and_minibatch():
    detector = _build_detector()
    session = TemporalSession(detector=detector)

    x_single = np.array([[0.1, 0.2, -0.1, 0.4]])
    x_batch = np.array([[0.0, 0.1, 0.2, 0.3], [0.4, -0.3, 0.1, -0.2]])

    single_result = session.step(x_single)
    batch_result = session.step(x_batch)

    assert single_result.online_decisions is None
    assert single_result.martingale_states is None
    assert single_result.triggered_alarms == ()
    assert np.asarray(single_result.p_values).shape == (1,)
    assert np.asarray(batch_result.p_values).shape == (2,)


def test_online_controller_matches_manual_loop():
    detector = _build_detector(seed=7)
    controller = ThresholdController(threshold=0.25)
    session = TemporalSession(detector=detector, online_controller=controller)

    rng = np.random.default_rng(13)
    x_test = rng.standard_normal((12, 4))
    result = session.step(x_test)

    p_values = np.asarray(result.p_values, dtype=float)
    manual = np.array([p <= 0.25 for p in p_values], dtype=bool)
    np.testing.assert_array_equal(
        np.asarray(result.online_decisions, dtype=bool), manual
    )
    np.testing.assert_allclose(controller.calls, p_values)


def test_martingale_updates_match_direct_update_many():
    detector = _build_detector(seed=15)
    martingale = PowerMartingale(epsilon=0.5)
    session = TemporalSession(detector=detector, martingale=martingale)

    rng = np.random.default_rng(22)
    x_test = rng.standard_normal((10, 4))
    result = session.step(x_test)
    p_values = np.asarray(result.p_values, dtype=float)

    expected_states = PowerMartingale(epsilon=0.5).update_many(p_values)
    assert result.martingale_states is not None
    assert len(result.martingale_states) == len(expected_states)
    for actual, expected in zip(result.martingale_states, expected_states, strict=True):
        assert actual.step == expected.step
        assert actual.triggered_alarms == expected.triggered_alarms
        np.testing.assert_allclose(actual.log_martingale, expected.log_martingale)
        np.testing.assert_allclose(actual.log_cusum, expected.log_cusum)
        np.testing.assert_allclose(
            actual.log_shiryaev_roberts,
            expected.log_shiryaev_roberts,
        )


def test_pandas_index_is_preserved_for_outputs():
    detector = _build_detector(seed=19)
    session = TemporalSession(
        detector=detector,
        online_controller=ThresholdController(threshold=0.2),
    )

    rng = np.random.default_rng(31)
    index = pd.Index([f"id_{i}" for i in range(6)])
    x_test = pd.DataFrame(rng.standard_normal((6, 4)), index=index)

    result = session.step(x_test, apply_batch_select=True, alpha=0.1)

    assert isinstance(result.p_values, pd.Series)
    assert isinstance(result.online_decisions, pd.Series)
    assert result.p_values.index.equals(index)
    assert result.online_decisions.index.equals(index)
    assert isinstance(session.last_batch_decisions, pd.Series)
    assert session.last_batch_decisions.index.equals(index)


def test_apply_batch_select_matches_bh_on_same_p_values():
    detector = _build_detector(seed=29)
    session = TemporalSession(detector=detector)

    rng = np.random.default_rng(30)
    x_test = rng.standard_normal((16, 4))
    result = session.step(x_test, apply_batch_select=True, alpha=0.15)

    expected = (
        false_discovery_control(
            np.asarray(result.p_values, dtype=float),
            method="bh",
        )
        <= 0.15
    )
    np.testing.assert_array_equal(
        np.asarray(session.last_batch_decisions, dtype=bool),
        expected,
    )


def test_hooks_fire_for_step_end_and_alarm():
    detector = _build_detector(seed=39)
    on_step_calls = 0
    on_alarm_calls = 0

    def _on_step(_):
        nonlocal on_step_calls
        on_step_calls += 1

    def _on_alarm(_):
        nonlocal on_alarm_calls
        on_alarm_calls += 1

    martingale = PowerMartingale(
        epsilon=0.5,
        alarm_config=AlarmConfig(ville_threshold=0.1),
    )
    session = TemporalSession(
        detector=detector,
        martingale=martingale,
        on_step_end=_on_step,
        on_alarm=_on_alarm,
    )
    rng = np.random.default_rng(40)
    _ = session.step(rng.standard_normal((5, 4)))

    assert on_step_calls == 1
    assert on_alarm_calls == 1


def test_invalid_controller_contract_raises():
    detector = _build_detector()
    with pytest.raises(TypeError, match="test_one"):
        TemporalSession(
            detector=detector,
            online_controller=BadControllerMissingMethod(),  # type: ignore[arg-type]
        )


def test_bad_controller_return_type_raises():
    detector = _build_detector()
    session = TemporalSession(
        detector=detector,
        online_controller=BadControllerReturnType(),  # type: ignore[arg-type]
    )
    rng = np.random.default_rng(41)
    with pytest.raises(TypeError, match="boolean"):
        session.step(rng.standard_normal((3, 4)))


def test_invalid_batch_alpha_raises():
    detector = _build_detector()
    session = TemporalSession(detector=detector)
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="alpha"):
        session.step(rng.standard_normal((4, 4)), apply_batch_select=True, alpha=1.2)


def test_reset_can_reset_online_controller_and_martingale():
    detector = _build_detector(seed=55)
    controller = ThresholdController(threshold=0.2)
    martingale = PowerMartingale(epsilon=0.5)
    session = TemporalSession(
        detector=detector,
        online_controller=controller,
        martingale=martingale,
    )

    rng = np.random.default_rng(56)
    _ = session.step(rng.standard_normal((5, 4)))
    assert martingale.state.step == 5

    session.reset(reset_online_controller=True, reset_martingale=True)
    assert martingale.state.step == 0
    assert controller.reset_calls == 1
