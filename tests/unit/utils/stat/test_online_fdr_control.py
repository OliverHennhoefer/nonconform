import numpy as np
import pytest

from nonconform.fdr import online_false_discovery_control


class ThresholdController:
    def __init__(self, threshold: float = 0.2) -> None:
        self.threshold = threshold
        self.calls: list[float] = []

    def test_one(self, p_value: float) -> bool:
        self.calls.append(float(p_value))
        return p_value <= self.threshold


class BadControllerMissingMethod:
    pass


class BadControllerReturnType:
    def test_one(self, p_value: float) -> str:
        _ = p_value
        return "yes"


def test_online_false_discovery_control_matches_manual_loop():
    p_values = np.array([0.01, 0.3, 0.2, 0.19, 0.8], dtype=float)
    controller = ThresholdController(threshold=0.2)

    decisions = online_false_discovery_control(p_values, controller)

    expected = np.array([True, False, True, True, False], dtype=bool)
    np.testing.assert_array_equal(decisions, expected)
    np.testing.assert_allclose(controller.calls, p_values)


def test_online_false_discovery_control_returns_bool_array():
    p_values = np.array([0.2, 0.5, 0.01], dtype=float)
    decisions = online_false_discovery_control(p_values, ThresholdController(0.2))
    assert decisions.dtype == bool
    assert decisions.shape == p_values.shape


def test_online_false_discovery_control_rejects_invalid_p_values():
    with pytest.raises(ValueError, match="p_values"):
        online_false_discovery_control(
            np.array([0.1, np.nan], dtype=float),
            ThresholdController(),
        )

    with pytest.raises(ValueError, match="within"):
        online_false_discovery_control(
            np.array([0.1, 1.5], dtype=float),
            ThresholdController(),
        )


def test_online_false_discovery_control_rejects_invalid_controller():
    with pytest.raises(TypeError, match="test_one"):
        online_false_discovery_control(
            np.array([0.1, 0.2], dtype=float),
            BadControllerMissingMethod(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="boolean"):
        online_false_discovery_control(
            np.array([0.1, 0.2], dtype=float),
            BadControllerReturnType(),  # type: ignore[arg-type]
        )
