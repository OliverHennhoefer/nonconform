import numpy as np
import pytest

from nonconform._internal.conditional_calibration import calibrate_conditional_p_values
from nonconform.scoring import ConditionalEmpirical


def test_conditional_empirical_rejects_weighted_inputs() -> None:
    estimation = ConditionalEmpirical(method="dkwm")
    with pytest.raises(ValueError, match="does not support weighted p-values"):
        estimation.compute_p_values(
            np.array([0.1, 0.2]),
            np.array([0.0, 0.3, 0.7]),
            weights=(np.ones(3), np.ones(2)),
        )


def test_conditional_empirical_accepts_empty_scores() -> None:
    estimation = ConditionalEmpirical(method="simes")
    output = estimation.compute_p_values(np.array([]), np.array([0.1, 0.2, 0.3]))
    assert output.shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": -0.1}, "delta must be in"),
        ({"delta": 1.0}, "delta must be in"),
        ({"method": "unknown"}, "must be one of"),
        ({"simes_kden": 0}, "simes_kden must be a positive integer"),
        ({"mc_num_simulations": 10}, "mc_num_simulations must be an integer >= 100"),
    ],
)
def test_conditional_empirical_invalid_init_raises(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ConditionalEmpirical(**kwargs)


def test_conditional_empirical_small_n_falls_back_to_dkwm_for_mc() -> None:
    scores = np.array([0.8, 0.1, -0.4])
    calibration = np.array([0.2, 0.0, -0.1, 0.7, -0.8])

    mc = ConditionalEmpirical(method="mc", tie_break="classical", delta=0.2)
    dkwm = ConditionalEmpirical(method="dkwm", tie_break="classical", delta=0.2)

    mc_values = mc.compute_p_values(scores, calibration)
    dkwm_values = dkwm.compute_p_values(scores, calibration)
    np.testing.assert_allclose(mc_values, dkwm_values, rtol=0.0, atol=1e-12)


def test_conditional_empirical_small_n_falls_back_to_dkwm_for_asymptotic() -> None:
    scores = np.array([0.8, 0.1, -0.4])
    calibration = np.array([0.2, 0.0, -0.1, 0.7, -0.8])

    asymptotic = ConditionalEmpirical(
        method="asymptotic",
        tie_break="classical",
        delta=0.2,
    )
    dkwm = ConditionalEmpirical(method="dkwm", tie_break="classical", delta=0.2)

    asym_values = asymptotic.compute_p_values(scores, calibration)
    dkwm_values = dkwm.compute_p_values(scores, calibration)
    np.testing.assert_allclose(asym_values, dkwm_values, rtol=0.0, atol=1e-12)


def test_conditional_calibration_accepts_numpy_integer_n_calibration() -> None:
    p_values = np.array([0.2, 0.8], dtype=float)
    calibrated, _ = calibrate_conditional_p_values(
        p_values,
        n_calibration=np.int64(10),
        delta=0.1,
        method="dkwm",
    )
    assert calibrated.shape == p_values.shape
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_conditional_calibration_rejects_boolean_n_calibration() -> None:
    p_values = np.array([0.2, 0.8], dtype=float)
    with pytest.raises(TypeError, match="n_calibration must be an integer"):
        calibrate_conditional_p_values(
            p_values,
            n_calibration=True,
            delta=0.1,
            method="dkwm",
        )
