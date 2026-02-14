import numpy as np
import pytest

from nonconform.fdr import (
    weighted_bh,
    weighted_bh_from_result,
    weighted_false_discovery_control,
    weighted_false_discovery_control_empirical,
    weighted_false_discovery_control_from_arrays,
)


class TestNoDiscoveries:
    def test_all_high_p_values(self, conformal_result):
        result = conformal_result(n_test=4, n_calib=50, seed=42)
        result.p_values = np.array([0.9, 0.95, 0.99, 0.999])
        discoveries = weighted_bh_from_result(result=result, alpha=0.05)
        assert np.sum(discoveries) == 0

    def test_no_significant_results(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=99)
        result.p_values = np.ones(20) * 0.9
        discoveries = weighted_false_discovery_control(result=result, alpha=0.01)
        assert isinstance(discoveries, np.ndarray)


class TestAllDiscoveries:
    def test_all_low_p_values(self, conformal_result):
        result = conformal_result(n_test=10, n_calib=100, seed=42)
        discoveries = weighted_bh_from_result(result=result, alpha=0.5)
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 10

    def test_very_permissive_alpha(self, conformal_result):
        result = conformal_result(n_test=4, n_calib=50, seed=42)
        result.p_values = np.array([0.1, 0.2, 0.3, 0.4])
        discoveries = weighted_bh_from_result(result=result, alpha=0.5)
        assert np.sum(discoveries) > 0


class TestSingleTestPoint:
    def test_single_test_point_low_p_value(self):
        p_values = np.array([0.01])
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        test_weights = np.array([1.0])
        calib_weights = np.array([1.0, 1.0, 1.0])

        discoveries = weighted_false_discovery_control_from_arrays(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.05,
        )
        assert len(discoveries) == 1

    def test_single_test_point_high_p_value(self, conformal_result):
        result = conformal_result(n_test=1, n_calib=50, seed=42)
        discoveries = weighted_bh_from_result(result=result, alpha=0.01)
        assert len(discoveries) == 1
        assert isinstance(discoveries[0], bool | np.bool_)


class TestAlphaBoundaries:
    def test_alpha_near_zero(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.001)
        assert len(discoveries) == 20

    def test_alpha_near_one(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.999)
        assert len(discoveries) == 20

    def test_invalid_alpha_zero(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        with pytest.raises(ValueError):
            weighted_false_discovery_control(result=result, alpha=0.0)

    def test_invalid_alpha_one(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        with pytest.raises(ValueError):
            weighted_false_discovery_control(result=result, alpha=1.0)

    def test_invalid_alpha_negative(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        with pytest.raises(ValueError):
            weighted_false_discovery_control(result=result, alpha=-0.1)

    def test_invalid_alpha_greater_than_one(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        with pytest.raises(ValueError):
            weighted_false_discovery_control(result=result, alpha=1.5)


class TestErrorHandling:
    def test_missing_required_inputs(self):
        with pytest.raises(TypeError):
            weighted_false_discovery_control(alpha=0.1)  # type: ignore[call-arg]

    def test_bh_missing_required_inputs(self):
        with pytest.raises(TypeError):
            weighted_bh(alpha=0.1)  # type: ignore[call-arg]

    def test_missing_test_scores(self, sample_scores, sample_weights, sample_p_values):
        _, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights, calib_weights = sample_weights(n_test=10, n_calib=50)
        p_values = sample_p_values(n=10)

        with pytest.raises(TypeError):
            weighted_false_discovery_control_from_arrays(
                p_values=p_values,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )  # type: ignore[call-arg]

    def test_missing_weights_with_scores(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)

        with pytest.raises(TypeError):
            weighted_false_discovery_control_empirical(
                test_scores=test_scores,
                calib_scores=calib_scores,
                alpha=0.1,
            )  # type: ignore[call-arg]


class TestInvalidWeights:
    @pytest.mark.parametrize("bad_value", [-0.1, np.nan, np.inf, -np.inf])
    def test_invalid_test_weights(self, sample_scores, bad_value):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.ones(10)
        test_weights[0] = bad_value
        calib_weights = np.ones(50)

        with pytest.raises(ValueError):
            weighted_false_discovery_control_empirical(
                test_scores=test_scores,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )

    @pytest.mark.parametrize("bad_value", [-0.1, np.nan, np.inf, -np.inf])
    def test_invalid_calib_weights(self, sample_scores, bad_value):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.ones(10)
        calib_weights = np.ones(50)
        calib_weights[0] = bad_value

        with pytest.raises(ValueError):
            weighted_false_discovery_control_empirical(
                test_scores=test_scores,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )


class TestExtremeWeights:
    def test_zero_weights(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.zeros(10)
        calib_weights = np.zeros(50)

        with pytest.raises(
            ValueError, match="calib_weights must sum to a positive value"
        ):
            weighted_false_discovery_control_empirical(
                test_scores=test_scores,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )

    def test_very_large_weights(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.ones(10) * 1e6
        calib_weights = np.ones(50) * 1e6

        discoveries = weighted_false_discovery_control_empirical(
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 10


class TestInvalidPValues:
    @pytest.mark.parametrize("bad_value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_invalid_p_values_wcs(self, bad_value):
        p_values = np.array([bad_value, 0.5])
        test_scores = np.array([1.0, 2.0])
        calib_scores = np.array([0.0, 1.0, 2.0])
        test_weights = np.ones(2)
        calib_weights = np.ones(3)

        with pytest.raises(ValueError):
            weighted_false_discovery_control_from_arrays(
                p_values=p_values,
                test_scores=test_scores,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_invalid_p_values_bh(self, conformal_result, bad_value):
        result = conformal_result(n_test=2, n_calib=10, seed=42)
        result.p_values = np.array([bad_value, 0.5])

        with pytest.raises(ValueError):
            weighted_bh_from_result(result=result, alpha=0.1)

    def test_invalid_p_values_shape_bh(self):
        with pytest.raises(ValueError):
            weighted_bh(np.array([[0.1], [0.2]]), alpha=0.1)

    def test_invalid_p_values_shape_wcs(self):
        test_scores = np.array([1.0, 2.0])
        calib_scores = np.array([0.0, 1.0, 2.0])
        test_weights = np.array([1.0, 1.0])
        calib_weights = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="p_values must be a 1D array"):
            weighted_false_discovery_control_from_arrays(
                p_values=np.array([[0.1], [0.2]]),
                test_scores=test_scores,
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )


class TestKDEMetadataValidation:
    def test_missing_kde_keys_raise(self, conformal_result):
        result = conformal_result(n_test=5, n_calib=20)
        result.metadata = {"kde": {"eval_grid": np.array([0.0, 1.0])}}
        with pytest.raises(ValueError, match="missing keys"):
            weighted_false_discovery_control(result=result, alpha=0.1)

    def test_non_increasing_eval_grid_raises(self, conformal_result):
        result = conformal_result(n_test=5, n_calib=20)
        result.metadata = {
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 1.0]),
                "cdf_values": np.array([0.0, 0.5, 1.0]),
                "total_weight": 10.0,
            }
        }
        with pytest.raises(ValueError, match="strictly increasing"):
            weighted_false_discovery_control(result=result, alpha=0.1)

    def test_non_positive_kde_total_weight_raises(self, conformal_result):
        result = conformal_result(n_test=5, n_calib=20)
        result.metadata = {
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 2.0]),
                "cdf_values": np.array([0.0, 0.5, 1.0]),
                "total_weight": 0.0,
            }
        }
        with pytest.raises(ValueError, match="finite positive value"):
            weighted_false_discovery_control(result=result, alpha=0.1)

    def test_tiny_cdf_jitter_is_tolerated(self, conformal_result):
        result = conformal_result(n_test=5, n_calib=20)
        result.metadata = {
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 2.0]),
                "cdf_values": np.array([0.0, 0.5, 0.5 - 5e-13]),
                "total_weight": 10.0,
            }
        }
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert discoveries.shape == (5,)

    def test_large_cdf_decrease_still_raises(self, conformal_result):
        result = conformal_result(n_test=5, n_calib=20)
        result.metadata = {
            "kde": {
                "eval_grid": np.array([0.0, 1.0, 2.0]),
                "cdf_values": np.array([0.0, 0.5, 0.49]),
                "total_weight": 10.0,
            }
        }
        with pytest.raises(ValueError, match="non-decreasing"):
            weighted_false_discovery_control(result=result, alpha=0.1)


class TestOutputValidation:
    def test_output_is_boolean(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert discoveries.dtype == bool

    def test_output_length_correct(self, conformal_result):
        for n_test in [5, 10, 20, 50]:
            result = conformal_result(n_test=n_test, n_calib=100)
            discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
            assert len(discoveries) == n_test
