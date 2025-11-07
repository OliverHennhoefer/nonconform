import numpy as np
import pytest

from nonconform.utils.stat.weighted_fdr import (
    weighted_bh,
    weighted_false_discovery_control,
)


class TestNoDiscoveries:
    def test_all_high_p_values(self, conformal_result):
        result = conformal_result(n_test=4, n_calib=50, seed=42)
        result.p_values = np.array([0.9, 0.95, 0.99, 0.999])
        discoveries = weighted_bh(result, alpha=0.05)
        assert np.sum(discoveries) == 0

    def test_no_significant_results(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=99)
        result.p_values = np.ones(20) * 0.9
        discoveries = weighted_false_discovery_control(result=result, alpha=0.01)
        assert isinstance(discoveries, np.ndarray)


class TestAllDiscoveries:
    def test_all_low_p_values(self, conformal_result):
        result = conformal_result(n_test=10, n_calib=100, seed=42)
        discoveries = weighted_bh(result, alpha=0.5)
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 10

    def test_very_permissive_alpha(self, conformal_result):
        result = conformal_result(n_test=4, n_calib=50, seed=42)
        result.p_values = np.array([0.1, 0.2, 0.3, 0.4])
        discoveries = weighted_bh(result, alpha=0.5)
        assert np.sum(discoveries) > 0


class TestSingleTestPoint:
    def test_single_test_point_low_p_value(self):
        p_values = np.array([0.01])
        test_scores = np.array([5.0])
        calib_scores = np.array([1.0, 2.0, 3.0])
        test_weights = np.array([1.0])
        calib_weights = np.array([1.0, 1.0, 1.0])

        discoveries = weighted_false_discovery_control(
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
        discoveries = weighted_bh(result, alpha=0.01)
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
        with pytest.raises(ValueError):
            weighted_false_discovery_control(alpha=0.1)

    def test_missing_test_scores(self, sample_scores, sample_weights):
        _, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights, calib_weights = sample_weights(n_test=10, n_calib=50)

        with pytest.raises(ValueError):
            weighted_false_discovery_control(
                calib_scores=calib_scores,
                test_weights=test_weights,
                calib_weights=calib_weights,
                alpha=0.1,
            )

    def test_missing_weights_with_scores(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)

        with pytest.raises(ValueError):
            weighted_false_discovery_control(
                test_scores=test_scores, calib_scores=calib_scores, alpha=0.1
            )


class TestExtremeWeights:
    def test_zero_weights(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.zeros(10)
        calib_weights = np.zeros(50)

        discoveries = weighted_false_discovery_control(
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 10

    def test_very_large_weights(self, sample_scores):
        test_scores, calib_scores = sample_scores(n_test=10, n_calib=50)
        test_weights = np.ones(10) * 1e6
        calib_weights = np.ones(50) * 1e6

        discoveries = weighted_false_discovery_control(
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 10


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
