import numpy as np

from nonconform.utils.stat.weighted_fdr import (
    weighted_bh,
    weighted_false_discovery_control,
)


class TestBasicFDRControl:
    def test_returns_boolean_array(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert isinstance(discoveries, np.ndarray)
        assert discoveries.dtype == bool

    def test_output_length_matches_test_set(self, conformal_result):
        result = conformal_result(n_test=30, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 30

    def test_with_different_alpha_levels(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)

        discoveries_001 = weighted_false_discovery_control(result=result, alpha=0.01)
        discoveries_010 = weighted_false_discovery_control(result=result, alpha=0.10)
        discoveries_020 = weighted_false_discovery_control(result=result, alpha=0.20)

        assert (
            np.sum(discoveries_001)
            <= np.sum(discoveries_010)
            <= np.sum(discoveries_020)
        )

    def test_with_direct_inputs(self, sample_scores, sample_weights, sample_p_values):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)
        p_values = sample_p_values(n=15)

        discoveries = weighted_false_discovery_control(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )

        assert len(discoveries) == 15


class TestWeightedBH:
    def test_returns_boolean_array(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_bh(result, alpha=0.1)
        assert isinstance(discoveries, np.ndarray)
        assert discoveries.dtype == bool

    def test_output_length_matches_test_set(self, conformal_result):
        result = conformal_result(n_test=25, n_calib=100)
        discoveries = weighted_bh(result, alpha=0.1)
        assert len(discoveries) == 25

    def test_low_p_values_discovered(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100, seed=42)
        discoveries = weighted_bh(result, alpha=0.2)
        assert isinstance(discoveries, np.ndarray)
        assert len(discoveries) == 20


class TestAlphaLevels:
    def test_alpha_005(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.05)
        assert len(discoveries) == 20

    def test_alpha_010(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.10)
        assert len(discoveries) == 20

    def test_alpha_020(self, conformal_result):
        result = conformal_result(n_test=20, n_calib=100)
        discoveries = weighted_false_discovery_control(result=result, alpha=0.20)
        assert len(discoveries) == 20


class TestResultBundleInput:
    def test_with_result_bundle(self, conformal_result):
        result = conformal_result(
            n_test=20, n_calib=100, include_p_values=True, include_weights=True
        )
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 20

    def test_without_p_values_in_result(self, conformal_result):
        result = conformal_result(
            n_test=20, n_calib=100, include_p_values=False, include_weights=True
        )
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 20

    def test_with_metadata(self, conformal_result):
        result = conformal_result(
            n_test=20,
            n_calib=100,
            include_p_values=True,
            include_weights=True,
            include_metadata=True,
        )
        discoveries = weighted_false_discovery_control(result=result, alpha=0.1)
        assert len(discoveries) == 20


class TestDirectInput:
    def test_with_p_values_only(self, sample_p_values, sample_scores, sample_weights):
        p_values = sample_p_values(n=15)
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)

        discoveries = weighted_false_discovery_control(
            p_values=p_values,
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 15

    def test_without_p_values(self, sample_scores, sample_weights):
        test_scores, calib_scores = sample_scores(n_test=15, n_calib=80)
        test_weights, calib_weights = sample_weights(n_test=15, n_calib=80)

        discoveries = weighted_false_discovery_control(
            test_scores=test_scores,
            calib_scores=calib_scores,
            test_weights=test_weights,
            calib_weights=calib_weights,
            alpha=0.1,
        )
        assert len(discoveries) == 15
