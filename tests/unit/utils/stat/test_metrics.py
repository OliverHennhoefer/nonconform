import numpy as np

from nonconform.utils.stat.metrics import false_discovery_rate, statistical_power


class TestFDRCalculation:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 1])
        y_hat = np.array([0, 0, 1, 1, 1])
        assert false_discovery_rate(y, y_hat) == 0.0

    def test_no_predictions(self):
        y = np.array([0, 0, 1, 1, 1])
        y_hat = np.array([0, 0, 0, 0, 0])
        assert false_discovery_rate(y, y_hat) == 0.0

    def test_all_false_positives(self):
        y = np.array([0, 0, 0, 0, 0])
        y_hat = np.array([1, 1, 1, 1, 1])
        assert false_discovery_rate(y, y_hat) == 1.0

    def test_mixed_predictions(self):
        y = np.array([0, 0, 1, 1])
        y_hat = np.array([1, 0, 1, 0])
        fdr = false_discovery_rate(y, y_hat)
        assert fdr == 0.5

    def test_known_confusion_matrix(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_hat = np.array([1, 1, 0, 1, 1, 0, 0, 0])
        tp = 2
        fp = 2
        expected_fdr = fp / (fp + tp)
        assert false_discovery_rate(y, y_hat) == expected_fdr


class TestStatisticalPower:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 1])
        y_hat = np.array([0, 0, 1, 1, 1])
        assert statistical_power(y, y_hat) == 1.0

    def test_no_detections(self):
        y = np.array([0, 0, 1, 1, 1])
        y_hat = np.array([0, 0, 0, 0, 0])
        assert statistical_power(y, y_hat) == 0.0

    def test_partial_detection(self):
        y = np.array([1, 1, 1, 1, 0, 0])
        y_hat = np.array([1, 1, 0, 0, 0, 0])
        assert statistical_power(y, y_hat) == 0.5

    def test_known_confusion_matrix(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        y_hat = np.array([1, 0, 0, 1, 0, 0])
        tp = 1
        fn = 2
        expected_power = tp / (tp + fn)
        assert statistical_power(y, y_hat) == expected_power


class TestEdgeCases:
    def test_all_zeros_labels(self):
        y = np.array([0, 0, 0, 0])
        y_hat = np.array([0, 0, 0, 0])
        assert false_discovery_rate(y, y_hat) == 0.0
        assert statistical_power(y, y_hat) == 0.0

    def test_all_ones_labels(self):
        y = np.array([1, 1, 1, 1])
        y_hat = np.array([1, 1, 1, 1])
        assert false_discovery_rate(y, y_hat) == 0.0
        assert statistical_power(y, y_hat) == 1.0

    def test_no_true_positives(self):
        y = np.array([0, 0, 0, 0])
        y_hat = np.array([1, 1, 1, 1])
        assert false_discovery_rate(y, y_hat) == 1.0

    def test_no_actual_positives(self):
        y = np.array([0, 0, 0, 0])
        y_hat = np.array([0, 0, 0, 0])
        assert statistical_power(y, y_hat) == 0.0

    def test_single_element_arrays(self):
        y = np.array([1])
        y_hat = np.array([1])
        assert false_discovery_rate(y, y_hat) == 0.0
        assert statistical_power(y, y_hat) == 1.0


class TestMathematicalCorrectness:
    def test_fdr_formula_verification(self):
        y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_hat = np.array([1, 0, 1, 0, 1, 1, 0, 0])

        tp = np.sum((y == 1) & (y_hat == 1))
        fp = np.sum((y == 0) & (y_hat == 1))

        expected_fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        assert false_discovery_rate(y, y_hat) == expected_fdr

    def test_power_formula_verification(self):
        y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_hat = np.array([1, 0, 1, 0, 1, 1, 0, 0])

        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))

        expected_power = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert statistical_power(y, y_hat) == expected_power

    def test_complementary_relationship(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_hat = np.array([1, 1, 0, 0, 0, 0, 0, 0])

        power = statistical_power(y, y_hat)
        fn_rate = 1.0 - power

        tp = np.sum((y == 1) & (y_hat == 1))
        fn = np.sum((y == 1) & (y_hat == 0))

        assert fn_rate == fn / (tp + fn)


class TestDataTypes:
    def test_integer_arrays(self):
        y = np.array([0, 1, 1, 0], dtype=int)
        y_hat = np.array([0, 1, 0, 1], dtype=int)
        fdr = false_discovery_rate(y, y_hat)
        power = statistical_power(y, y_hat)
        assert isinstance(fdr, float)
        assert isinstance(power, float)

    def test_boolean_arrays(self):
        y = np.array([False, True, True, False])
        y_hat = np.array([False, True, False, True])
        fdr = false_discovery_rate(y, y_hat)
        power = statistical_power(y, y_hat)
        assert isinstance(fdr, float)
        assert isinstance(power, float)
