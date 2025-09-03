import unittest

import numpy as np

from nonconform.utils.stat.statistical import calculate_p_val


class TestUtilsStatistical(unittest.TestCase):
    def test_calculate_p_val_with_no_score(self):
        score = np.array([])
        calib = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        p_val = calculate_p_val(score, calib)

        # Empty array should return empty array
        self.assertEqual(len(p_val), 0)
        self.assertIsInstance(p_val, np.ndarray)

    def test_calculate_p_val_with_one_score(self):
        score = np.array([0.95])
        calib = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        p_val = calculate_p_val(score, calib)

        expected = np.array([(1 + 1) / (10 + 1)])
        np.testing.assert_array_almost_equal(p_val, expected)

    def test_calculate_p_val_with_two_scores(self):
        score = np.array([0.45, 0.95])
        calib = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        p_val = calculate_p_val(score, calib)

        expected = np.array([(6 + 1) / (10 + 1), (1 + 1) / (10 + 1)])
        np.testing.assert_array_almost_equal(p_val, expected)


if __name__ == "__main__":
    unittest.main()
