import unittest

from nonconform.utils.stat.statistical import calculate_p_val


class TestUtilsStatistical(unittest.TestCase):
    def test_calculate_p_val_with_no_score(self):
        score = []
        calib = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        p_val = calculate_p_val(score, calib)

        self.assertEqual(p_val, [])

    def test_calculate_p_val_with_one_score(self):
        score = [0.95]
        calib = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        p_val = calculate_p_val(score, calib)

        self.assertEqual(p_val, [(1 + 1) / (10 + 1)])

    def test_calculate_p_val_with_two_scores(self):
        score = [0.45, 0.95]
        calib = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        p_val = calculate_p_val(score, calib)

        self.assertEqual(p_val, [(6 + 1) / (10 + 1), (1 + 1) / (10 + 1)])


if __name__ == "__main__":
    unittest.main()
