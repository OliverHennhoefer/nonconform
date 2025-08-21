import unittest

from nonconform.utils.func.enums import Aggregation
from nonconform.utils.stat.aggregation import aggregate


class TestUtilsAggregation(unittest.TestCase):
    def test_aggregate_median(self):
        scores = [0.1, 0.15, 0.2, 0.25, 0.9, 10]
        res = aggregate(Aggregation.MEDIAN, scores)

        self.assertEqual(res, 0.225)

    def test_aggregate_mean(self):
        scores = [0.1, 0.15, 0.2, 0.25, 0.9, 2]
        res = aggregate(Aggregation.MEAN, scores)

        self.assertEqual(res, 0.6)

    def test_aggregate_minimum(self):
        scores = [0.1, 0.15, 0.2, 0.25]
        res = aggregate(Aggregation.MINIMUM, scores)

        self.assertEqual(res, 0.1)

    def test_aggregate_maximum(self):
        scores = [0.1, 0.15, 0.2, 0.25]
        res = aggregate(Aggregation.MAXIMUM, scores)

        self.assertEqual(res, 0.25)


if __name__ == "__main__":
    unittest.main()
