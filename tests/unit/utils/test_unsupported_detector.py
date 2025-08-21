import unittest

from nonconform.estimation.standard_conformal import StandardConformalDetector
from nonconform.strategy.split import Split
from pyod.models.cblof import CBLOF


class TestUnsupportedDetector(unittest.TestCase):
    def test_unsupported_detector(self):
        with self.assertRaises(ValueError) as _:
            StandardConformalDetector(detector=CBLOF(n_clusters=2), strategy=Split())


if __name__ == "__main__":
    unittest.main()
