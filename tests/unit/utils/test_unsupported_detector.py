import unittest

from nonconform.estimation import ConformalDetector
from nonconform.strategy.split import Split
from pyod.models.cblof import CBLOF


class TestUnsupportedDetector(unittest.TestCase):
    def test_unsupported_detector(self):
        with self.assertRaises(ValueError) as _:
            ConformalDetector(detector=CBLOF(n_clusters=2), strategy=Split())


if __name__ == "__main__":
    unittest.main()
