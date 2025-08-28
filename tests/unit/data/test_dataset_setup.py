import unittest

import pandas as pd
from pandas._testing import assert_frame_equal, assert_series_equal

from nonconform.utils.data import Dataset, load


class TestDatasetSetup(unittest.TestCase):
    def test_dataset_setup_wbc(self):
        x_train, x_test, y_test = load(Dataset.WBC, setup=True, seed=1)

        data = {
            "V1": [3.0, 3.0, 1.0],
            "V2": [2.0, 1.0, 3.0],
            "V3": [2.0, 1.0, 1.0],
            "V4": [1.0, 1.0, 1.0],
            "V5": [4.0, 2.0, 2.0],
            "V6": [3.0, 1.0, 1.0],
            "V7": [2.0, 2.0, 2.0],
            "V8": [1.0, 1.0, 2.0],
            "V9": [1.0, 1.0, 1.0],
        }

        index = [113, 25, 143]
        x_train_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(x_train.iloc[[3, 50, 100], :], x_train_ref)

        data = {
            "V1": [1.0, 3.0, 5.0],
            "V2": [1.0, 1.0, 8.0],
            "V3": [1.0, 1.0, 9.0],
            "V4": [1.0, 1.0, 4.0],
            "V5": [1.0, 2.0, 3.0],
            "V6": [1.0, 1.0, 10.0],
            "V7": [1.0, 2.0, 7.0],
            "V8": [1.0, 2.0, 1.0],
            "V9": [1.0, 1.0, 1.0],
        }

        index = [3, 22, 33]
        x_test_ref = pd.DataFrame(data, index=index)
        assert_frame_equal(x_test.iloc[[3, 22, 33], :], x_test_ref)

        data = [0, 1, 1, 1]
        index = [31, 32, 33, 34]
        y_test_ref = pd.Series(data, index=index, dtype="int32", name="Class")
        assert_series_equal(y_test.iloc[31:], y_test_ref)


if __name__ == "__main__":
    unittest.main()
