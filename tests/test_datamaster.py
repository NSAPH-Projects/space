import unittest

import pandas as pd

from spacebench.datasets.datamaster import DataMaster


class TestDataMaster(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masterfile = DataMaster()

    def test_list_datasets(self):
        out = self.masterfile.list_datasets()
        assert isinstance(out, list) and len(out) > 0

    def test_getitem(self):
        datasets = self.masterfile.list_datasets()
        out = self.masterfile[datasets[0]]
        assert isinstance(out, pd.Series)


if __name__ == "__main__":
    unittest.main()
