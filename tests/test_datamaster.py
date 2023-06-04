import unittest

import pandas as pd

from spacebench.datamaster import DataMaster


class TestDataMaster(unittest.TestCase):
    def setUp(self) -> None:
        self.masterfile = DataMaster()

    def test_list_datasets(self):
        out = self.masterfile.list_datasets()
        assert isinstance(out, list), "Output should be a list."
        assert len(out) > 0, "Output should not be empty."

    def test_getitem(self):
        datasets = self.masterfile.list_datasets()
        out = self.masterfile[datasets[0]]
        assert isinstance(out, pd.Series), "Output should be a pandas Series."

    def test_getitem_non_existing(self):
        assert self.masterfile["non_existing_dataset"] is None


if __name__ == "__main__":
    unittest.main()
