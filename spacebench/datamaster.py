from importlib import resources

import pandas as pd

import spacebench


class DataMaster:
    """Class for managing the masterfile and collections metadata

    Attributes:
        masterfile (pd.DataFrame): dataframe with metadata about available datasets
        collections (pd.DataFrame): dataframe with information about the collections
                                    where the datasets are generated from
    """

    def __init__(self):
        with resources.open_text(spacebench, "masterfile.csv") as io:
            self.master = pd.read_csv(io, index_col=0)


    def list_datasets(self) -> list[str]:
        """Returns a list of available datasets

        Returns:
            list[str]: A list of available datasets
        """
        return self.master.index.tolist() 

    def __getitem__(self, key: str) -> pd.Series:
        """Returns the row of the masterfile corresponding to the dataset"""
        return self.master.loc[key]
