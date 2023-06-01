from importlib import resources

import pandas as pd

import spacebench
from spacebench.log import LOGGER


class DataMaster:
    """Class for managing the masterfile and collections metadata

    Parameters
    ----------

    masterfile: pd.DataFrame 
        A dataframe with metadata about available datasets.
    collections: pd.DataFrame
        A dataframe with information about the collections
        where the datasets are generated from.
    """

    def __init__(self):
        try:
            with resources.open_text(spacebench, "masterfile.csv") as io:
                self.master = pd.read_csv(io, index_col=0)
        except FileNotFoundError:
            LOGGER.error("Masterfile not found.")
            raise FileNotFoundError(("The masterfile.csv is not present in the "
                                     "expected directory. Please ensure the " 
                                     "file is correctly placed."))

    def list_datasets(self) -> list[str]:
        """
        Returns a list of names of available datasets.

        Returns
        -------
            
        list[str] 
            Names of all available datasets.
        """
        return self.master.index.tolist() 

    def __getitem__(self, key: str) -> pd.Series:
        """
        Retrieves the row corresponding to the provided dataset key from the masterfile.
        
        Parameters
        ----------
        key : str
            The identifier for the dataset.

        Returns
        -------
        pd.Series or None
            The corresponding dataset row if found, else None.
        """
        try:
            return self.master.loc[key]
        except KeyError:
            LOGGER.error(f"Dataset {key} not found in masterfile.")
            return None
