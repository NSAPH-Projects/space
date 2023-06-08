from importlib import resources

import numpy as np
import pandas as pd

import spacebench
from spacebench.log import LOGGER


class DataMaster:
    """
    Class for managing the masterfile and collections metadata

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
            raise FileNotFoundError(
                (
                    "The masterfile.csv is not present in the "
                    "expected directory. Please ensure the "
                    "file is correctly placed."
                )
            )

    def list_datasets(
        self, binary: bool | None = None, continuous: bool | None = None
    ) -> list[str]:
        """
        Returns a list of names of available datasets.

        Arguments
            binary : bool, optional. If True, only binary datasets are returned.
            continuous : bool, optional. If True, only continuous datasets are 
            returned.

        Returns
           list[str]:  Names of all available datasets.
        """
        master = self.master
        index = np.zeros(master.shape[0], dtype=bool)
        if binary is None and continuous is None:
            return master.index.to_list()
        
        if binary is not None:
            index[master.treatment_type == "binary"] = True

        if continuous is not None:
            index[master.treatment_type == "continuous"] = True
        
        return master.index[index].to_list()


    def __getitem__(self, key: str) -> pd.Series:
        """
        Retrieves the row corresponding to the provided dataset key from the 
        masterfile.

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
