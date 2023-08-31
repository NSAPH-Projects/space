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

    Examples
    --------

    >>> from spacebench.datamaster import DataMaster
    >>> dm = DataMaster()
    >>> print(dm)
    Available datasets (total: 11):
    <BLANKLINE>
      healthd_dmgrcs_mortality_disc
      cdcsvi_limteng_hburdic_cont
      climate_relhum_wfsmoke_cont
      climate_wfsmoke_minrty_disc
      healthd_hhinco_mortality_cont
      ...
      county_educatn_election_cont
      county_phyactiv_lifexpcy_cont
      county_dmgrcs_election_disc
      cdcsvi_nohsdp_poverty_cont
      cdcsvi_nohsdp_poverty_disc

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

    def list_envs(
        self,
        binary: bool | None = None,
        continuous: bool | None = None,
        collection: str | None = None,
    ) -> list[str]:
        """
        Returns a list of names of available envs.

        Arguments
            binary : bool, optional. If True, only binary envs are returned.
            continuous : bool, optional. If True, only continuous envs are
            returned.
            collection : str, optional. If provided, only envs from the given
            collection are returned.

        Returns
           list[str]:  Names of all available datasets.
        """
        master = self.master
        index = np.zeros(master.shape[0], dtype=bool)
        if binary is None and continuous is None:
            return master.index.to_list()

        if binary is not None:
            index[master.treatment_type == "binary"] = binary

        if continuous is not None:
            index[master.treatment_type == "continuous"] = continuous

        if collection is not None:
            index[master.collection != collection] = False

        return master.index[index].to_list()
    
    def list_collections(self) -> list[str]:
        """
        Returns a list of names of available collections.

        Returns
           list[str]:  Names of all available collections.
        """
        return self.master.collection.unique().tolist()

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

    def __repr__(self) -> str:
        datasets = self.list_envs()
        if len(datasets) > 10:
            datasets_str = "\n  ".join(datasets[:5] + ["..."] + datasets[-5:])
        else:
            datasets_str = "\n  ".join(datasets)

        return f"Available datasets (total: " f"{len(datasets)}):\n\n  {datasets_str}"
    