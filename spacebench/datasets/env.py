"""Module for defining the SpaceEnvironment class"""
import os
from typing import Literal

import pandas as pd
import networkx as nx
import geopandas as gpd

from pyDataverse.api import NativeApi, DataAccessApi
from spacebench.datasets.datamaster import DataMaster
from spacebench.log import LOGGER
from spacebench.datasets.datasets import SpatialMetadata, CausalDataset


class SpaceEnv:
    """Main class for generating new datasets.
    It is loosely based on the gym environment class."""

    def __init__(self, name: str, path: str = "downloads"):
        self.name = name
        self.datamaster = DataMaster()

        if name not in self.datamaster.list_datasets():
            raise ValueError(f"Dataset {name} not available")

        self.info = self.datamaster[name]
        self.data = {}

        # download data, metadata, graph, and geojson
        for k in ["data", "metadata", "graph", "geodata"]:
            filename_or_url = self.info[k]
            filename = download_dataset(filename_or_url, path, self.info.source)
            self.data[k] = read_data(filename, k, path)

    def make(self) -> CausalDataset:
        # TODO: connnect this to the API to sample/and mask a dataset
        # it must return the causal dataset using the DatasetGenerator.make_dataset method
        # it is possible that we don't need separate classes for datasetgenerator?
        # but maybe we can start like to reuse the code from the old version
        # the function must return the CausalDataset which is a prototype.
        # in the future should be replaced in favor of a dataloader class
        # the causaldataset class MUST be extended to contain information about the complexity
        # of the dataset, the number of samples, the number of features, etc..
        raise NotImplementedError


def download_dataset(
    filename_or_url: str | None, path: str, source: Literal["dataverse"]
) -> str:
    """Download a file from a url to a path.

    Args:
        id_or_url (str | None): download identifier
        path (str): directory to download to
        source (Literal['dataverse']): source of the data. Currently only dataverse is supported.

    Returns:
        str: filename. If url is None, returns None and does nothiing.
    """
    if filename_or_url is None:
        return None

    os.makedirs(path, exist_ok=True)

    if source == "dataverse":
        # This is a minimal API to download data from dataverse
        base_url = "https://dataverse.harvard.edu/"
        doi = "doi:10.7910/DVN/SYNPBS"
        api = NativeApi(base_url)
        data_api = DataAccessApi(base_url)
        dataset = api.get_dataset(doi)
        files = dataset.json()["data"]["latestVersion"]["files"]
        file2id = {file["label"]: file["dataFile"]["id"] for file in files}
        response = data_api.get_datafile(file2id[filename_or_url])
        dest = os.path.join(path, filename_or_url)
        if not os.path.exists(dest):
            with open(dest, mode="wb") as io:
                io.write(response.content)
        filename = filename_or_url
    else:
        # TODO: we can support other sources in the future, such
        # as directly downloading from a URL outside dataverse
        raise NotImplementedError(f"Invalid source {source}")

    return filename


def read_data(
    filename: str | None,
    datatype: Literal["data", "metadata", "graph", "geojson"],
    path: str = "downloads",
) -> pd.DataFrame:
    """Read data from a file.

    Args:
        filename (str | None): 
        datatype (Literal["data", "metadata", "graph", "geojson"]): One of possible datatypes. 
            The datatype determines the file extension and the function used to read the file.
        path (str, optional): _description_. Defaults to "downloads".

    Returns:
        Any: Data loaded in the appropriate format. Returns None if filename is None.
    """ """"""
    if filename is None:
        return None

    extension = os.path.splitext(filename)[1]

    match datatype:
        case "data":
            if extension == ".csv":
                return pd.read_csv(os.path.join(path, filename))
            else:
                raise ValueError(
                    f"Invalid extension {extension} for datatype {datatype}"
                )
        case "metadata":
            if extension == ".json":
                return SpatialMetadata.from_json(os.path.join(path, filename))
            else:
                raise ValueError(
                    f"Invalid extension {extension} for datatype {datatype}"
                )
        case "graph":
            if extension == ".graphml":
                return nx.read_graphml(os.path.join(path, filename))
            else:
                raise ValueError(
                    f"Invalid extension {extension} for datatype {datatype}"
                )
        case "geodata":
            if extension == ".geojson":
                return gpd.GeoDataFrame.from_file(os.path.join(path, filename))
            else:
                raise ValueError(
                    f"Invalid extension {extension} for datatype {datatype}"
                )
        case _:
            raise ValueError(f"Invalid datatype {datatype}")


if __name__ == "__main__":
    # small test
    # TODO: convert in unit test
    dm = DataMaster()
    generator = SpaceEnv(dm.list_datasets()[0])
    datasets = [generator.make() for _ in range(10)]
    LOGGER.debug("ok")
