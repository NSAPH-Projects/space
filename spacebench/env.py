"""Module for defining the SpaceEnvironment class"""
from dataclasses import dataclass
import os
import zipfile
import yaml

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse

from spacebench.api.dataverse import DataverseAPI
from spacebench.datamaster import DataMaster
from spacebench.log import LOGGER


@dataclass
class SpaceDataset:
    """Class for storing a spatial causal inference benchmark dataset"""

    treatment: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray
    edges: list[tuple[int, int]]
    treatment_values: np.ndarray
    smoothness_of_missing: float | None = None
    confounding_of_missing: float | None = None
    counterfactuals: np.ndarray | None = None
    coordinates: np.ndarray | None = None

    def has_binary_treatment(self) -> bool:
        """Returns true if treatment is binary"""
        return len(self.treatment_values) == 2
    
    def erf(self) -> np.ndarray:
        """Returns the exposure-response function, also known 
        in the literature as the average dose-response function
        
        Returns:
            np.ndarray: The exposure-response function"""
        return self.counterfactuals.mean(0)

    def adjacency_matrix(
        self, sparse: bool = False
    ) -> np.ndarray | scipy.sparse.csr_matrix:
        """Returns the adjacency matrix of the graph

        Args:
            sparse (bool, optional): If True, returns a sparse matrix
                of type csr_matrix. Defaults to False.

        Returns:
            np.ndarray | scipy.sparse.csr_matrix: Adjacency matrix where
                entry (i, j) is 1 if there is an edge between node i and node j.
        """
        n = len(self.treatment)
        if sparse:
            adj = scipy.sparse.csr_matrix((n, n))
        else:
            adj = np.zeros((n, n))
        for e in self.edges:
            adj[e[0], e[1]] = 1
            adj[e[1], e[0]] = 1
        return adj


class SpaceEnv:
    """Class for a SpaCE environment. It holdss the data and metadata that is
    used to generate the datasets by masking a covariate, which becomes a
    missing confounder."""

    def __init__(self, name: str, dir: str | None = None):
        """Initializes the SpaceEnv class using a dataset name. When the dataset
        is not found in the directory, it is downloaded from the dataverse.

        Args:
            name (str): Name of the dataset. See the DataMaster.list_datasets() method
                for a list of available datasets.
            dir (str, optional): Directory where the dataset is stored.
                Defaults to a temporary directory.
        """
        self.name = name
        self.datamaster = DataMaster()
        self.api = DataverseAPI(dir)
        self.dir = self.api.dir  # will be tmp if dir is None

        # check if dataset is available
        if name not in self.datamaster.list_datasets():
            raise ValueError(f"Dataset {name} not available")

        # download .zip detaset if necessary
        tgtdir = os.path.join(self.dir, name)
        if not os.path.exists(os.path.join(self.dir, name)):
            # download .zip file
            zip_path = self.api.download_data(name + ".zip")

            # unzip foder
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tgtdir)

            # remove .zip file
            os.remove(zip_path)

        # birth certificate/config
        with open(os.path.join(tgtdir, "config.yaml"), "r") as f:
            self.config = yaml.load(f, Loader=yaml.BaseLoader)

        # extract synthetic data and metadata properties
        self.synthetic_data = pd.read_csv(
            os.path.join(tgtdir, "synthetic_data.csv"), index_col=0
        )
        with open(os.path.join(tgtdir, "metadata.yaml"), "r") as f:
            self.metadata = yaml.load(f, Loader=yaml.BaseLoader)

        # read graph
        self.graph = nx.read_graphml(os.path.join(tgtdir, "graph.graphml"))

        # information about spatial complexity
        # TODO: there is an inconsistency in the names confounding_score and spatial_scores
        # plural, singular
        self.confounding_score_dict = {
            x: float(v) for x, v in self.metadata["confounding_score"].items()
        }
        self.smoothness_score_dict = {
            x: float(v) for x, v in self.metadata["spatial_scores"].items()
        }

    def __masking_candidates(
        self,
        min_confounding: float = 0.0,
        max_confounding: float = 1.0,
        min_smoothness: float = 0.0,
        max_smoothness: float = 1.0,
    ) -> str:
        """Auxiliary method for finding a covariate that satisfies the
        requirements for masking."""
        candidates = []
        for c in self.metadata["covariates"]:
            smoothness = self.smoothness_score_dict[c]
            confounding = self.confounding_score_dict[c]
            if (
                min_confounding <= confounding <= max_confounding
                and min_smoothness <= smoothness <= max_smoothness
            ):
                candidates.append(c)
        if len(candidates) == 0:
            raise ValueError("No covariate found with the specified requirements")
        return candidates

    def __gen__dataset__from__observed_and_missing(
        self,
        missing: str | None,
        observed: list[str],
    ) -> SpaceDataset:
        """Generates a SpaceDataset from a list of observed covariates"""
        if missing is not None:
            observed = [c for c in observed if c != missing]
            missing_smoothness = self.smoothness_score_dict[missing]
            missing_confounding = self.confounding_score_dict[missing]
        else:
            observed = self.metadata["covariates"]
            missing_smoothness = None
            missing_confounding = None

        # counterfactulas, outcome and treatment
        # for counterfactuals, we need to make sure they are in the right order
        outcome = self.synthetic_data["Y_synth"].values
        columns = self.synthetic_data.columns
        cfcols = columns.str.startswith("Y_synth_")
        treatment_index = [int(x[-1]) for x in columns[cfcols].str.split("_")]
        cfcols_order = np.argsort(treatment_index)
        cfcols = columns[cfcols][cfcols_order]
        counterfactuals = self.synthetic_data[cfcols].values
        treatment = self.synthetic_data[self.metadata["treatment"]].values

        # extract graph in usable format
        node2id = {n: i for i, n in enumerate(self.graph.nodes)}
        edge_list = [(node2id[e[0]], node2id[e[1]]) for e in self.graph.edges]
        if self.config["data"]["has_coordinates"]:
            coordinates = [
                [float(xi) for xi in x.values()] for x in self.graph.nodes.values()
            ]
            coordinates = np.array(coordinates)
        else:
            coordinates = None

        # treatment values, make sure they are float
        treatment_values = np.array(
            [float(x) for x in self.metadata["treatment_values"]]
        )

        return SpaceDataset(
            treatment=treatment,
            covariates=self.synthetic_data[observed].values,
            outcome=outcome,
            counterfactuals=counterfactuals,
            edges=edge_list,
            coordinates=coordinates,
            smoothness_of_missing=missing_smoothness,
            confounding_of_missing=missing_confounding,
            treatment_values=treatment_values,
        )

    def make_unmasked(self) -> SpaceDataset:
        """Generates a SpaceDataset with all covariates observed (no missing confounding)

        Returns:
            SpaceDataset: A SpaceDataset with all covariates observed.
        """
        missing = None
        observed = self.metadata["covariates"]
        return self.__gen__dataset__from__observed_and_missing(missing, observed)

    def make(
        self,
        missing: str | None = None,
        min_confounding: float = 0.0,
        max_confounding: float = 1.0,
        min_smoothness: float = 0.0,
        max_smoothness: float = 1.0,
    ) -> SpaceDataset:
        """Generates a SpaceDataset by masking a covariate.

        Args:
            missing (str, optional): Name of the covariate to be masked. If no
                covariate is specified, a covariate is selected at random from the
                ones that satisfy requirements for masking in terms of smoothness
                and confounding. Defaults to None.
            min_confounding (float, optional): Minimum confounding score for the
                covariate to be masked. Defaults to 0.0.
            max_confounding (float, optional): Maximum confounding score for the
                covariate to be masked. Defaults to 1.0.
            min_smoothness (float, optional): Minimum smoothness score for the
                covariate to be masked. Defaults to 0.0.
            max_smoothness (float, optional): Maximum smoothness score for the
                covariate to be masked. Defaults to 1.0.

        Returns:
            SpaceDataset: A SpaceDataset
        """
        if missing is None:
            candidates = self.__masking_candidates(
                min_confounding, max_confounding, min_smoothness, max_smoothness
            )
            missing = np.random.choice(candidates)

        observed = [c for c in self.metadata["covariates"] if c != missing]
        return self.__gen__dataset__from__observed_and_missing(missing, observed)

    def make_all(
        self,
        min_confounding: float = 0.0,
        max_confounding: float = 1.0,
        min_smoothness: float = 0.0,
        max_smoothness: float = 1.0,
    ):
        """Generates all possible SpaceDatasets by masking all posssible covariates.

        Args:
            min_confounding (float, optional): Minimum confounding score for the
                covariate to be masked. Defaults to 0.0.
            max_confounding (float, optional): Maximum confounding score for the
                covariate to be masked. Defaults to 1.0.
            min_smoothness (float, optional): Minimum smoothness score for the
                covariate to be masked. Defaults to 0.0.
            max_smoothness (float, optional): Maximum smoothness score for the
                covariate to be masked. Defaults to 1.0.

        Returns:
            Generator[SpaceDataset]: Generator of SpaceDatasets
        """
        for c in self.metadata["covariates"]:
            smoothness = self.smoothness_score_dict[c]
            confounding = self.confounding_score_dict[c]
            if (
                min_confounding <= confounding <= max_confounding
                and min_smoothness <= smoothness <= max_smoothness
            ):
                yield self.make(missing=c)


if __name__ == "__main__":
    # small test
    # TODO: convert in unit test
    dm = DataMaster()
    envname = dm.list_datasets()[0]
    dir = "downloads"
    generator = SpaceEnv(envname, dir)
    datasets = [generator.make() for _ in range(10)]
    LOGGER.debug("ok")
