"""Module for defining the SpaceEnvironment class"""
import itertools
import json
import os
import zipfile
from dataclasses import dataclass
from glob import glob
import tarfile
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import yaml

from spacebench.api.dataverse import DataverseAPI
from spacebench.datamaster import DataMaster
from spacebench.log import LOGGER


@dataclass
class SpaceDataset:
    """
    Class for storing a spatial causal inference benchmark dataset.
    """

    treatment: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray
    edges: list[tuple[int, int]]
    treatment_values: np.ndarray
    counterfactuals: np.ndarray
    missing_covariates: np.ndarray | None = None
    smoothness_score: list[float] | None = None
    confounding_score: dict[Literal["ate", "erf", "ite"], list[float]] | None = None
    coordinates: np.ndarray | None = None
    parent_env: str | None = None

    def has_binary_treatment(self) -> bool:
        """
        Returns true if treatment is binary.
        """
        return len(self.treatment_values) == 2

    def erf(self) -> np.ndarray:
        """
        Returns the exposure-response function, also known
        in the literature as the average dose-response function.

        Returns
        -------
        np.ndarray: The exposure-response function
        """

        return self.counterfactuals.mean(0)

    def adjacency_matrix(
        self, sparse: bool = False
    ) -> np.ndarray | scipy.sparse.csr_matrix:
        """
        Returns the adjacency matrix of the graph.

        Parameters
        ----------
        sparse: bool, optional (default is False)
            If True, returns a sparse matrix of type csr_matrix. If False,
            returns a dense matrix.

        Returns
        -------
        np.ndarray | scipy.sparse.csr_matrix
            Adjacency matrix where entry (i, j) is 1 if there is an edge
            between node i and node j.
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

    @property
    def unmasked_covariates(self) -> np.ndarray:
        """
        Returns the covariates without the missing confounder.
        """
        if self.missing_covariates is None:
            return self.covariates
        else:
            return np.hstack([self.covariates, self.missing_covariates])

    def __repr__(self) -> str:
        warning_msg = (
            "WARNING ⚠️ : this dataset contains a (realistic) synthetic outcome!\n"
            + "By using it, you agree to understand its limitations."
            + "The variable names have been masked to emphasize that no"
            + "inferences can be made about the source data.\n"
        )
        cs = {x: float(np.round(v, 4)) for x, v in self.confounding_score.items()}
        b = "binary" if self.has_binary_treatment() else "continuous"
        s = f"SpaceDataset with a missing spatial confounder:\n"
        s += f"  treatment: {self.treatment.shape} ({b})\n"
        s += f"  outcome: {self.outcome.shape}\n"
        s += f"  counterfactuals: {self.counterfactuals.shape}\n"
        s += f"  covariates: {self.covariates.shape}\n"
        s += f"  missing covariates: {self.missing_covariates.shape}\n"
        s += f"  confounding score of missing: {cs}\n"
        s += f"  spatial smoothness score of missing: {self.smoothness_score:.2f}\n"
        s += f"  graph edge list: {np.array(self.edges).shape}\n"
        s += f"  graph node coordinates: {self.coordinates.shape}\n"
        s += f"  parent SpaceEnv: {self.parent_env}\n"
        s += warning_msg
        return s

    # add indexing method that retuns a new SpaceDataset, and subset of the graph
    # it should only support indexing by a list of indices, boolean mask or nd array
    def __getitem__(self, idx: list[int | bool]) -> "SpaceDataset":
        # if list of boolean, extract indices
        if isinstance(idx, list) and isinstance(idx[0], bool):
            idx = np.arange(len(idx))[idx]

        ind_ = set(idx)
        subedges = [e for e in self.edges if e[0] in ind_ and e[1] in ind_]
        new_ind = sorted(set(itertools.chain.from_iterable(subedges)))
        new_ind = {x: i for i, x in enumerate(new_ind)}
        new_edges = [(new_ind[e[0]], new_ind[e[1]]) for e in subedges]

        return SpaceDataset(
            treatment=self.treatment[idx],
            covariates=self.covariates[idx],
            outcome=self.outcome[idx],
            edges=new_edges,
            treatment_values=self.treatment_values,
            missing_covariates=self.missing_covariates,
            smoothness_score=self.smoothness_score,
            confounding_score=self.confounding_score,
            counterfactuals=self.counterfactuals[idx],
            coordinates=self.coordinates[idx] if self.coordinates is not None else None,
            parent_env=self.parent_env,
        )

    def size(self) -> int:
        """Returns the number of nodes in the dataset"""
        return len(self.treatment)

    def remove_islands(self) -> "SpaceDataset":
        """Returns a space dataset without islands

        Returns
        _______
        SpaceDataset
            A new SpaceDataset without islands. The components of the
            spacedataset and edge indices are updated accordingly.
            When no islands are found, it returns self. When islands are found it
            returns a subset without islands.
        """
        num_neighbors = np.zeros(len(self.treatment), dtype=int)
        for e in self.edges:
            num_neighbors[e[0]] += 1
            num_neighbors[e[1]] += 1
        islands = num_neighbors == 0

        if sum(islands) == 0:
            LOGGER.debug("No islands found. Returning self.")
            return self
        else:
            LOGGER.debug(f"Found {sum(islands)} islands. Removing them.")
            return self[~islands]
        
    def unmask(self, inplace: bool = False) -> "SpaceDataset":
        """
        Returns a SpaceDataset with the masked covariate unmasked.

        Parameters
        ----------
        inplace: bool, optional (default is False)
            If True, the covariates are unmasked inplace. If False, a new
            SpaceDataset is returned.

        Returns
        -------
        SpaceDataset
            A new SpaceDataset with the masked covariate unmasked.
        """
        if self.missing_covariates is None:
            raise ValueError("Dataset is already unmasked")

        if inplace:
            self.covariates = self.unmasked_covariates
            self.missing_covariates = None
            return self
        else:
            return SpaceDataset(
                treatment=self.treatment,
                covariates=self.unmasked_covariates,
                outcome=self.outcome,
                edges=self.edges,
                treatment_values=self.treatment_values,
                missing_covariates=None,
                smoothness_score=self.smoothness_score,
                confounding_score=self.confounding_score,
                counterfactuals=self.counterfactuals,
                coordinates=self.coordinates,
                parent_env=self.parent_env,
            )


class SpaceEnv:
    """
    Class for a SpaCE environment.

    It holdss the data and metadata that is used to generate the datasets by
    masking a covariate, which becomes a missing confounder.

    Attributes
    ----------

    api: DataverseAPI
        Dataverse API object.
    config: dict
        Dictionary with the configuration of the dataset.
    counfound_score_dict: dict
        Dictionary with the confounding scores of the covariates.
    datamaster: DataMaster
        DataMaster object.
    dir: str
        Directory where the dataset is stored.
    graph: networkx.Graph
        Graph of the dataset.
    metadata: dict
        Dictionary with the metadata of the dataset.
    name: str
        Name of the dataset.
    smoothness_score_dict: dict
        Dictionary with the smoothness scores of the covariates.
    synthetic_data: pd.DataFrame
        Synthetic data of the dataset.

    """

    def __init__(
        self,
        name: str,
        dir: str | None = None,
    ):
        """
        Initializes the SpaceEnv class using a dataset name.

        When the dataset is not found in the directory, it is downloaded from
        the dataverse.

        Parameters
        ----------

        name: str
            Name of the dataset. See the DataMaster.list_envs() method
            for a list of available datasets.
        dir: str, optional
            Directory where the dataset is stored. Defaults to a temporary
            directory.
        """
        self.name = name
        self.datamaster = DataMaster()
        self.api = DataverseAPI(dir)
        self.dir = self.api.dir  # will be tmp if dir is None

        # check if dataset is available
        if name not in self.datamaster.list_envs():
            raise ValueError(f"Dataset {name} not available")

        # download .zip dataset if necessary
        tgtdir = os.path.join(self.dir, name)
        if not os.path.exists(os.path.join(self.dir, name)):
            # download .zip file
            zip_path = self.api.download_data(name + ".zip")

            # unzip foder
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tgtdir)

            # remove .zip file
            os.remove(zip_path)

        # -- 1. config (birth certificate) and metadata --
        with open(os.path.join(tgtdir, "config.yaml"), "r") as f:
            self.config = yaml.load(f, Loader=yaml.BaseLoader)

        with open(os.path.join(tgtdir, "metadata.yaml"), "r") as f:
            self.metadata = yaml.load(f, Loader=yaml.BaseLoader)

        # -- full data --
        ext = ".".join(glob(os.path.join(tgtdir, "synthetic_data.*"))[0].split(".")[1:])
        if ext == "csv":
            data = pd.read_csv(os.path.join(tgtdir, "synthetic_data.csv"), index_col=0)
        elif ext in ("tab", "tsv"):
            data = pd.read_csv(os.path.join(tgtdir, "synthetic_data.tab"), sep="\t")
        elif ext == "parquet":
            data = pd.read_parquet(os.path.join(tgtdir, "synthetic_data.parquet"))
        else:
            raise ValueError(f"Unknown file extension: {ext}")

        # -- 2. outcome (Y) --
        self.outcome = data["Y_synth"].values

        # -- 3. counterfactuals (Ycf) --
        cfcols = sorted(
            data.columns[data.columns.str.startswith("Y_synth_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        self.counterfactuals = data[cfcols].values

        # -- 4. treatment --
        self.treatment = data[self.metadata["treatment"]].values
        self.treatment_values = np.array(
            sorted(float(x) for x in self.metadata["treatment_values"])
        )

        # for 0/1 when treatment is binary
        if len(self.treatment_values) == 2:
            self.treatment = self.treatment == self.treatment_values[1]
            self.treatment_values = np.array([0, 1])

        # -- 5. graph, edges --
        ext = ".".join(glob(os.path.join(tgtdir, "graph.*"))[0].split(".")[1:])
        if ext in ("graphml", "graphml.gz"):
            graph = nx.read_graphml(os.path.join(tgtdir, f"graph.{ext}"))
        elif ext == "tar.gz":
            with tarfile.open(os.path.join(tgtdir, "graph.tar.gz"), "r:gz") as tar:
                edges = pd.read_parquet(tar.extractfile("graph/edges.parquet"))
                coords = pd.read_parquet(tar.extractfile("graph/coords.parquet"))

            graph = nx.Graph()
            graph.add_nodes_from(coords.index)
            graph.add_edges_from(edges.values)
       
        node2id = {n: i for i, n in enumerate(graph.nodes)}
        self.edge_list = [(node2id[e[0]], node2id[e[1]]) for e in graph.edges]
        self.graph = nx.from_edgelist(self.edge_list)

        coordinates = []
        for v in graph.nodes.values():
            coordinates.append([float(x) for x in v.values()])
        self.coordinates = np.array(coordinates)

        # -- 6. covariates --
        # keep it as pandas dataframe so that is easier to subset
        self.covariates_df = data[self.metadata["covariates"]]

        # covariate groups
        self.covariate_groups = {}
        for c in self.metadata["covariate_groups"]:
            if isinstance(c, dict):
                self.covariate_groups.update(c)
            else:
                self.covariate_groups[c] = [c]

        # -- 7. confounding and smoothness scores --
        self.confounding_score = dict()
        for metric in ["erf", "ate", "ite", "importance"]:
            if metric != "importance":
                cs = self.metadata[f"confounding_score_{metric}"]
                values = {x: float(v) for x, v in cs.items()}
            else:
                cs = self.metadata[f"confounding_score"]
                values = {x: float(v) for x, v in cs.items()}
            self.confounding_score[metric] = values
        self.smoothness_score = {
            x: float(v) for x, v in self.metadata["spatial_scores"].items()
        }

    def make(
        self,
        missing_group: str | None = None,
    ) -> SpaceDataset:
        """
        Generates a SpaceDataset by ramasking a covariate.

        Parameters
        ----------
        missing_group: str, optional (Default is None)
            Name of the covariate group to be masked. See self.covariate_groups for
            a list. If no covariate  groupis specified, a
            covariate group is selected at random.

        Returns
        -------
        SpaceDataset
            A SpaceDataset.
        """
        if missing_group is None:
            keys = list(self.covariate_groups.keys())
            missing_group = np.random.choice(keys)
            LOGGER.debug(
                f"Missing covariate group (selected at random): {missing_group}"
            )
        else:
            LOGGER.debug(f"Missing covariate group: {missing_group}")

        # observed covariates
        obs_covars_cols = list(
            itertools.chain.from_iterable(
                [v for k, v in self.covariate_groups.items() if k != missing_group]
            )
        )
        obs_covars = self.covariates_df[obs_covars_cols].values

        # missing covariates
        miss_covars_cols = self.covariate_groups[missing_group]
        miss_covars = self.covariates_df[miss_covars_cols].values

        # smoothness scores
        miss_smoothness = min(self.smoothness_score[x] for x in miss_covars_cols)

        # confounding scores
        miss_confounding = {}
        cs = self.confounding_score
        for k in ["erf", "ate", "ite"]:
            miss_confounding[k] = cs[k].get(missing_group, np.nan)
        miss_confounding["importance"] = max(
            cs["importance"][x] for x in miss_covars_cols
        )

        return SpaceDataset(
            treatment=self.treatment,
            covariates=obs_covars,
            missing_covariates=miss_covars,
            outcome=self.outcome,
            counterfactuals=self.counterfactuals,
            edges=self.edge_list,
            coordinates=self.coordinates,
            smoothness_score=miss_smoothness,
            confounding_score=miss_confounding,
            treatment_values=self.treatment_values,
            parent_env=self.name,
        )

    def make_all(self):
        """
        Generates all possible SpaceDatasets by masking all posssible
        covariates.

        Returns
        -------
        Generator[SpaceDataset]: Generator of SpaceDatasets
        """
        for c in self.covariate_groups:
            yield self.make(missing_group=c)

    def has_binary_treatment(self) -> bool:
        """
        Returns true if treatment is binary.
        """
        return len(self.treatment_values) == 2

    def __repr__(self) -> str:
        warning_msg = (
            "WARNING ⚠️ : this env contains data with a (realistic) synthetic outcome!\n"
            + "No inferences about the source data collection can be made.\n"
            + "By using it, you agree to understand its limitations."
        )

        s = f"SpaceEnv with birth certificate config:\n"
        s += f"{json.dumps(self.config, indent=2)}\n"
        s += warning_msg
        return s


if __name__ == "__main__":
    # small test
    # TODO: convert in unit test
    # import spacebench
    # dm = DataMaster()
    # envname = dm.list_envs()[0]
    envname = "healthd_dmgrcs_mortality_disc"
    dir = "downloads"
    env = SpaceEnv(envname, dir)
    print(env)
    data = env.make()
    print(data)
    datasets = [env.make() for _ in range(10)]
    LOGGER.debug("ok")
