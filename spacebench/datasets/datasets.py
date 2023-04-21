from dataclasses import dataclass
import sys
import json
import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from os.path import join as join_path


import spacebench.datasets.error_sampler as err


def read_geodata(geodata_file: str) -> gpd.GeoDataFrame:
    """Reads geodata from file"""
    ext = geodata_file.split(".")[-1]
    if ext not in ("shp", "geojson"):
        raise ValueError("spatial_file must be a shapefile or geojson")
    else:
        geodata = gpd.GeoDataFrame.from_file(geodata_file)
    if "index" in geodata.columns:
        geodata = geodata.set_index("index")
    return geodata


def read_graph(graph_file: str) -> nx.Graph:
    """Reads graph from file"""
    ext = graph_file.split(".")[-1]
    if ext != "graphml":
        raise ValueError("graph_file must be a graphml file")
    return nx.read_graphml(graph_file)


def _get_error_sampler_type(error_type: str) -> str:
    """Returns the error sampler type"""
    if error_type == "gp":
        return "GPSampler"
    else:
        raise ValueError("error_type must be gp")


@dataclass
class CausalDataset:
    treatment: pd.DataFrame | np.ndarray
    covariates: pd.DataFrame | np.ndarray
    outcome: pd.DataFrame | np.ndarray
    counterfactuals: pd.DataFrame | np.ndarray

    def save_dataset(self, path: str):
        df = pd.concat(
            [self.treatment,
             self.covariates,
             self.outcome,
             self.counterfactuals]
            , axis=1
        )
        df.to_csv(path, index=False)

@dataclass
class SpatialMetadata:
    """The purpose of this class is simply to validate a common
    interface to store a dataset's metadata.
    A dataset is instantiated with metadata file."""

    data_file: str
    metadata_file: str
    geodata_file: str
    graph_file: str
    source_data: str
    predictor_model: str
    continuous_treatment: bool
    treatment_vals: list[int | float]
    error_type: str
    root: str = "."
    error_params: dict | None = (None,)
    variable_importance: dict | list | None = None
    variable_smoothness: dict | list | None = None
    variable_score: dict | list | None = None

    @classmethod
    def from_json(cls, json_path: str) -> "SpatialMetadata":
        # extract root from json_path
        with open(json_path, "r") as io:
            meta = json.load(io)

        if meta["data_file"] is None:
            raise ValueError("data_file must be specified")
        
        vi = meta["variable_importance"]
        vs = meta["variable_smoothness"]
        if vi is not None and vs is not None:
            # take min of vs and vi, case by list or dict
            if isinstance(vi, list):
                meta["variable_score"] = [min(vi[i], vs[i]) for i in range(len(vi))]
            elif isinstance(vi, dict):
                meta["variable_score"] = {k: min(vi[k], vs[k]) for k in vi.keys()}

        return cls(
            data_file=meta["data_file"],
            metadata_file=meta["metadata_file"],
            geodata_file=meta["geodata_file"],
            graph_file=meta["graph_file"],
            source_data=meta["source_data"],
            predictor_model=meta["predictor_model"],
            continuous_treatment=meta["continuous_treatment"],
            treatment_vals=meta["treatment_vals"],
            error_type=meta["error_type"],
            error_params=meta["error_params"],
            root="/".join(json_path.split("/")[:-1]),
            variable_importance=meta["variable_importance"],
            variable_smoothness=meta["variable_smoothness"],
            variable_score=meta["variable_score"],
        )

    @property
    def geodata_path(self) -> str:
        if self.geodata_file is None:
            return None
        else:
            return join_path(self.root, self.geodata_file)

    @property
    def graph_path(self) -> str:
        if self.graph_file is None:
            return None
        else:
            return join_path(self.root, self.graph_file)

    @property
    def data_path(self) -> str:
        return join_path(self.root, self.data_file)


class DatasetGenerator:
    """This class generates datasets"""

    def __init__(self, random_state, metadata: SpatialMetadata):
        self.metadata = metadata

        # read data and split into treatment, covariates, outcome, and counterfactual
        data = pd.read_csv(
            self.metadata.data_path, dtype={0: str}
        )  # index always string
        data.rename(columns={"Unnamed: 0": "index"}, inplace=True)
        data.set_index("index", inplace=True)
        self.treatment = data.treatment.copy()
        self.covariates = data[[c for c in data.columns if c.startswith("X")]].copy()
        self.pred = data.pred.copy()
        self.predcf = data[[c for c in data.columns if c.startswith("predcf")]].copy()

        # read geodata if given
        if self.metadata.geodata_path:
            self.geodata = read_geodata(self.metadata.geodata_path)

        # read graph if given
        if self.metadata.graph_path:
            self.graph = read_graph(self.metadata.graph_path)
            error_attrs = pd.DataFrame(
                self.graph.nodes.values(), index=self.graph.nodes
            )
            self.error_attrs = error_attrs.loc[self.treatment.index]  # align

        # make error generator
        sampler_fun = getattr(err, _get_error_sampler_type(self.metadata.error_type))
        self.error_sampler = sampler_fun(self.metadata.error_params)

    @classmethod
    def from_json(cls, json_path: str) -> "DatasetGenerator":
        metadata = SpatialMetadata.from_json(json_path)
        return cls(metadata)

    def make_dataset(self) -> CausalDataset:
        res = self.error_sampler.sample(self.error_attrs)
        outcome = self.pred + res
        counterfactuals = self.predcf.copy()
        for c in counterfactuals.columns:
            counterfactuals[c] += res

        dataset = CausalDataset(
            treatment=self.treatment,
            covariates=self.covariates,
            outcome=outcome,
            counterfactuals=counterfactuals,
        )
        self.mask(dataset)
        return dataset
    
    def mask(self, dataset: CausalDataset) -> CausalDataset:
        score = self.metadata.variable_score
        # if score is dict make list
        
        if isinstance(score, dict):
            score = [score[c] for c in dataset.covariates.columns]
        
        # take indices of top 10 from score
        top10 = np.argsort(score)[-10:]

        # take a random index from top 10
        masked = np.random.choice(top10)

        # remove the column from data.covariates
        dataset.covariates = dataset.covariates.drop(dataset.covariates.columns[masked], axis=1)

