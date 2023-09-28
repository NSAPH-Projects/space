"""
utils.py
====================================
Utility function for manipulating spacebench object
"""

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from spacebench import SpaceDataset
from spacebench.log import LOGGER


def spatial_train_test_split(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int = 0,
    seed: int | None = None,
) -> tuple[list, list, list]:
    """Utility function for splitting a graph into training, tuning, and buffer sets.
    Provides a simple way for spatially aware train/test splits.

    Arguments
    ---------
    graph : nx.Graph
        Graph object to split

    init_frac : float
        Fraction of nodes to use as initial tuning centroids

    levels : int
        Number of levels of neighbors to include in tuning set by breadth-first search
        starting from the initial tuning centroids

    buffer : int
        Number of levels of neighbors to include in buffer set by breadth-first search
        starting from the tuning set. The buffer set is not used in training or tuning. Default is 0.

    seed : int, optional
        Seed for random number generator

    Returns
    -------
    training_nodes : list
        List of training node indices
    tuning_nodes : list
        List of tuning node indices
    buffer_nodes : list
        List of buffer node indices
    """
    LOGGER.debug(
        f"Selecting tunning split removing {levels} level and {buffer} buffer from val. pts."
    )

    # make dict of neighbors from graph
    node_list = np.array(graph.nodes())
    n = len(node_list)
    nbrs = {node: set(graph.neighbors(node)) for node in node_list}

    # first find the centroid of the tuning subgraph
    num_tuning_centroids = int(init_frac * n)
    rng = np.random.default_rng(seed)
    tuning_nodes = rng.choice(n, size=num_tuning_centroids, replace=False)
    tuning_nodes = set(node_list[tuning_nodes])

    # not remove all neighbors of the tuning centroids from the training data
    for _ in range(levels):
        tmp = tuning_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                tuning_nodes.add(nbr)
    tuning_nodes = list(tuning_nodes)

    # buffer
    buffer_nodes = set(tuning_nodes.copy())
    for _ in range(buffer):
        tmp = buffer_nodes.copy()
        for node in tmp:
            for nbr in nbrs[node]:
                buffer_nodes.add(nbr)
    buffer_nodes = list(set(buffer_nodes))
    buffer_nodes = list(set(tuning_nodes) - set(buffer_nodes))

    training_nodes = list(set(node_list) - set(tuning_nodes) - set(buffer_nodes))

    # make training, tune and buffer in terms of integer indices

    return training_nodes, tuning_nodes, buffer_nodes


def graph_data_loader(
    dataset: SpaceDataset,
    feat_scaler: StandardScaler | None = None,
    output_scaler: StandardScaler | None = None,
    treatment_value: float | None = None,
):
    """Utility function for loading a SpaceDataset into a PyTorch Geometric DataLoader.

    Arguments
    ----------
    dataset : SpaceDataset
        The dataset to load into a DataLoader.
    feat_scaler : sklearn.preprocessing.StandardScaler, optional
        The feature scaler to use. If None, a new scaler is created.
    output_scaler : sklearn.preprocessing.StandardScaler, optional
        The output scaler to use. If None, a new scaler is created.
    treatment_value : float, optional
        The treatment value to use. If not none, then the treatment value is
        fixed to the given value for all samples.

    Returns
    -------
    loader : pytorch_geometric.loader.DataLoader
        The DataLoader containing the dataset.
    feat_scaler : sklearn.preprocessing.StandardScaler
        The fitted feature scaler.
    output_scaler : sklearn.preprocessing.StandardScale
        The fitted output scaler.
    """

    if treatment_value is not None:
        treatment = np.full((dataset.size(), 1), treatment_value)
    else:
        treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    outcome = dataset.outcome.reshape(-1, 1)
    features = np.hstack([treatment, covariates])

    if feat_scaler is None:
        feat_scaler = StandardScaler()
        feat_scaler.fit(features)

    if output_scaler is None:
        output_scaler = StandardScaler()
        output_scaler.fit(outcome)

    edge_index = torch.LongTensor(dataset.edges).T

    x = torch.FloatTensor(feat_scaler.transform(features))
    y = torch.FloatTensor(output_scaler.transform(outcome))

    loader = DataLoader(
        [Data(x=x, y=y, edge_index=edge_index)],
        batch_size=features.shape[0],
        shuffle=False,
        num_workers=0,
    )

    return loader, feat_scaler, output_scaler
