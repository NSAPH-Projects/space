"""Utility function for manipulating spacebench objects"""
import networkx as nx
import numpy as np

from spacebench.log import LOGGER


def spatial_train_test_split(
    graph: nx.Graph,
    init_frac: float,
    levels: int,
    buffer: int,
    seed: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Utility function for splitting a graph into training, tuning, and buffer sets.
    Provides a simple way for spatially aware train/test splits.

    Arguments
    __________
    graph : nx.Graph
        Graph object to split

    init_frac : float
        Fraction of nodes to use as initial tuning centroids

    levels : int
        Number of levels of neighbors to include in tuning set by breadth-first search
        starting from the initial tuning centroids

    buffer : int
        Number of levels of neighbors to include in buffer set by breadth-first search
        starting from the tuning set. The buffer set is not used in training or tuning.

    seed : int, optional
        Seed for random number generator

    Returns
    _______
    training_nodes : list[int]
        List of training node indices
    tuning_nodes : list[int]
        List of tuning node indices
    buffer_nodes : list[int]
        List of buffer node indices
    """
    LOGGER.debug(f"Selecting tunning split removing {levels} nbrs from val. pts.")

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

    return training_nodes, tuning_nodes, buffer_nodes
