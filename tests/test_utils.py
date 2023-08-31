import unittest

import networkx as nx
import numpy as np

from spacebench.algorithms import datautils


class TestTrainTestSplitGenerator(unittest.TestCase):
    def setUp(self) -> None:
        edges = [(i, i + 1) for i in range(100)]
        self.graph = nx.Graph(edges)

    def test_split_size(self):
        # test total number of nodes is correct
        ix_train, ix_test, ix_buffer = datautils.spatial_train_test_split(
            self.graph, init_frac=0.05, levels=1, buffer=1, seed=123
        )
        n = len(self.graph.nodes())
        assert len(ix_train) + len(ix_test) + len(ix_buffer) == n

    def test_split_seed(self):
        # test seed
        ix_train_1, ix_test_1, ix_buffer_1 = datautils.spatial_train_test_split(
            self.graph, init_frac=0.05, levels=1, buffer=1, seed=123
        )
        ix_train_2, ix_test_2, ix_buffer_2 = datautils.spatial_train_test_split(
            self.graph, init_frac=0.05, levels=1, buffer=1, seed=123
        )
        # test all equal
        assert np.all(ix_train_1 == ix_train_2)
        assert np.all(ix_test_1 == ix_test_2)
        assert np.all(ix_buffer_1 == ix_buffer_2)


if __name__ == "__main__":
    unittest.main()
