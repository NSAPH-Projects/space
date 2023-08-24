import unittest

import networkx as nx

from spacebench import utils


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        edges = [(i, i + 1) for i in range(100)]
        self.graph = nx.Graph(edges)

    def test_split_train_test(self):
        ix_train, ix_test, ix_buffer = utils.spatial_train_test_split(
            self.graph, init_frac=0.05, levels=1, buffer=1, seed=0
        )
        n = len(self.graph.nodes())
        assert len(ix_train) + len(ix_test) + len(ix_buffer) == n


if __name__ == "__main__":
    unittest.main()
