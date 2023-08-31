import unittest

import numpy as np
import scipy.sparse

from spacebench import SpaceEnv, SpaceDataset, DataMaster


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = DataMaster().list_envs()[0]

    def test_create_env(self):
        env = SpaceEnv(self.dataset_name)
        assert isinstance(env, SpaceEnv)

    def test_all_envs(self):
        # TODO: Find a different way to test that all environments
        # in datamaster can be accessed.
        # for dataset in DataMaster().list_envs():
        #     env = SpaceEnv(dataset)
        #     assert isinstance(env, SpaceEnv)
        pass

    def test_create_dataset(self):
        env = SpaceEnv(self.dataset_name)
        dataset = env.make()
        assert isinstance(dataset, SpaceDataset)


class TestSpaceDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SpaceDataset(
            treatment=np.array([0, 1]),
            covariates=np.array([[1, 2, 3], [4, 5, 6]]),
            outcome=np.array([1, 0]),
            edges=[(0, 1)],
            treatment_values=np.array([0, 1]),
            counterfactuals=np.array([[0, 1], [1, 0]]),
        )

    def test_has_binary_treatment(self):
        self.assertTrue(self.dataset.has_binary_treatment())

    def test_erf(self):
        np.testing.assert_array_equal(self.dataset.erf(), np.array([0.5, 0.5]))

    def test_negative_case_binary_treatment(self):
        self.dataset.treatment_values = np.array([0, 1, 2])
        self.assertFalse(self.dataset.has_binary_treatment())

    def test_adjacency_matrix_dense(self):
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_array_equal(
            self.dataset.adjacency_matrix(sparse=False), expected
        )

    def test_adjacency_matrix_sparse(self):
        expected = scipy.sparse.csr_matrix([[0, 1], [1, 0]])
        np.testing.assert_array_equal(
            self.dataset.adjacency_matrix(sparse=True).toarray(), expected.toarray()
        )

    def test_indexing(self):
        np.testing.assert_array_equal(
            self.dataset[[0, 1]].treatment, self.dataset.treatment
        )
        np.testing.assert_array_equal(
            self.dataset[[0]].treatment, self.dataset.treatment[:1]
        )

    def test_size(self):
        self.assertEqual(self.dataset.size(), 2)


class TestRemoveIsland(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SpaceDataset(
            treatment=np.array([0, 1, 2]),
            covariates=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            outcome=np.array([1, 0, -1]),
            edges=[(0, 1)],
            treatment_values=np.array([0, 1, 2]),
            counterfactuals=np.array([[0, 1], [1, 0], [1, 1]]),
        )

    def test_remove_island(self):
        dataset = self.dataset.remove_islands()
        np.testing.assert_array_equal(dataset.treatment, np.array([0, 1]))
        np.testing.assert_array_equal(
            dataset.covariates, np.array([[1, 2, 3], [4, 5, 6]])
        )
        np.testing.assert_array_equal(dataset.outcome, np.array([1, 0]))
        np.testing.assert_array_equal(dataset.edges, [(0, 1)])
        np.testing.assert_array_equal(dataset.treatment_values, np.array([0, 1, 2]))
        np.testing.assert_array_equal(
            dataset.counterfactuals, np.array([[0, 1], [1, 0]])
        )


if __name__ == "__main__":
    unittest.main()
