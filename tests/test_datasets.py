import unittest

from spacebench import SpaceEnv, SpaceDataset, DataMaster


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = DataMaster().list_datasets()[0]

    def test_create_env(self):
        env = SpaceEnv(self.dataset_name)
        assert isinstance(env, SpaceEnv)

    def test_all_envs(self):
        # TODO: Find a different way to test that all environments
        # in datamaster can be accessed.
        # for dataset in DataMaster().list_datasets():
        #     env = SpaceEnv(dataset)
        #     assert isinstance(env, SpaceEnv)
        pass
    
    def test_create_dataset(self):
        env = SpaceEnv(self.dataset_name)
        dataset = env.make()
        assert isinstance(dataset, SpaceDataset)


if __name__ == "__main__":
    unittest.main()