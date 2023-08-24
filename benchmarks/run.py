import hydra
from omegaconf import DictConfig

import spacebench
from spacebench.log import LOGGER


def fit_dataset(dataset: spacebench.SpaceDataset, cfg: DictConfig):
    """Fits the algorithm to the given dataset and returns an
    estimation of the causal effects."""
    method = hydra.utils.instantiate(cfg.algo.algo)
    effects = method.fit(dataset)
    LOGGER.info(f"Effects: {effects}")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dm = spacebench.DataMaster()
    envs = dm.list_envs(binary=cfg.algo.binary, continuous=cfg.algo.continuous)

    for env_name in envs:
        env = spacebench.SpaceEnv(env_name)
        for i, dataset in enumerate(env.make_all()):
            LOGGER.info(f"Running {i}-th dataset")
            fit_dataset(dataset, cfg)


if __name__ == "__main__":
    main()
