import logging
import os
import shutil
import time

import hydra
import jsonlines
import ray
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from ray import tune

import spacebench
from spacebench.algorithms.datautils import spatial_train_test_split

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.global_seed)

    # make logfile
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = cfg.logfile
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    LOGGER.info(f"Logging to {logfile}")

    # check if logfile exists and overwrite, delete it and continue
    # otherwise return
    if os.path.exists(logfile):
        if cfg.overwrite:
            LOGGER.info(f"Cleaning logfile {logfile}")
            os.remove(logfile)
        else:
            LOGGER.info(f"Logfile {logfile} already exists, skipping")
            return

    # check if run_config.storage_path exists, and if so, delete it
    raydir = f"{cfg.algo.tune.run_config.local_dir}/{cfg.algo.name}"
    if os.path.exists(raydir):
        LOGGER.info(f"Cleaning ray path {raydir}")
        shutil.rmtree(raydir)

    env_name = cfg.spaceenv
    env = spacebench.SpaceEnv(env_name, dir="downloads")
    train_ix, test_ix, _ = spatial_train_test_split(
        env.graph, **cfg.spatial_train_test_split
    )

    for i, full_dataset in enumerate(env.make_all()):
        LOGGER.info(f"Running dataset {i} from {env_name}")

        # train/test split
        LOGGER.info("...splitting dataset into train/test")
        if cfg.algo.needs_train_test_split:
            train_dataset = full_dataset[train_ix]
            test_dataset = full_dataset[test_ix]
        else:
            train_dataset = full_dataset
            test_dataset = full_dataset

        # setup hyperparameter tuning objective
        param_space = dict(hydra.utils.instantiate(cfg.algo.tune.param_space))
        if len(param_space) > 0:

            def objective(config):
                method = hydra.utils.instantiate(cfg.algo.method, **config)
                method.fit(train_dataset)
                tune_metric = method.tune_metric(test_dataset)
                return tune_metric

            objective = tune.with_resources(objective, dict(cfg.algo.tune.resources))

            # create tuner
            LOGGER.info("...setting up hyperparameter tuning")
            tune_config = hydra.utils.instantiate(cfg.algo.tune.tune_config)
            run_config = hydra.utils.instantiate(
                cfg.algo.tune.run_config, name=f"{i:02d}"
            )
            ray.shutdown()
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                # num_cpus=cfg.concurrency,
            )
            tuner = tune.Tuner(
                objective,
                tune_config=tune_config,
                param_space=param_space,
                run_config=run_config,
            )

            # run hyperparameter tuning
            LOGGER.info("...running hyperparameter tuning")
            results = tuner.fit()

            # save all grid results in all_results
            best_params = results.get_best_result().config
            LOGGER.info(f"Best params: {best_params}")
        else:
            LOGGER.info("...skipping hyperparameter tuning, no param space provided")
            best_params = {}

        LOGGER.info("...training full model")
        method = hydra.utils.instantiate(cfg.algo.method, **best_params)
        method.fit(full_dataset)
        effects = method.eval(full_dataset)

        # load evaluator
        evaluator = spacebench.DatasetEvaluator(full_dataset)
        eval_results = evaluator.eval(**effects)
        eval_results = {k: eval_results.get(k, None) for k in ("ate", "erf", "ite")}
        eval_results["env"] = env_name
        eval_results["dataset_id"] = i
        eval_results["algo"] = cfg.algo.name
        eval_results["smoothness"] = full_dataset.smoothness_score
        eval_results["confounding"] = full_dataset.confounding_score
        eval_results["timestamp"] = timestamp
        eval_results["binary"] = full_dataset.has_binary_treatment()

        LOGGER.info(f"eval results: {eval_results}")

        with jsonlines.open(logfile, mode="a") as writer:
            writer.write(eval_results)


if __name__ == "__main__":
    main()
