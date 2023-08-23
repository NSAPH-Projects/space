import os
import time
import argparse
import concurrent.futures

import jsonlines
import numpy as np
from xgboost import XGBRegressor

from spacebench import SpaceEnv, SpaceDataset, DataMaster, DatasetEvaluator


def run_xgboost(
    dataset: SpaceDataset,
    use_coords: bool,
    **kwargs,
):
    model = XGBRegressor()
    model_coords = XGBRegressor()

    treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    outcome = dataset.outcome

    # make train matrix
    trainmat = np.hstack([covariates, treatment])
    if use_coords:
        coords = np.array(dataset.coordinates)
        trainmat = np.hstack([trainmat, coords])

    # fit model
    model.fit(trainmat, outcome)

    # predict counterfactuals
    tvals = dataset.treatment_values
    counterfactuals = []
    for tval in tvals:
        trainmat = np.hstack([covariates, np.full_like(treatment, tval)])
        if use_coords:
            trainmat = np.hstack([trainmat, coords])
        counterfactuals.append(model.predict(trainmat))
    counterfactuals = np.stack(counterfactuals, axis=1)

    evaluator = DatasetEvaluator(dataset)
    erf = counterfactuals.mean(0)
    err_eval = evaluator.eval(erf=erf, counterfactuals=np.squeeze(counterfactuals))

    res = {}
    for key, value in err_eval.items():
        if isinstance(value, np.ndarray):
            res[key] = value.tolist()
    res["smoothness"] = dataset.smoothness_of_missing
    res["confounding"] = dataset.confounding_of_missing
    res.update(**kwargs)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    method_choices = ["xgboost", "xgboost_coords"]
    parser.add_argument("--method", choices=method_choices, default="xgboost")
    args = parser.parse_args()

    start = time.perf_counter()
    use_coords = True if "coords" in args.method else False

    datamaster = DataMaster()
    envs = datamaster.list_envs()

    filename = f"results/results_{args.method}.jsonl"
    if not os.path.exists("results"):
        os.mkdir("results")

    # Clean the file
    if args.overwrite:
        if os.path.exists(filename):
            os.remove(filename)

    for envname in envs:
        print(f"Running {envname}")
        env = SpaceEnv(envname, dir="downloads")
        dataset_list = list(env.make_all())
        binary = True if "disc" in envname else False

        # remove from the list the datasets that have been already computed
        if os.path.exists(filename):
            with jsonlines.open(filename) as reader:
                results = list(reader)
        else:
            results = []

        to_remove = []
        for dataset in dataset_list:
            spatial_score = dataset.smoothness_of_missing
            confounding_score = dataset.confounding_of_missing
            for result in results:
                if (
                    result["envname"] == envname
                    and result["smoothness"] == spatial_score
                    and result["confounding"] == confounding_score
                ):
                    to_remove.append(id(dataset))
        dataset_list = [
            dataset for dataset in dataset_list if id(dataset) not in to_remove
        ]

        with concurrent.futures.ProcessPoolExecutor(args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_xgboost, dataset, use_coords=use_coords, dataset_num=i
                )
                for i, dataset in enumerate(dataset_list)
            }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode="a") as writer:
                    result["envname"] = envname
                    writer.write(result)

    finish = time.perf_counter()

    print(f"Finished in {round(finish-start, 2)} second(s)")
