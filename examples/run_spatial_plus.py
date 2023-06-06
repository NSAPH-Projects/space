import concurrent.futures
import os
import jsonlines
import time
from spacebench.algorithms import spatial, spatialplus
import numpy as np
import argparse
import pandas as pd
from spacebench import (
    SpaceEnv,
    SpaceDataset,
    DataMaster,
    DatasetEvaluator,
)
from typing import Literal


def run_spatial_plus(
    dataset: SpaceDataset,
    binary_treatment: bool,
    method: Literal["spatial", "spatial_plus"],
):
    treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    # Scale covariates
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    outcome = dataset.outcome
    coords = np.array(dataset.coordinates)

    # Scale coordinates
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    covnames = ["cov" + str(i + 1) for i in range(covariates.shape[1])]
    df = pd.DataFrame(
        np.column_stack((coords, covariates, treatment, outcome)),
        columns=["coord1", "coord2"] + covnames + ["X", "Y"],
    )
    tvals = dataset.treatment_values

    # Create spatial cfs
    fun = spatialplus if method == "spatial_plus" else spatial
    beta_spatial = fun.fit(
        treatment, outcome, coords, df, binary_treatment=binary_treatment
    )

    counterfactuals_spatial = []
    for tval in tvals:
        counterfactuals_spatial.append(
            outcome + beta_spatial * np.squeeze(tval - treatment, axis=-1)
        )
    counterfactuals_spatial = np.stack(counterfactuals_spatial, axis=1)

    evaluator = DatasetEvaluator(dataset)

    if binary_treatment:
        err_spatial_eval = evaluator.eval(
            ate=beta_spatial, counterfactuals=counterfactuals_spatial
        )
    else:
        erf_spatial = counterfactuals_spatial.mean(0)
        err_spatial_eval = evaluator.eval(
            erf=erf_spatial, counterfactuals=counterfactuals_spatial
        )

    # this is because json cannot serialize numpy arrays
    for key, value in err_spatial_eval.items():
        if isinstance(value, np.ndarray):
            err_spatial_eval[key] = value.tolist()

    res = {}
    res.update(**err_spatial_eval)
    res["beta"] = beta_spatial
    res["smoothness"] = dataset.smoothness_of_missing
    res["confounding"] = dataset.confounding_of_missing

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    method_choices = ["spatial", "spatial_plus"]
    parser.add_argument("--method", choices=method_choices, default="spatial_plus")
    args = parser.parse_args()

    start = time.perf_counter()

    datamaster = DataMaster()
    envs = datamaster.list_datasets()

    filename = f"results_{args.method}.jsonl"


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
                executor.submit(run_spatial_plus, dataset, binary, args.method)
                for dataset in dataset_list  # REMOVE [:1] FOR THE FULL RUN
            }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode="a") as writer:
                    result["envname"] = envname
                    writer.write(result)

    finish = time.perf_counter()

    print(f"Finished in {round(finish-start, 2)} second(s)")
