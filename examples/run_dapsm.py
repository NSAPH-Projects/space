import os
import time
import argparse
import concurrent.futures

import jsonlines
import numpy as np
from sklearn.linear_model import SGDClassifier

from spacebench import (
    SpaceEnv,
    SpaceDataset,
    DataMaster,
    DatasetEvaluator,
)
from spacebench.algorithms import dapsm


def run_dapsm(
    dataset: SpaceDataset,
    **kwargs,
):
    treatment = dataset.treatment
    covariates = dataset.covariates
    tvals = dataset.treatment_values
    tind = (treatment == tvals[1])
    
    # Fit propensity score model
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.1, max_iter=500)
    covars_ = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    model.fit(covars_, tind)
    propensity = model.predict_proba(covariates)[:, 1]

    # Compute distances
    coords = dataset.coordinates
    dists2 = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[1]):
        dists2 += (coords[:, i][:, None] - coords[:, i][None, :]) ** 2
    distmat_full = np.sqrt(dists2) / np.sqrt(dists2).max()

    # Call DAPSM
    model = dapsm.DAPSm(
        causal_dataset=dataset,
        ps_score=propensity,
        spatial_dists_full=distmat_full,
        balance_cutoff=np.inf,
    )

    # Compute counterfactuals
    ate, *_ = model.estimate("att")

    evaluator = DatasetEvaluator(dataset)
    res = evaluator.eval(ate=ate)
    res["smoothness"] = dataset.smoothness_score
    res["confounding"] = dataset.confounding_score
    res.update(**kwargs)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    method_choices = ["dapsm"]
    parser.add_argument("--method", choices=method_choices, default="dapsm")
    args = parser.parse_args()

    start = time.perf_counter()

    datamaster = DataMaster()
    envs = datamaster.list_envs(binary=True)

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
            spatial_score = dataset.smoothness_score
            confounding_score = dataset.confounding_score
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
                executor.submit(run_dapsm, dataset, dataset_num=i)
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
