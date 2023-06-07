import concurrent.futures
import os
import jsonlines
import time
import numpy as np
import argparse
from spacebench import (
    SpaceEnv,
    SpaceDataset,
    DataMaster,
    DatasetEvaluator,
)
from typing import Literal
import libpysal as lp
from pysal.model.spreg import GM_Lag, GM_Error, ML_Error, ML_Lag
# import scaler 
from sklearn.preprocessing import StandardScaler


# Define a function that takes an option and returns the result
def process_method(method):
    options = {
        "GM_Lag": GM_Lag,
        "GM_Error": GM_Error,
        "ML_Error": ML_Error,
        "ML_Lag": ML_Lag,
    }
    kwargs = {
        "GM_Lag": {"vm": False},
        "GM_Error": {"vm": False},
        "ML_Error":{"vm": False, "method": "ord", "epsilon": 1e-3},
        "ML_Lag": {"vm": False, "method": "ord", "epsilon": 1e-3},
    }
    return options.get(method), kwargs.get(method)


def run_pysal_reg(
    dataset: SpaceDataset,
    binary_treatment: bool,
    method_name: Literal["GM_Lag", "GM_Error", "ML_Error", "ML_Lag"],
):
    # # Add a super tiny random walk to the matrix for stability
    # # and avoid crashes with disconnected graphs
    # # The random walk must not have self loops
    A = dataset.adjacency_matrix()
    eps = 1e-6
    n = A.shape[0]
    rwalk = np.ones((n, n)) / (n - 1)
    rwalk[np.diag_indices(n)] = 0
    A = (1 - eps) * A + eps * rwalk
    W = lp.weights.util.full2W(A)

    # # Make coordinate distance matrix
    # coords = dataset.coordinates
    # dists2 = np.zeros((coords.shape[0], coords.shape[0]))
    # for i in range(coords.shape[1]):
    #     dists2 += (coords[:, i][:, None] - coords[:, i][None, :]) ** 2
    # distmat_full = np.sqrt(dists2) / np.sqrt(dists2).max()
    # W = lp.weights.util.full2W(distmat_full)

    # Convert to a spatial weights object
    #W = lp.weights.WSP2W(sparse_matrix)

    #knn = lp.weights.KNN.from_adjlist # from_dataframe(db, k=1)

    #W = lp.weights.full2W(dataset.adjacency_matrix()) 
    treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    outcome = dataset.outcome

    tscaler = StandardScaler()
    treatment = tscaler.fit_transform(treatment)
    cscaler = StandardScaler()
    covariates = cscaler.fit_transform(covariates)
    oscaler = StandardScaler()
    outcome = oscaler.fit_transform(outcome[:, None])[:, 0]


    # add noise to the synt outcome
    # outcome = outcome + 0.001 * np.random.random(outcome.shape)
    # covariates = covariates + 0.001 * np.random.random(covariates.shape)

    # make train matrix
    trainmat = np.hstack([covariates, treatment])

    method, kwargs = process_method(method_name)
    model = method(outcome,trainmat,w=W, **kwargs)

    counterfactuals = []
    tvals = dataset.treatment_values
    for tval in tvals:
        treatment_beta = model.betas[-2]
        diff = np.squeeze(tval-treatment, axis=-1)
        counterfactuals.append(outcome + treatment_beta*(diff))
    counterfactuals = np.stack(counterfactuals, axis=1)

    evaluator = DatasetEvaluator(dataset)

    if binary_treatment: 
        err_eval = evaluator.eval(ate=treatment_beta, counterfactuals=counterfactuals)
    else:
        erf = counterfactuals.mean(0)
        err_eval = evaluator.eval(
            erf=erf, counterfactuals=counterfactuals)

    # this is because json cannot serialize numpy arrays
    for key, value in err_eval.items():
        if isinstance(value, np.ndarray):
            err_eval[key] = value.tolist()

    res = {}
    res.update(**err_eval)
    res["beta"] = treatment_beta.tolist() * tscaler.scale_
    res["smoothness"] = dataset.smoothness_of_missing
    res["confounding"] = dataset.confounding_of_missing

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    method_choices = ["GM_Lag", "GM_Error", "ML_Error", "ML_Lag", "Ridge"]
    parser.add_argument("--method", choices=method_choices, default="ML_Lag")
    args = parser.parse_args()

    start = time.perf_counter()

    datamaster = DataMaster()
    envs = datamaster.list_datasets()

    filename = f"results/results_{args.method}.jsonl"
    print(f"Method {args.method}")

    # Clean the file
    if args.overwrite:
        if os.path.exists(filename):
            os.remove(filename)

    for envname in envs[3:4]: # reversed(envs):
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
                executor.submit(run_pysal_reg, dataset, binary, args.method)
                for dataset in dataset_list  
            }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode="a") as writer:
                    result["envname"] = envname
                    writer.write(result)

        # helper = []
        # for dataset in dataset_list:
        #     missing = dataset.missing
        #     smoothness = dataset.smoothness_of_missing
        #     confounding = dataset.confounding_of_missing
        #     with open('results/mapping.txt', 'a') as f:
        #         f.write(f'{envname},{missing},{smoothness},{confounding}\n')
        #     #helper.append([])


    finish = time.perf_counter()

    print(f"Finished in {round(finish-start, 2)} second(s)")