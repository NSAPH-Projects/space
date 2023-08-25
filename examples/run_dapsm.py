import argparse
import concurrent.futures
import os
import time

import jsonlines

from spacebench import DataMaster, DatasetEvaluator, SpaceDataset, SpaceEnv
from spacebench.algorithms import dapsm


def run_dapsm(
    dataset: SpaceDataset,
    **kwargs,
):
    treatment = dataset.treatment
    tvals = dataset.treatment_values

    # Call DAPSM
    model = dapsm.DAPSm(0.5)
    model.fit(dataset)
    effects = model.eval(dataset)

    # Compute counterfactuals
    ate = effects["ate"]

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
