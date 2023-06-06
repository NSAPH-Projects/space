import concurrent.futures
import jsonlines
import time
from spacebench.algorithms import spatial
import numpy as np
import pandas as pd
from spacebench import (
    SpaceEnv,
    SpaceDataset,
    DataMaster,
    DatasetEvaluator,
    EnvEvaluator,
)

def run_spatial(dataset, binary_treatment):
    treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    # Scale covariates
    covariates = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    outcome = dataset.outcome
    coords = np.array(dataset.coordinates)

    # Scale coordinates
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    covnames = ['cov' + str(i+1) for i in range(covariates.shape[1])]
    df = pd.DataFrame(np.column_stack((coords, covariates, treatment, outcome)), 
                    columns=['coord1', 'coord2'] + covnames + ['X', 'Y'])
    tvals = dataset.treatment_values

    # Create spatial cfs
    beta_spatial = spatial.fit(treatment, outcome, coords, df)

    counterfactuals_spatial = []
    for tval in tvals:
        counterfactuals_spatial.append(outcome + beta_spatial*np.squeeze(tval-treatment, axis = -1))
    counterfactuals_spatial = np.stack(counterfactuals_spatial, axis=1)

    evaluator = DatasetEvaluator(dataset)

    if binary_treatment: # THERE SEEMS TO BE A PROBLEM HERE
        err_spatial_eval = evaluator.eval(ate=beta_spatial, counterfactuals=counterfactuals_spatial)
    else:
        erf_spatial = counterfactuals_spatial.mean(0)
        err_spatial_eval = evaluator.eval(
            erf=erf_spatial, counterfactuals=counterfactuals_spatial)
    
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


if __name__ == '__main__':
    start = time.perf_counter()

    datamaster = DataMaster()
    datasets = datamaster.master 

    filename = 'results_spatial.csv'

    envs = datasets.index.values
    envs = envs # ADD [:1] FOR DEBUGGING


    # Clean the file
    with open(filename, 'w') as csvfile:
        pass

    for envname in envs:
        env = SpaceEnv(envname, dir="downloads")
        dataset_list = list(env.make_all())
        binary = True if "disc" in envname else False
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(
                run_spatial, dataset, binary) for dataset in 
                dataset_list 
                }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode='a') as writer:
                    result["envname"] = envname
                    writer.write(result)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
