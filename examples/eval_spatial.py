import numpy as np
import pandas as pd
from spacebench.algorithms import spatialplus
from spacebench.algorithms import spatial
from spacebench import (
    SpaceDataset,
    DatasetEvaluator,
)
import csv 
import os

def eval_func_spatial(dataset:SpaceDataset, 
                      pid:int = 0, 
                      envname:str = '', 
                      binary_treatment:bool = False):
    """Helper function for parallelization notebook used to calculate the ERF and return errors for the spatial and spatialplus algorithms.
    
    Arguments
    ---------
    dataset: SpaceDataset
        A SpaceDataset object.
    pid: int
        The id of the datset in the env.
    envname: str
        The name of the env.
    binary_treatment: bool
        Whether the treatment is binary or not.
        
    Returns
    -------
    
    """
    if binary_treatment:
        filename = 'binary.csv'
    else:
        filename = 'continuous.csv'
    
    if not os.path.exists(filename):
        open(filename, 'w').close()

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        results = pd.read_csv(filename, header=None)
        results = results.set_index([0,1])
        if (envname, pid) in results.index:
            return
    
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
    beta_spatial = spatial.fit(treatment, 
                               outcome, 
                               coords, 
                               df)
    counterfactuals_spatial = []
    for tval in tvals:
        counterfactuals_spatial.append(outcome + beta_spatial*np.squeeze(tval-treatment, axis = -1))
    counterfactuals_spatial = np.stack(counterfactuals_spatial, axis=1)

    # Create spatial plus cfs
    beta_spatialplus = spatialplus.fit(treatment, 
                                       outcome, 
                                       coords, 
                                       df, 
                                       binary_treatment=binary_treatment)
    counterfactuals_spatialplus = []
    for tval in tvals: 
        counterfactuals_spatialplus.append(outcome + beta_spatialplus*np.squeeze(tval-treatment, axis = -1))
    counterfactuals_spatialplus = np.stack(counterfactuals_spatialplus, axis=1)

    evaluator = DatasetEvaluator(dataset)
    smoothness = dataset.smoothness_score
    confounding = dataset.confounding_score

    if binary_treatment:
        eval_spatial = evaluator.eval(ate=beta_spatial)
        eval_spatialplus = evaluator.eval(ate=beta_spatialplus)
        ate_error_spatial = eval_spatial["ate_error"]
        ate_error_spatialplus = eval_spatialplus["ate_error"]
        ate_se_spatial = eval_spatial["ate_se"]
        ate_se_spatialplus = eval_spatialplus["ate_se"]

        # Write results to file
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([envname, 
                                pid, 
                                smoothness, 
                                confounding,
                                beta_spatial,
                                beta_spatialplus,
                                ate_error_spatial, 
                                ate_error_spatialplus, 
                                ate_se_spatial,
                                ate_se_spatialplus
                                ])

    else:
        erf_spatial = counterfactuals_spatial.mean(0)
        erf_spatialplus = counterfactuals_spatialplus.mean(0)
        err_spatial_eval = evaluator.eval(erf=erf_spatial, 
                                          ite=counterfactuals_spatial)
        err_spatialplus_eval = evaluator.eval(erf=erf_spatialplus, 
                                              ite=counterfactuals_spatialplus)
        
        erf_error_spatial = err_spatial_eval["erf_error"]
        erf_error_spatialplus = err_spatialplus_eval["erf_error"]
        erf_av_error_spatial = err_spatial_eval["erf_av"]
        erf_av_error_spatialplus = err_spatialplus_eval["erf_av"]
        pehe_spatial = err_spatial_eval["pehe_curve"]
        pehe_spatialplus = err_spatialplus_eval["pehe_curve"]
        pehe_av_spatial = err_spatial_eval["pehe_av"]
        pehe_av_spatialplus = err_spatialplus_eval["pehe_av"]


        # Write results to file
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([envname, 
                                pid, 
                                smoothness, 
                                confounding,
                                dataset.treatment_values,
                                dataset.erf(),
                                erf_spatial,
                                erf_spatialplus,
                                erf_error_spatial, 
                                erf_error_spatialplus, 
                                erf_av_error_spatial,
                                erf_av_error_spatialplus,
                                pehe_spatial, 
                                pehe_spatialplus,
                                pehe_av_spatial,
                                pehe_av_spatialplus,
                                treatment.mean(),
                                outcome.mean()
                                ]) 
    return 