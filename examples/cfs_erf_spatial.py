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

def erf_spatial(dataset:SpaceDataset, pid:int = 0, envname:str = '', filename:str = ''):
    """Helper function for parallelization notebook used to calculate the ERF and return errors for the spatial and spatialplus algorithms.
    
    Arguments
    ---------
    dataset: SpaceDataset
        A SpaceDataset object.
        
    Returns
    -------
    counterfactuals: np.ndarray
        A n x m matrix of counterfactuals, where n is the number of units and m is the number of treatment values.
    """

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
    beta = spatial.fit(treatment, outcome, coords, df)
    counterfactuals_spatial = []
    for tval in tvals:
        #dfpred_y = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval))), 
                        #columns= covnames + ['X'])
        counterfactuals_spatial.append(outcome + beta*np.squeeze(tval-treatment, axis = -1))
    counterfactuals_spatial = np.stack(counterfactuals_spatial, axis=1)


    beta = spatialplus.fit(treatment, outcome, coords, df, binary_treatment=False)
    counterfactuals_spatialplus = []
    for tval in tvals: 
        #dfpred_y = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval)
                                                 #-treatment)),
                    #columns= covnames + ['r_X'])
        counterfactuals_spatialplus.append(outcome + beta*np.squeeze(tval-treatment, axis = -1))
    counterfactuals_spatialplus = np.stack(counterfactuals_spatialplus, axis=1)

    evaluator = DatasetEvaluator(dataset)
    erf_spatial = counterfactuals_spatial.mean(0)
    erf_spatialplus = counterfactuals_spatialplus.mean(0)
    err_spatial_eval = evaluator.eval(erf=erf_spatial, counterfactuals=counterfactuals_spatial)
    err_spatialplus_eval = evaluator.eval(erf=erf_spatialplus, counterfactuals=counterfactuals_spatialplus)
    
    erf_error_spatial = err_spatial_eval["erf_error"]
    erf_error_spatialplus = err_spatialplus_eval["erf_error"]
    erf_av_error_spatial = err_spatial_eval["erf_av"]
    erf_av_error_spatialplus = err_spatialplus_eval["erf_av"]
    pehe_spatial = err_spatial_eval["pehe_curve"]
    pehe_spatialplus = err_spatialplus_eval["pehe_curve"]
    pehe_av_spatial = err_spatial_eval["pehe_av"]
    pehe_av_spatialplus = err_spatialplus_eval["pehe_av"]


    smoothness = dataset.smoothness_of_missing
    confounding = dataset.confounding_of_missing

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
                            pehe_av_spatialplus
                            ]) 
    return 