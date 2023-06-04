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
    fit_bs_y = spatial.fit(treatment, outcome, coords, df)
    counterfactuals_spatial = []
    for tval in tvals:
        dfpred_y = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval))), 
                        columns= covnames + ['X'])
        counterfactuals_spatial.append(fit_bs_y.predict(dfpred_y, coords))
    counterfactuals_spatial = np.stack(counterfactuals_spatial, axis=1)

    # Create spatialplus cfs
    fit_bs_x, fit_bs_y = spatialplus.fit(treatment, outcome, coords, df, binary_treatment=False)
    dfpred_x = pd.DataFrame(covariates,
                    columns= covnames)
    xpred = fit_bs_x.predict(dfpred_x, coords)
    counterfactuals_spatialplus = []
    for tval in tvals: # change this!! can do directly.
        dfpred_y = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval)
                                                 -xpred.values.reshape(-1,1))), 
                    columns= covnames + ['r_X'])
        counterfactuals_spatialplus.append(fit_bs_y.predict(dfpred_y, coords))
        # Alternate method
        #mu_cf = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval))), 
                   # columns= covnames + ['r_X'])
        # mu = pd.DataFrame(np.column_stack((covariates, xpred.values.reshape(-1,1))), 
                   # columns= covnames + ['r_X'])   
        #ycf = y + (mu_cf - mu)
    counterfactuals_spatialplus = np.stack(counterfactuals_spatialplus, axis=1)

    evaluator = DatasetEvaluator(dataset)
    erf_spatial = counterfactuals_spatial.mean(0)
    erf_spatialplus = counterfactuals_spatialplus.mean(0)
    err_spatial = evaluator.eval(erf=erf_spatial)#, counterfactuals=counterfactuals_spatial)
    err_spatialplus = evaluator.eval(erf=erf_spatialplus)#, counterfactuals=counterfactuals_spatialplus)
    erf_error_spatial = err_spatial["erf_av"]
    erf_error_spatialplus = err_spatialplus["erf_av"]
    #pehe_spatial = err_spatial["pehe_av"]
    #pehe_spatialplus = err_spatialplus["pehe_av"]


    smoothness = dataset.smoothness_of_missing
    confounding = dataset.confounding_of_missing

    # Write results to file
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([envname, pid, smoothness, confounding, erf_error_spatial, erf_error_spatialplus])#, pehe_spatial, pehe_spatialplus]) # can save # dataset as well
    return 