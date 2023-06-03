import numpy as np
import pandas as pd
from spacebench.algorithms import spatialplus
from spacebench.algorithms import spatial
from spacebench import (
    SpaceDataset,
    DatasetEvaluator,
)
import csv 

def erf_spatial(dataset:SpaceDataset):
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
    for tval in tvals:
        dfpred_y = pd.DataFrame(np.column_stack((covariates, np.full_like(dataset.treatment[:, None], tval)-xpred.values.reshape(-1,1))), 
                    columns= covnames + ['r_X'])
        counterfactuals_spatialplus.append(fit_bs_y.predict(dfpred_y, coords))
    counterfactuals_spatialplus = np.stack(counterfactuals_spatialplus, axis=1)

    evaluator = DatasetEvaluator(dataset)
    erf_spatial = counterfactuals_spatial.mean(0)
    erf_spatialplus = counterfactuals_spatialplus.mean(0)
    err_spatial = evaluator.eval(erf=erf_spatial)
    err_spatialplus = evaluator.eval(erf=erf_spatialplus)
    erf_error_spatial = np.square(err_spatial["erf_error"]).mean()
    erf_error_spatialplus = np.square(err_spatialplus["erf_error"]).mean()

    smoothness = dataset.smoothness_of_missing
    confounding = dataset.confounding_of_missing

    # Write results to out.csv
    with open('out.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([smoothness, confounding, erf_error_spatial, erf_error_spatialplus])
    return np.array([smoothness, confounding, erf_error_spatial, erf_error_spatialplus])