import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd

def fit(
        X: np.ndarray,
        Y: np.ndarray,
        coord: np.ndarray,
        df: pd.DataFrame,
        ):
    """Spatial algorithm.

    Arguments
    ---------
    X: np.ndarray 
        A vector of exposures (could be binary or continuous).
    Y: np.ndarray
        A vector of outcomes.
    coord: np.ndarray
        A 2 column matrix of coordinates (latitude and longitude).
    df: pd.DataFrame
        A dataframe of coordinates, X, and Y with corresponding column names 'coord1', 'coord2', 'X', 'Y'
        as well as covariates.

    Returns
    -------
        fit_bs_y: statsmodels.gam.generalized_linear_model.GLMGamResultsWrapper
            The fitted model.
    """
    # Make X and Y n x 1 matrices
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    covs = [col for col in df.columns if col not in ['Y', 'coord1', 'coord2', 'X']]

    bs = BSplines(coord, df=[10, 10], degree=[3, 3]) # df and deg inputs
    formula = f"Y ~ X + {' + '.join(covs)}"
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs) # fit outcome model without penalty
    fit_bs_y = gam_bs.fit()
    alphay = gam_bs.select_penweight(criterion="gcv")[0] # select penalty
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs, alpha=alphay) # fit outcome model with penalty
    fit_bs_y = gam_bs.fit()
    return(fit_bs_y)
