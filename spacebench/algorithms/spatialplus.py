import numpy as np
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd

def fit(
        X: np.ndarray,
        Y: np.ndarray,
        coord: np.ndarray,
        df: pd.DataFrame,
        binary_treatment: bool = False
        ):
    """Spatial plus algorithm: fitting.

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
    binary_treatment: bool
        Whether the exposure is binary or not.

    Returns
    -------
        fit_bs_x: statsmodels.gam.generalized_linear_model.GLMGamResultsWrapper
            The fitted model of x on covariates and coordinates.
        fit_bs_y: statsmodels.gam.generalized_linear_model.GLMGamResultsWrapper
            The fitted model of y on residuals from fit_bs_x, covariates, and coordinates.
    """
    # Make X and Y n x 1 matrices
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    covs = [col for col in df.columns if col not in ['Y', 'coord1', 'coord2', 'X']]

    bs = BSplines(coord, df=[5, 5], degree=[3, 3]) # df and deg inputs
    if binary_treatment: # binary exposure
        formula = f"X ~ {' + '.join(covs)}"
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother = bs, 
                        family=sm.families.Binomial())
        fit_bs_x = gam_bs.fit() # fit model without penalty
        alphax = gam_bs.select_penweight(criterion = "gcv")[0] # select penalty weight
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother=bs, alpha=alphax, 
                        family=sm.families.Binomial())
        
    else: # non-binary exposure
        formula = f"X ~ {' + '.join(covs)}"
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother = bs)
        fit_bs_x = gam_bs.fit() # fit model without penalty
        alphax = gam_bs.select_penweight(criterion = "gcv", method = 'minimize')[0] # select penalty weight
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother=bs, alpha=alphax)

    #bs = BSplines(coord, df=[10, 10], degree=[3, 3]) # df and deg inputs
    fit_bs_x = gam_bs.fit()
    r_X = fit_bs_x.resid_response
    df['r_X'] = r_X # save residuals
    formula = f"Y ~ r_X + {' + '.join(covs)}"
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs) # fit outcome model without penalty
    fit_bs_y = gam_bs.fit()
    alphay = gam_bs.select_penweight(criterion="gcv", method = 'minimize')[0] # select penalty
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs, alpha=alphay) # fit outcome model with penalty
    fit_bs_y = gam_bs.fit()
    return(fit_bs_x, fit_bs_y)


if __name__ == "__main__": # this is for testing
    import scipy.special
    np.random.seed(20)
    n = 10000
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    xx, yy = np.meshgrid(x, y)
    coord = np.column_stack((xx.ravel(), yy.ravel()))
    cov1 = np.random.rand(n) + 0.5
    cov2 = np.random.rand(n) - 0.5
    X = np.random.rand(n) + coord @ [0.5,0.7]  + np.square(coord) @ [0.3,-1]+ 2*cov1 - 0.5*cov2
    Xbin = np.random.binomial(size=n, n=1, p=scipy.special.expit(coord @ [0.5,0.7] + np.square(coord) @ [0.3,-1] + 2*cov1 - 0.5*cov2))
    Y = 0.3*X + np.random.rand(n) + np.square(coord) @ [-0.5,0.1] + np.power(coord,3) @ [-0.3,1]+ 0.5*cov1 - 3*cov2
    Ybin = 0.3*Xbin + np.random.rand(n) + np.square(coord) @ [-0.5,0.1] + np.power(coord,3) @ [-0.3,1]+ 0.5*cov1 - 3*cov2
    df = pd.DataFrame(np.column_stack((coord, cov1, cov2, X, Y)), 
                    columns=['coord1', 'coord2', 'cov1', 'cov2', 'X', 'Y'])
    dfbin = pd.DataFrame(np.column_stack((coord, cov1, cov2, Xbin, Ybin)),
                         columns=['coord1', 'coord2', 'cov1', 'cov2', 'X', 'Y'])
    print(fit(X,Y,coord, df)) # should return 0.3
    print(fit(Xbin,Ybin,coord, dfbin)) # should return 0.3