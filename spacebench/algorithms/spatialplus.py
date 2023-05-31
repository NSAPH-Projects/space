import numpy as np
from statsmodels.gam.api import GLMGam, BSplines
import pandas as pd
import statsmodels.api as sm

def spatial_plus(
        X: np.ndarray,
        Y: np.ndarray,
        coord: np.ndarray,
        df: pd.DataFrame,
        binary_treatment: bool = False
        ):
    """Spatial plus algorithm.

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
        coef: float
            The coefficient of the exposure in the outcome model.
    """
    # Make X and Y n x 1 matrices
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    covs = [col for col in df.columns if col not in ['Y', 'coord1', 'coord2', 'X']]

    bs = BSplines(coord, df=[10, 10], degree=[3, 3]) # df and deg inputs
    if binary_treatment: # binary exposure
        formula = f"X ~ {' + '.join(covs)}"
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother = bs, 
                        family=sm.families.Binomial())
        res_bs = gam_bs.fit() # fit model without penalty
        alphax = gam_bs.select_penweight(criterion = "gcv")[0] # select penalty weight
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother=bs, alpha=alphax, 
                        family=sm.families.Binomial())
        
    else: # non-binary exposure
        formula = f"X ~ {' + '.join(covs)}"
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother = bs)
        res_bs = gam_bs.fit() # fit model without penalty
        alphax = gam_bs.select_penweight(criterion = "gcv")[0] # select penalty weight
        gam_bs = GLMGam.from_formula(formula=formula, data = df, smoother=bs, alpha=alphax)

    res_bs = gam_bs.fit()
    r_X = res_bs.resid_response
    df['r_X'] = r_X # save residuals
    formula = f"Y ~ r_X + {' + '.join(covs)}"
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs) # fit outcome model without penalty
    res_bs = gam_bs.fit()
    alphay = gam_bs.select_penweight(criterion="gcv")[0] # select penalty
    gam_bs = GLMGam.from_formula(formula=formula, data = df,
                                smoother=bs, alpha=alphay) # fit outcome model with penalty
    res_bs = gam_bs.fit()
    coef = res_bs.params[1]
    return(coef)

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
    print(spatial_plus(X,Y,coord, df)) # should return 0.3
    print(spatial_plus(Xbin,Ybin,coord, dfbin)) # should return 0.3