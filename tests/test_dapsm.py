from spacebench.algorithms import dapsm
import pandas as pd


toydata3 = pd.read_csv("tests/dapsm_toydata.csv")
treated = toydata3[toydata3["Z"] == 1]
control = toydata3[toydata3["Z"] == 0]
toydata3.drop(toydata3.columns[[0, 1]], axis=1, inplace=True)


# Trying out the dist_ps function
daps = dapsm.dist_ps(
    treated=treated,
    control=control,
    caliper=0.05,
    coords_columns=[4, 5],  # distance = StandDist,
    caliper_type="DAPS",  # 'PS'
    matching_algorithm="optimal",
)


dapsm.DAPSopt(
    toydata3,
    caliper=0.5,
    caliper_type="DAPS",
    matching_algorithm="optimal",
    coords_cols=[3, 4],
    cov_cols=[5, 6],
    cutoff=0.5,
    trt_col=0,
    w_tol=0.01,
    distance=dapsm.StandDist,
    quiet=False,
    coord_dist=False,
    remove_unmatchables=False,
)
