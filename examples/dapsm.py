#%%
import numpy as np
import pandas as pd

import spacebench.algorithms.dapsm as dapsm
from spacebench import CausalDataset

#%%
df = pd.read_csv("examples/data/dapsm_toydata.csv", index_col=0)
X = df[[c for c in df.columns if c.startswith("X")]].values
A = df.Z.values
beta = np.ones(X.shape[1])
Y0 = X @ beta
Y1 = Y0 + 10
Y = Y0 * (1 - A) + Y1 * A
causal_dataset = CausalDataset(
    treatment=A,
    covariates=X,
    outcome=Y,
    counterfactuals=np.stack([Y0, Y1], axis=1),
)
ps_score = df.prop_scores.values
long = df.long.values
lat = df.lat.values
distmat_full = np.sqrt((long[:, None] - long[None, :]) ** 2 + (lat[:, None] - lat[None, :]) ** 2)

# %%
method = dapsm.DAPSm(
    causal_dataset=causal_dataset,
    ps_score=ps_score,
    spatial_dists_full=distmat_full,
    balance_cutoff=0.4,
)
att, weight, matches = method.estimate(metric="att")
print("Estimated ATT:", att)
print("True ATT:", np.mean(Y1 - Y0))
print("Found weight:", weight)
print("Number of matches:", len(matches.pairs()))
