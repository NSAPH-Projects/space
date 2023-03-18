#%% load packages ----
import pandas as pd
import numpy as np
from causalinference import CausalModel

import os
import json
import sys
import subprocess
import argparse

#%%

#%%
def main(args):
    df = pd.read_csv(args.dataset)

    # Rename columns
    df = df.rename({'pred': 'outcome'}, axis=1)

    # Create the treatment variable, and change boolean values to 1 and 0
    df['treatment'] = df['treatment'].apply(lambda x: 1 if x >= 12 else 0)
    X_cols = [f"X{i}" for i in list(range(1,35))]

    # Run causal model
    causal = CausalModel(
        Y = df['outcome'].values,
        D = df['treatment'].values, 
        X = df[X_cols].values
    )

    # Print summary statistics
    print(causal.summary_stats)

    # OLS treatment estimation adj=0
    causal.est_via_ols(adj=0, rcond=None)
    print('adj=0', causal.estimates)
    # OLS treatment estimation adj=1
    causal.est_via_ols(adj=1, rcond=None)
    print('adj=1', causal.estimates)
    # OLS treatment estimation adj=2
    causal.est_via_ols(adj=2, rcond=None)
    print('adj=2', causal.estimates)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = './medisynth-xgboost-binary-sample.csv')
    args = parser.parse_args()
    main(args)