import importlib
from spacebench.algorithms import dapsm
import numpy as np
import pandas as pd

df = pd.read_csv('tests/treated.csv')
treated = df.values
df = pd.read_csv('tests/control.csv')
control = df.values
df = pd.read_csv('tests/dist_mat.csv')
dist_mat = df.values
df = pd.read_csv('tests/ps_diff.csv')
ps_diff = df.values

# Trying out the dist_ps function
daps = dapsm.dist_ps(dist_mat, ps_diff, caliper=2, caliper_type='DAPS', distance=dapsm.StandDist,
                     weight=0.8,
                     matching_algorithm='optimal')

# Trying out the DAPSopt function
dapsout = dapsm.DAPSopt(treated, control, dist_mat, ps_diff, caliper=1, caliper_type='DAPS', matching_algorithm='optimal',
              cov_cols=[5, 6, 7, 8], cutoff=0.5,
              w_tol=0.2, distance=dapsm.StandDist,
              quiet=False)
