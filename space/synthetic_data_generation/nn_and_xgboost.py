#%%
# for scripting and data manipulation
import seaborn as sns
import pickle
import os
import requests
import zipfile
from pyDataverse.api import NativeApi, DataAccessApi
from tqdm import tqdm
import random
import logging
import json

# data manipuation
import numpy as np
import pandas as pd
import geopandas as gpd

# models for data fitting
from xgboost import XGBRegressor, plot_importance
import torch
import torch.nn as nn
from net import CausalNet
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx


# for hyper-parameter optimization
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler, LabelBinarizer

# for fitting error distributions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("datasets", exist_ok=True)
# %%
random.seed(110104)
np.random.seed(110104)
torch.manual_seed(110104)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

#%% Download Synth Medicare data for the condounders and exposure
if not os.path.exists("datasets/Study_dataset_2010.tab"):
    url = "https://dataverse.harvard.edu"
    api = NativeApi(url)
    data_api = DataAccessApi(url)
    DOI = "doi:10.7910/DVN/L7YF2G"
    dataset = api.get_dataset(DOI)

    files_list = dataset.json()['data']['latestVersion']['files']
    for file in files_list:
        filename = file["dataFile"]["filename"]
        file_id = file["dataFile"]["id"]
        print(f"Downloaded file datasets/{filename}, id {file_id}")
        response = data_api.get_datafile(file_id)
        with open("datasets/" + filename, "wb") as f:
            f.write(response.content)

#%% Download the us census polygons
if not os.path.exists("datasets/tl_2010_us_county10/tl_2010_us_county10.shp"):
    url = "https://www2.census.gov/geo/tiger/TIGER2010/COUNTY/2010/tl_2010_us_county10.zip"
    response = requests.get(url)
    with open("datasets/tl_2010_us_county10.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile("datasets/tl_2010_us_county10.zip", "r") as zip_ref:
        zip_ref.extractall("datasets/tl_2010_us_county10")

# %% Load the CDC all cause morality data
# Users should manually download these data from https://wonder.cdc.gov/controller/datarequest/D77
# Using the options:
#    (1) Group results by: county;
#    (3) In demographics choose the options of >65 years old.
#    (4) Year 2010.
#    (7) ICD codes I00-I99, J00-J98
#    (8) Send results to a file. 
# Click send and save the file to "./datasets/cdc.tsv".
mort = pd.read_csv("./datasets/cdc.tsv", sep="\t", dtype={"County Code": "object"})
mort[mort.Deaths.isin(["Suppressed"])] = np.nan
mort[mort.Deaths.isin(["Missing"])] = np.nan
mort["Deaths"] = mort["Deaths"].astype(float)
mort["Population"] = mort["Population"].astype(float)
mort["cdc_mortality_pct"] = 1e3 * mort["Deaths"] / mort["Population"]
mort = mort.rename({"County Code": "FIPS"}, axis=1).set_index("FIPS")

#%% Read confounder and exposure data
df = pd.read_csv("datasets/Study_dataset_2010.tab", index_col=0, sep="\t", dtype={"FIPS": object})
id_vars = ["NAME", "STATE_CODE", "STATE"]
discrete_vars = ["region"]
drop_vars = ["cs_total_population", "cs_area"]
df_id = df[id_vars]
df_discrete = []
for c in discrete_vars:
    col = df[c]
    lb = LabelBinarizer()
    lb.fit(col)
    bcols = pd.DataFrame(lb.transform(col), columns=["bin_" + x for x in lb.classes_], index=df.index)
    df_discrete.append(bcols.drop(columns="bin_" + lb.classes_[0]))
df_discrete = pd.concat(df_discrete, axis=1)
df = df.drop(columns=id_vars + discrete_vars + drop_vars)
df = df.merge(mort, how="left", right_index=True, left_index=True)
df = pd.concat([df, df_discrete], axis=1)

# %% Read Shapefile and merge
counties = gpd.read_file('datasets/tl_2010_us_county10/tl_2010_us_county10.shp')
counties = counties[~counties.STATEFP10.isin(["02", "72", "78", "15"])]  # remove alaska, pr, vi, hawaii
# counties = counties.merge(df, how="left", left_on="GEOID10", right_index=True)
# counties = counties.merge(mort[["Deaths", "Population"]], how="left", left_on="GEOID10", right_index=True)
# counties["cdc_mortality_pct"] = counties.Deaths / counties.Population
# counties.head().copy()
L = counties[["GEOID10", "INTPTLAT10", "INTPTLON10"]].set_index("GEOID10")
L["INTPTLAT10"] = L["INTPTLAT10"].astype("float")
L["INTPTLON10"] = L["INTPTLON10"].astype("float")
df = df.merge(L, right_index=True, left_index=True)
counties = counties.merge(df, how='left', left_on='GEOID10', right_index=True)

#%% Extract Confounders, Exposure, Outcome, Location
A = df["qd_mean_pm25"]
A_binary = (A >= 12.0).astype(float)
cols = [c for c in df.columns if c.startswith(("cdc", "cs", "gmet", "bin_")) and not c.endswith("mortality_pct")]
X = df[cols].copy()
y = df["cdc_mortality_pct"]
L = df[["INTPTLAT10", "INTPTLON10"]]

#%% Create adjacency graph
nodes = df.index.values
adj = np.zeros((len(nodes), len(nodes)))  # adj,mat
index_map = {x: i for i, x in enumerate(nodes)}

for _, row in counties.iterrows():
    idx = index_map[row.GEOID10]
    nbrs = counties[counties.geometry.touches(row.geometry)].GEOID10.values
    nbr_idxs = [index_map[n] for n in nbrs]
    adj[nbr_idxs, idx] = 1

#%% Compute Moran's I for the covariates to measure spatial smoothness
def compute_moran_i(values, adjacency_matrix):
    n = len(values)
    w = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)  # row-standardize adjacency matrix
    mean_val = np.mean(values)
    z = (values - mean_val) / np.std(values)  # standardize values
    numerator = np.sum(w * np.outer(z, z.T))
    denominator = np.sum(z ** 2)
    moran_i = (n / np.sum(adjacency_matrix)) * (numerator / denominator)

    return moran_i

moran_i = {}
for c in cols:
    moran_i[c] = compute_moran_i(df[c].values, adj)
sorted(moran_i.items(), key=lambda x: x[1])

#%% Save dataset and moran_i computations
pd.concat([L, A, X, y], axis=1).to_csv("datasets/processed_dataset.csv")

##% Split train test for fitting models
mask = (~np.isnan(X.values).any(1)) & (~np.isnan(y.values))
splits = train_test_split(X.iloc[mask], A.iloc[mask], y.iloc[mask], L.iloc[mask], test_size=0.1)
train_X, test_X, train_A, test_A, train_y, test_y, train_L, test_L = splits

#%% train and optimize xgobost model using 
# train_D = xgb.DMatrix(np.concatenate([train_A, train_X], axis=1), label=train_y)
# test_D = xgb.DMatrix(np.concatenate([test_A, test_X], axis=1), label=test_y)
# D = xgb.DMatrix(np.concatenate([A.to_numpy()[:, None], X.to_numpy()], axis=1))

#%%
# weights = [100.0] + list(np.full((X.shape[1], ), 0.5/X.shape[1]))
params = {
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [1, 2, 3],
    'n_estimators': [100, 1000, 2000],
    'min_child_weight': [1, 3, 5],
}

param_sampler = ParameterSampler(params, n_iter=20)

#%% ======== Tune and fit xgboost model ===========
print("Optimizing xgboost model...")
results = []
train_Z = pd.concat([train_A, train_X], axis=1)
test_Z = pd.concat([test_A, test_X], 1)
# train_Z = np.concatenate([train_A, train_X], 1)
# test_Z = np.concatenate([test_A, test_X], 1)

for p in tqdm(param_sampler):
    model = XGBRegressor(**p, random_state=110104)
    model.fit(train_Z, train_y)
    pred = model.predict(test_Z)
    # error = np.mean((pred - test_y)**2)
    error = np.mean((pred - test_y.values)**2)
    p["val_mse"] = error
    print(p)
    results.append(p)
opt_pars = min(results, key=lambda x: x["val_mse"])
print("best params: ", min(results, key=lambda x: x["val_mse"])
)

#%% evaluate prediction and counterfactuals
pars = opt_pars.copy()
pars.pop("val_mse")
model = XGBRegressor(**pars, random_state=110104, importance_type='gain')
model_binary = XGBRegressor(**pars, random_state=110104, importance_type='gain')
# Z = np.concatenate([scaler_A.transform(A.values[:, None]), scaler_X.transform(X.values)], 1)
Z = pd.concat([A, X], axis=1)
Z_binary = pd.concat([A_binary, X], axis=1)
model.fit(Z[mask], y.values[mask])
model_binary.fit(Z_binary[mask], y.values[mask])

# main xgboost prediction
mu_xgb = model.predict(Z)
mu_xgb_binary = model_binary.predict(Z_binary)
# mu_xgb = scaler_y.inverse_transform(mu_xgb[:, None])[:, 0]

# make all counterfactuals
mu_xgb_cf = []
mu_xgb_cf_binary = []
a_grid = np.linspace(A.min(), A.max(), num=100)
for a in tqdm(a_grid):
    Za = Z.copy()
    Za.iloc[:, 0] = a
    mu_xgb_cf.append(model.predict(Za))

for a in (0, 1):
    Za = Z.copy()
    Za.iloc[:, 0] = a
    mu_xgb_cf_binary.append(model_binary.predict(Za))

mu_xgb_cf = np.stack(mu_xgb_cf, 1)
mu_xgb_cf_binary = np.stack(mu_xgb_cf_binary, 1)
ate = (mu_xgb_cf_binary[:, 1] - mu_xgb_cf_binary[:, 0]).mean()
print("ATE (xgboost binary):", ate)

#%%
plot_importance(model)
variable_importance = {c: v for c, v in zip(Z.columns, model.feature_importances_)}

# %%
nplots = 100
for i in range(nplots):
    plt.plot(a_grid, mu_xgb_cf[i, :], c="gray", alpha=0.1)
plt.scatter(A[:nplots], mu_xgb[:nplots], c="red", alpha=0.25)

#%%
plt.scatter(mu_xgb, y, alpha=0.1)

#%% Fit a GP to the error distribution
res = (y.values - mu_xgb)[mask]
scaler_rx = StandardScaler()
RX = pd.concat([L], axis=1)[mask]
scaler_rx.fit(RX)
RX = scaler_rx.transform(RX)
# kernel = ConstantKernel(constant_value=0.0001, constant_value_bounds=(1e-4, 10)) * RBF(length_scale=np.ones(RX.shape[1]), length_scale_bounds=(1e-2, 1.0))
kernel = ConstantKernel(constant_value=0.0001, constant_value_bounds=(1e-4, 10)) * RBF(length_scale=[0.1, 0.1], length_scale_bounds=(1e-4, 0.1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.05, optimizer='fmin_l_bfgs_b')
gp.fit(RX, res)
print("GP Kernel xgb: ", gp.kernel_)
# res_sim = gp.sample_y(RX)
# sns.histplot(res_sim)
# sns.histplot(res)
kernel_xgb = gp.kernel_.__dict__


#%%
gp1 = GaussianProcessRegressor(kernel=gp.kernel_ + WhiteKernel(0.001))
res_xgb = gp1.sample_y(RX)[:, 0]
ytilde_xgb = mu_xgb[mask] + res_xgb

#%%
res_xgb_ = pd.Series(res_xgb, index=df.index[mask], name="residuals1")
#%%
res_true_ = pd.Series(res, index=df.index[mask], name="residuals_true")
counties = counties.merge(res_true_, left_on="GEOID10", right_index=True)
counties.plot(column="residuals_true", cmap="RdBu", vmin=-1, vmax=1)

#%%
plt.figure(figsize=(5, 4))
ax = sns.histplot(ytilde_xgb, label="synthetic outcome", lw=0, alpha=0.5)
sns.histplot(y, label="real outcome", lw=0, alpha=0.5)
ax.set(xlabel=None, ylabel=None)
plt.legend()
plt.savefig("marginal-outcome.png", bbox_inches="tight")

#%%
print("Real:", compute_moran_i(y[mask], adj[mask][:, mask]))
print("Synth:", compute_moran_i(ytilde_xgb, adj[mask][:, mask]))
new_col = pd.DataFrame({"synth": ytilde_xgb}, index=df.index[mask])
tmp = counties.merge(new_col, left_on="GEOID10", right_index=True)
tmp.plot(column="synth", cmap="RdBu", vmin=3, vmax=18)
plt.axis("off")
plt.savefig("map_synth.png", bbox_inches="tight")
#%%
tmp.plot(column="cdc_mortality_pct", cmap="RdBu", vmin=3, vmax=18)
plt.axis("off")
plt.savefig("map_real.png", bbox_inches="tight")

#%% ======== Tune and fit the NN model =============
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_L = StandardScaler()

scaler_X.fit(train_X)
scaler_y.fit(pd.DataFrame(train_y))
scaler_L.fit(train_L)

train_X, test_X = scaler_X.transform(train_X), scaler_X.transform(test_X)
train_y, test_y = scaler_y.transform(pd.DataFrame(train_y))[:, 0], scaler_y.transform(pd.DataFrame(test_y))[:, 0]
train_L, test_L = scaler_L.transform(train_L), scaler_L.transform(test_L)

train_A_ = torch.FloatTensor(train_A)
train_X_ = torch.FloatTensor(train_X)
train_y_ = torch.FloatTensor(train_y)
test_A_ = torch.FloatTensor(test_A)
test_X_ = torch.FloatTensor(test_X)
test_y_ = torch.FloatTensor(test_y)

ds_train = TensorDataset(train_A_, train_X_, train_y_)
ds_test = TensorDataset(test_A_, test_X_, test_y_)

#%% make params
params = {
    "d": [X.shape[1]],
    "hidden": [16],
    "act": ['silu'],
    "body_layers": [2],
    "weight_decay": [0.01],
    "tmin": [A.min()],
    "tmax": [A.max()],
    "lr": [1e-3],
    "degree": [3],
    "knots": [[0.33, 0.66]]
}

param_sampler = ParameterSampler(params, n_iter=1)

max_epochs = 500
batch_size = 64

def make_config(param):
    cfg_body = [(param['d'], param['hidden'], 1, param['act'])]
    for _ in range(param['body_layers'] - 1):
        cfg_body.append((param['hidden'], param['hidden'], 1, param['act']))
    cfg_head = [(param['hidden'], param['hidden'], 1, param['act'])]
    cfg_head.append((param['hidden'], 1, 1, 'id'))
    return cfg_body, cfg_head


train_loader = DataLoader(ds_train, batch_size=batch_size)
test_loader = DataLoader(ds_test, batch_size=batch_size)



#%%
class MyModel(pl.LightningModule):
    def __init__(self, param: dict, binary=False):
        super().__init__()
        cfg_body, cfg_head = make_config(param)
        self.net = CausalNet(
            cfg_body,
            cfg_head,
            param['degree'],
            param['knots'],
            param['tmin'],
            param['tmax'],
            binary=binary
        )
        self.wd = param["weight_decay"]
        self.lr = param["lr"]

    def forward(self, a, x):
        return self.net(a, x)

    def training_step(self, batch, batch_idx):
        a, x, y = batch
        y_hat = self(a, x).squeeze(1)
        loss = nn.MSELoss()(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        a, x, y = batch
        y_hat = self(a, x).squeeze(1)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", float(loss), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)


#%%
print("Optimizing NN model...")
results = []
for p in tqdm(param_sampler):
    model = MyModel(p)
    trainer = pl.Trainer(max_epochs=max_epochs, auto_lr_find=True, logger=False, enable_model_summary=False)
    trainer.fit(model, train_loader, test_loader)
    pred = model(test_A_, test_X_)[:, 0]
    with torch.no_grad():
        error = ((pred - test_y_)**2).mean()
    p["val_mse"] = float(error)
    print(p)
    results.append(p)
opt_pars_nn = min(results, key=lambda x: x["val_mse"])
print("best params: ", min(results, key=lambda x: x["val_mse"])
)
#%% evaluate NN prediction and counterfactuals
pars = opt_pars_nn.copy()
pars.pop("val_mse")
model = MyModel(pars)
model_binary = MyModel(pars, binary=True)

# Z = np.concatenate([scaler_A.transform(A.values[:, None]), scaler_X.transform(X.values)], 1)
A_ = torch.FloatTensor(A.values)
A_binary_ = torch.FloatTensor(A_binary.values)
X_ = torch.FloatTensor(scaler_X.transform(X))
y_ = torch.FloatTensor(scaler_y.transform(y[:, None]))[:, 0]
loader = DataLoader(TensorDataset(A_[mask], X_[mask], y_[mask]), batch_size=batch_size)
loader_binary = DataLoader(TensorDataset(A_binary_[mask], X_[mask], y_[mask]), batch_size=64)
trainer = pl.Trainer(max_epochs=max_epochs, auto_lr_find=True, logger=False)
trainer.fit(model, loader)
trainer.fit(model_binary, loader_binary)

#%%
with torch.no_grad():
    mu_nn = model(A_, X_).numpy()
    mu_nn_binary = model(A_binary_, X_).numpy()
mu_nn = scaler_y.inverse_transform(mu_nn)[:, 0]
mu_nn_binary = scaler_y.inverse_transform(mu_nn_binary)[:, 0]


# %% make counterfacutals
mu_nn_cf = []
mu_nn_cf_binary = []
for a in a_grid:
    with torch.no_grad():
        pred = model(torch.full_like(A_, a), X_).numpy()
        pred = scaler_y.inverse_transform(pred)[:, 0]
    mu_nn_cf.append(pred)

for a in (0, 1):
    with torch.no_grad():
        pred_binary = model(torch.full_like(A_binary_, a), X_).numpy()
        pred_binary = scaler_y.inverse_transform(pred_binary)[:, 0]
        mu_nn_cf_binary.append(pred_binary)
        
mu_nn_cf = np.stack(mu_nn_cf, 1)
mu_nn_cf_binary = np.stack(mu_nn_cf_binary, 1)

ate = (mu_nn_cf_binary[:, 1] - mu_nn_cf_binary[:, 0]).mean()
print("ATE (NN binary):", ate)


#%%
nplots = 100
for i in range(nplots):
    plt.plot(a_grid, mu_nn_cf[i, :], c="gray", alpha=0.1)
plt.scatter(A[:nplots], mu_nn[:nplots], c="red", alpha=0.25)

#%%
plt.scatter(mu_nn, y, alpha=0.1)

#%% Fit a GP to the error distribution
res = (y.values - mu_nn)[mask]
scaler_rx = StandardScaler()
RX = pd.concat([L], axis=1)[mask]
scaler_rx.fit(RX)
RX = scaler_rx.transform(RX)
kernel_ = ConstantKernel(constant_value=0.0001, constant_value_bounds=(1e-4, 1)) * RBF(length_scale=[0.1, 0.1], length_scale_bounds=(1e-4, 1.0))
# kernel = ConstantKernel(constant_value=0.0001, constant_value_bounds=(1e-4, 10)) * RBF(length_scale=np.ones(RX.shape[1]), length_scale_bounds=(1e-2, 1.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, optimizer='fmin_l_bfgs_b')
gp.fit(RX, res)
print("GP Kernel NN: ", gp.kernel_)

#%%
gp1 = GaussianProcessRegressor(kernel=gp.kernel_ + WhiteKernel(0.001))
res_sim = gp1.sample_y(RX)[:, 0]
# sns.histplot(res_sim)
# sns.histplot(res)
kernel_nn = gp.kernel_.__dict__
# #%%
# plt.figure(figsize=(5, 4))
# sns.histplot(y, label="Real data")
# sns.histplot(mu_nn + res_sim, label="Synthetic data")
# plt.legend()

#%%
res_true_ = pd.Series(res, index=df.index[mask], name="residuals_nn_true")
counties = counties.merge(res_true_, left_on="GEOID10", right_index=True)
counties.plot(column="residuals_nn_true", cmap="RdBu", vmin=-1, vmax=1)

#%%
res_sim_ = pd.Series(res_sim, index=df.index[mask], name="res_sim_nn_true")
counties = counties.merge(res_sim_, left_on="GEOID10", right_index=True)
counties.plot(column="res_sim_nn_true", cmap="RdBu", vmin=-1, vmax=1)


# %% save all files
os.makedirs('outputs', exist_ok=True)
prefix = "medisynth"

# %%
def serialize_kernel(kernel):
    items = {}
    for _, v in kernel.items():
        key = type(v).__name__
        items[key] = {k: (v if not isinstance(v, np.ndarray) else list(v)) for k, v in v.__dict__.items()}
    # use json to transform to string
    return json.dumps(items)

# %%
# Save geodata
counties = gpd.read_file('datasets/tl_2010_us_county10/tl_2010_us_county10.shp')
counties = counties[~counties.STATEFP10.isin(["02", "72", "78", "15"])]
geom = counties.set_index("GEOID10").loc[df.index]
geodata = gpd.GeoDataFrame(L, geometry=geom.geometry)
geodata.to_file("outputs/counties.geojson", driver="GeoJSON")
geodata.to_file("outputs/counties.shp")

# %% Save it also as graphml
g = nx.Graph()
g.add_nodes_from(geom.index.values)
feats = [c for c in L.columns if c != "geometry"]
L_ = pd.DataFrame(scaler_rx.transform(L[feats]), columns=feats, index=L.index)

# make edge list from GeoPandas DataFrame
edge_list = []
for index, row in geom.iterrows():
    for f in feats:
        g.nodes[index][f] = L_.loc[index][f]
    nbrs = geom[geom.geometry.touches(row.geometry)].index.values
    for nbr in nbrs:
        edge_list.append((index, nbr))
g.add_edges_from(edge_list)

#%%
nx.write_graphml(g, "outputs/counties.graphml")

#%%
variable_importance_ = [float(variable_importance[k]) for k in X.columns]
moran_i_ = [float(moran_i[k]) for k in X.columns]
X_anon = X.copy()
X_anon.columns = [f"X{i}" for i in range(len(X.columns))]
A_anon = pd.Series(A, name="treatment")
A_binary_anon = pd.Series(A_binary, name="treatment")


# %% XGboost discrete version 
metadata = {
    "data_file": f"{prefix}-xgboost-binary.csv",
    "metadata_file": f"{prefix}-xgboost-binary.json",
    "geodata_file": "counties.geojson",
    "graph_file": "counties.graphml",
    # "spatial_type": "shapefile",
    "source_data": "medicare_synthetic",
    "predictor_model": "xgboost",
    "continuous_treatment": False,
    "treatment_vals": [0, 1],
    "error_type": "gp",
    "error_params": serialize_kernel(kernel_xgb),
    "variable_importance": variable_importance_,
    "variable_smoothness": moran_i_
}
pred_ = pd.Series(mu_xgb_binary, name='pred', index=A.index)
cf_ = pd.DataFrame(mu_xgb_cf_binary, columns=['predcf_0', 'predcf_1'], index=A.index)
out = pd.concat([A_binary_anon, X_anon, pred_, cf_], axis=1)
out.to_csv(f"outputs/{metadata['data_file']}")
with open(f"outputs/{metadata['metadata_file']}", 'w') as io:
    json.dump(metadata, io)


# %% XGboost continuous version 
metadata = {
    "data_file": f"{prefix}-xgboost-continuous.csv",
    "metadata_file": f"{prefix}-xgboost-continuous.json",
    "geodata_file": "counties.geojson",
    "graph_file": "counties.graphml",
    # "spatial_type": "shapefile",
    "source_data": "medicare_synthetic",
    "predictor_model": "xgboost",
    "continuous_treatment": True,
    "treatment_type": list(a_grid),
    "error_type": "gp",
    "error_params": serialize_kernel(kernel_xgb),
    "variable_importance": variable_importance_,
    "variable_smoothness": moran_i_
}
pred_ = pd.Series(mu_xgb, name='pred', index=A.index)
cf_ = pd.DataFrame(mu_xgb_cf, columns=[f'predcf_{d}' for d in range(len(a_grid))], index=A.index)
out = pd.concat([A_anon, X_anon, pred_, cf_], axis=1)
out.to_csv(f"outputs/{metadata['data_file']}")
with open(f"outputs/{metadata['metadata_file']}", 'w') as io:
    json.dump(metadata, io)

# %% NN discrete version 
metadata = {
    "data_file": f"{prefix}-nn-binary.csv",
    "metadata_file": f"{prefix}-nn-binary.json",
    "geodata_file": "counties.geojson",
    "graph_file": "counties.graphml",
    # "spatial_type": "shapefile",
    "source_data": "medicare_synthetic",
    "predictor_model": "tarnet",
    "continuous_treatment": False,
    "treatment_vals": [0, 1],
    "error_type": "gp",
    "error_params": serialize_kernel(kernel_nn),
    "variable_importance": variable_importance_,
    "variable_smoothness": moran_i_
}

pred_ = pd.Series(mu_nn_binary, name='pred', index=A.index)
cf_ = pd.DataFrame(mu_nn_cf_binary, columns=['predcf_0', 'predcf_1'], index=A.index)
out = pd.concat([A_binary_anon, X_anon, pred_, cf_], axis=1)
out.to_csv(f"outputs/{metadata['data_file']}")
with open(f"outputs/{metadata['metadata_file']}", 'w') as io:
    json.dump(metadata, io)

# %% NN continuous version 
metadata = {
    "data_file": f"{prefix}-nn-continuous.csv",
    "metadata_file": f"{prefix}-nn-continuous.json",
    "geodata_file": "counties.geojson",
    "graph_file": "counties.graphml",
    # "spatial_type": "shapefile",
    "source_data": "medicare_synthetic",
    "predictor_model": "vcnet",
    "continuous_treatment": True,
    "treatment_vals": list(a_grid),
    "error_type": "gp",
    "error_params": serialize_kernel(kernel_nn),
    "variable_importance": variable_importance_,
    "variable_smoothness": moran_i_
}
pred_ = pd.Series(mu_nn_binary, name='pred', index=A.index)
cf_ = pd.DataFrame(mu_nn_cf, columns=[f'predcf_{d}' for d in range(len(a_grid))], index=A.index)
out = pd.concat([A_anon, X_anon, pred_, cf_], axis=1)
out.to_csv(f"outputs/{metadata['data_file']}")
with open(f"outputs/{metadata['metadata_file']}", 'w') as io:
    json.dump(metadata, io)


def geopandas_to_networkx(geodata: gpd.GeoDataFrame) -> nx.Graph:
    """Converts a GeoPandas DataFrame to a NetworkX Graph"""
    # make graph with nodes from geodata index
    g = nx.Graph()
    g.add_nodes_from(geodata.index.values)

    # make edge list from GeoPandas DataFrame
    edge_list = []
    for _, row in geodata.iterrows():
        nbrs = geodata[geodata.geometry.touches(row.geometry)].index.values
        for nbr in nbrs:
            edge_list.append((row.index, nbr))


#%%


# metadata = {
#     "treatment": A,
#     "covariates": X,
#     "synthetic_outcome": y,
#     "models": {
#         "xgboost": {
#             "error_gp_kernel": kernel_xgb,
#             "predictions": mu_xgb,
#             "counterfactuals": mu_xgb_cf
#         },
#         "nn_vcnet": {
#             "error_gp_kernel": kernel_nn,
#             "predictions": mu_nn,
#             "counterfactuals": mu_nn_cf
#         },
#     },
#     "spatial_smoothness": moran_i,
#     "importance": variable_importance
# }
# with open("datasets/generated_core.pkl", "wb") as io:
#     pickle.dump(outputs, io, pickle.HIGHEST_PROTOCOL)

#%%

#%%
pallette = sns.color_palette("bright", 10)
sns.set_palette(pallette)
plt.figure(figsize=(5, 4))
nplots = 50
ix = np.random.choice(range(X.shape[0]), size=100)
for i in range(nplots):
    lab = "Counterfactuals" if i == 0 else None
    plt.plot(a_grid, 0.5 * (mu_nn_cf + mu_xgb_cf)[ix[i], :], c="gray", alpha=0.25, label=lab)
plt.scatter(A[ix[:nplots]], 0.5 * (mu_nn + mu_xgb)[ix[:nplots]], alpha=0.5, label="Synthetic observations")
plt.legend(loc="best")
plt.savefig("potential_outcomes.png", bbox_inches="tight")

#%%
ate = ((mu_xgb_cf_binary + mu_nn_cf_binary)[:, 1] - (mu_xgb_cf_binary + mu_nn_cf_binary)[:, 0]).mean()
print("ATE (ensemble binary):", ate)
print("GAIN (ensemble binary):", ate / (mu_xgb_cf_binary + mu_nn_cf_binary)[:, 0].mean())

#%%
