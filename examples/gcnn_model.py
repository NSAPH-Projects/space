import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import concurrent.futures
import jsonlines
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import LearningRateFinder

import pytorch_lightning as pl

from spacebench import (
    SpaceEnv,
    DataMaster,
    DatasetEvaluator,
)

class GCN(pl.LightningModule):
    def __init__(self, dim, edge_index, hidden_channels, output_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        self.edge_index = edge_index

    def forward(self, data):
        x, edge_index = data.x, self.edge_index
        x = self.conv1(x, edge_index)
        x = F.silu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = self.conv2(x, edge_index)
        # x = F.silu(x)
        # x = self.dropout2(x)
        # x = self.conv3(x, edge_index)

        return x

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        #pdb.set_trace()
        loss = F.mse_loss(y_hat, batch.y)
        self.log('train_loss', loss)  # Logs loss to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        #pdb.set_trace()
        loss = F.mse_loss(y_hat, batch.y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=0.005, lr=0.001)
        return optimizer


def run_gcn(dataset):
    # make train matrix
    treatment = dataset.treatment[:, None]
    covariates = dataset.covariates
    outcome = dataset.outcome.reshape(-1, 1)
    features = np.hstack([covariates, treatment])

    # Standardize input and output data
    feats_scaler = StandardScaler()
    output_scaler = StandardScaler()
    features = feats_scaler.fit_transform(features)
    output = output_scaler.fit_transform(outcome)

    batch_size = features.shape[0]

    # Initialize the model and trainer
    model = GCN(features.shape[1], torch.LongTensor(dataset.edges).T, hidden_channels=32, output_channels=1)
    trainer = pl.Trainer(accelerator="cpu", enable_checkpointing=False, logger=False) #gpus=1 if torch.cuda.is_available() else 0)
    
    train_loader = DataLoader([Data(x=torch.tensor(
        features,dtype=torch.float), 
        y=torch.tensor(output, dtype=torch.float), 
        edge_index=torch.LongTensor(dataset.edges).T)], batch_size=batch_size, shuffle=False, num_workers=0)
    trainer.fit(model, train_loader)
    
    # predict counterfactuals
    tvals = dataset.treatment_values
    counterfactuals = []

    for tval in tvals:
        trainmat = np.hstack([covariates, np.full_like(treatment, tval)])
        trainmat = feats_scaler.transform(trainmat)

        cfs_loader = DataLoader([Data(
            x=torch.tensor(trainmat,dtype=torch.float),
            edge_index=torch.LongTensor(dataset.edges).T)], batch_size=batch_size, shuffle=False, num_workers=0)

        pred = trainer.predict(model, cfs_loader)
        counterfactuals.append(pred[0])
        
    counterfactuals = np.stack(counterfactuals, axis=1)
    for i in range(counterfactuals.shape[1]):
        counterfactuals[:, i] = output_scaler.inverse_transform(counterfactuals[:, i])

    evaluator = DatasetEvaluator(dataset)
    erf =counterfactuals.mean(0)
    err_spatial_eval = evaluator.eval(
            erf=erf, counterfactuals=np.squeeze(counterfactuals))

    res = {}
    for key, value in err_spatial_eval.items():
        if isinstance(value, np.ndarray):
            res[key] = value.tolist()
    res["smoothness"] = dataset.smoothness_of_missing
    res["confounding"] = dataset.confounding_of_missing

    return res 


if __name__ == '__main__':
    start = time.perf_counter()

    datamaster = DataMaster()
    datasets = datamaster.master 

    filename = 'results_gcn.csv'

    envs = datasets.index.values
    envs = envs[:1] # REMOVE [:1] FOR THE FULL RUN

    # Clean the file
    with open(filename, 'w') as csvfile:
        pass

    for envname in envs:
        env = SpaceEnv(envname, dir="downloads")
        dataset_list = list(env.make_all())
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(
                run_gcn, dataset) for dataset in 
                dataset_list
                }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode='a') as writer:
                    result["envname"] = envname
                    writer.write(result)

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start, 2)} second(s)')
