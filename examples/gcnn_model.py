import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import networkx as nx
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


import pytorch_lightning as pl

from spacebench import (
    SpaceEnv,
    SpaceDataset,
    DataMaster,
    DatasetEvaluator,
    EnvEvaluator,
)

envname = "healthd_dmgrcs_mortality_disc"
env = SpaceEnv(envname, dir="downloads")
env.__dict__.keys()
dataset = env.make()


class GCN(pl.LightningModule):
    def __init__(self, num_features, hidden_channels, output_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.silu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)

        return x
    
    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.mse_loss(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.mse_loss(out[batch.test_mask], batch.y[batch.test_mask])
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)


treatment = dataset.treatment[:, None]
covariates = dataset.covariates
outcome = dataset.outcome

# make train matrix
features = np.hstack([covariates, treatment])
adj_matrix = dataset.adjacency_matrix()

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)

# Create edge_index from adjacency matrix
edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()

# Create a list of Data objects
data_list = [Data(x=features, edge_index=edge_index)]

# Create a DataLoader
dataloader = DataLoader(data_list, batch_size=16, shuffle=True, num_workers=0)

model = GCN(features.shape[1], hidden_channels=16, output_channels=1) # Specify the number of output channels.

trainer = pl.Trainer()

trainer.fit(model, dataloader)

# Test the model


# trainer.test(dataloader)