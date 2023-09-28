import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import LearningRateFinder
import pytorch_lightning as pl
import numpy as np

from spacebench.env import SpaceDataset


class GCN(pl.LightningModule):
    """
    A Graph Convolutional Network (GCN), a
    type of neural network architecture for graph structured data.
    """

    def __init__(
        self,
        dim,
        hidden_channels,
        output_channels,
        dropout,
        lr,
        weight_decay,
    ):
        """Initialize Graph Convolutional Network class

        Attributes
        ----------
        dim : int
            The dimension of the input data, or the number of features of each node.
        hidden_channels : int
            The number of hidden channels in the GCN.
        dropout : float, optional
            The dropout rate for the network.
            By default, the dropout rate is 0.0, which means no dropout is applied.
        lr : float, optional
            The learning rate for the optimizer in the network.
            By default, the learning rate is 0.01.
        weight_decay : float, optional
            The L2 regularization coefficient.
            By default, weight decay is 1e-3.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim, hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)

        return x

    def training_step(self, batch):
        y_hat = self.forward(batch.x, batch.edge_index)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        return optimizer


class GCNModel:
    def __init__(
        self,
        dataset: SpaceDataset,
        binary_treatment: bool,
        hidden_channels=16,
        output_channels=1,
        dropout=0.0,
        lr=0.01,
        weight_decay=1e-3,
        auto_lr=False,
    ):
        """Initialize Graph Convolutional Network class

        Attributes
        ----------
        dataset: SpaceDataset
            An instance of SpaceDataset class
        binary_treatment : bool
            Flags if the treatment is binary or continous.
        hidden_channels : int
            The number of hidden channels in the GCN.
        output_channels : int
            The number of output channels.
        dropout : float, optional
            The dropout rate for the network, which is a regularization technique to prevent overfitting.
            By default, the dropout rate is 0.0, which means no dropout is applied.
        lr : float, optional
            The learning rate for the optimizer in the network.
            By default, the learning rate is 0.01.
        weight_decay : float, optional
            The L2 regularization coefficient.
            By default, weight decay is 1e-3.
        auto_lr : bool
            Flags if auto_lr should be used.
        """
        self._validate_dataset(dataset)
        self._data_prep(dataset)
        self.binary_treatment = binary_treatment

        self.model = self._initialize_model(
            dataset, hidden_channels, output_channels, dropout, lr, weight_decay
        )
        self.trainer = self._initialize_trainer(auto_lr)

    def _validate_dataset(self, dataset: SpaceDataset):
        if not isinstance(dataset, SpaceDataset):
            raise ValueError("dataset must be an instance" "of SpaceDataset")

    def _data_prep(self, dataset: SpaceDataset):
        """Prepares training and prediction data."""
        self.treatment = dataset.treatment[:, None]
        self.covariates = dataset.covariates
        self.outcome = dataset.outcome.reshape(-1, 1)
        self.features = np.hstack([self.covariates, self.treatment])
        self.tvals = dataset.treatment_values

        self.feats_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self.edge_index = torch.LongTensor(dataset.edges).T

    def _initialize_model(
        self,
        dataset: SpaceDataset,
        hidden_channels: int,
        output_channels: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ):
        return GCN(
            self.features.shape[1],
            hidden_channels=hidden_channels,
            output_channels=output_channels,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )

    def _initialize_trainer(self, auto_lr: bool):
        callbacks = [LearningRateFinder(min_lr=1e-4, max_lr=1.0)] if auto_lr else []

        return pl.Trainer(
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=1,
            enable_progress_bar=False,
            callbacks=callbacks,
        )

    def _create_counterfactuals(self, residuals):
        counterfactuals = []

        for tval in self.tvals:
            counterfactuals.append(self._calculate_counterfactual(tval, residuals))

        return np.stack(counterfactuals, axis=1)

    def _calculate_counterfactual(self, tval: float, residuals: np.array):
        trainmat = np.hstack([self.covariates, np.full_like(self.treatment, tval)])

        xcf = torch.FloatTensor(self.feats_scaler.transform(trainmat))

        with torch.no_grad():
            cfspred = self.model(xcf, self.edge_index) + residuals
            cfspred = cfspred.cpu().numpy()
            cfspred = self.output_scaler.inverse_transform(cfspred)[:, 0]

        return cfspred

    def gcn(self):
        """
        Implementation of the Graph Convolutional Network and estimation of the
        average treatment effect on the treated (ATT).

        Returns
        -------
            float: ate or the average treatment effect on the treated.
            float: erf.
            np.ndarray: The counterfactuals.
        """

        batch_size = self.features.shape[0]

        x = torch.FloatTensor(self.feats_scaler.fit_transform(self.features))
        y = torch.FloatTensor(self.output_scaler.fit_transform(self.outcome))

        train_loader = DataLoader(
            [Data(x=x, y=y, edge_index=self.edge_index)],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.trainer.fit(self.model, train_loader)

        with torch.no_grad():
            preds = self.model(x, self.edge_index).cpu().numpy()
            residuals = y - preds

        counterfactuals = self._create_counterfactuals(residuals)

        if self.binary_treatment:
            ate = (counterfactuals[:, 1] - counterfactuals[:, 0]).mean()
            counterfactuals = np.squeeze(counterfactuals)
            return ate, counterfactuals
        else:
            erf = counterfactuals.mean(0)
            counterfactuals = np.squeeze(counterfactuals)
            return erf, counterfactuals
