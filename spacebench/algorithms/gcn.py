import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from pytorch_lightning.callbacks import LearningRateFinder
from torch.optim import Adam
from torch_geometric.nn import GCNConv

from spacebench.algorithms import SpaceAlgo
from spacebench.algorithms.datautils import graph_data_loader
from spacebench.env import SpaceDataset
from spacebench.log import LOGGER


class GCN(SpaceAlgo):
    """
    Wrapper for Graph Convolutional Network (GCN) trainer.
    """
    supports_continuous = True
    supports_binary = True

    def __init__(
        self,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        act: str = "relu",
        epochs: int = 2000,
        auto_lr: bool = True,
        verbose: bool = True,
    ):
        """Initialize Graph Convolutional Network class

        Attributes
        ----------
        hidden_dim : int
            The number of hidden channels in the GCN.
        hidden_layers : int
            The number of hidden layers in the GCN.
        dropout : float
            The dropout rate for the network, which is a regularization technique to prevent overfitting.
            By default, the dropout rate is 0.0, which means no dropout is applied.
        lr : float
            The learning rate for the optimizer in the network.
            By default, the learning rate is 0.001.
        weight_decay : float
            The L2 regularization coefficient.
            By default, weight decay is 1e-3.
        act : str
            The activation function to use.
            By default, the activation function is relu. Must be a valid activation function
            from torch.nn.functional.
        epochs : int
            The number of epochs to train the network for.
            By default, the number of epochs is 1000.
        auto_lr : bool
            Use auto_lr to find the optimal learning rate.
            By default, auto_lr is True.
        verbose : bool
            Print model summary and training progress.
        """
        super().__init__()
        self.impl_kwargs = {
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "act": act,
        }
        self.epochs = epochs
        self.auto_lr = auto_lr
        self.verbose = verbose

    def fit(self, dataset: SpaceDataset):
        LOGGER.debug("Building GCN model...")
        input_dim = dataset.covariates.shape[1]
        self.model = _GCN_impl(input_dim, **self.impl_kwargs)

        LOGGER.debug("Preparing torch geometric data loader...")
        loader, self.feat_scaler, self.output_scaler = graph_data_loader(dataset)

        LOGGER.debug("Preparing trainer...")
        callbacks = (
            [LearningRateFinder(min_lr=1e-5, max_lr=1.0)] if self.auto_lr else []
        )
        self.trainer = pl.Trainer(
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=10.0,
            enable_progress_bar=self.verbose,
            callbacks=callbacks,
            max_epochs=self.epochs,
            enable_model_summary=self.verbose,
        )

        LOGGER.debug("Training GCN model...")
        self.trainer.fit(self.model, loader)
        LOGGER.debug("Finished training GCN model.")

    def eval(self, dataset: SpaceDataset):
        self.model.eval()
        LOGGER.debug("Preparing torch geometric data loader with existing scaler...")
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        preds = torch.cat(self.trainer.predict(self.model, loader))
        preds = preds.cpu().numpy()

        # scale back the preds
        preds = self.output_scaler.inverse_transform(preds)
        resid = dataset.outcome - preds[:, 0]

        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            loader, *_ = graph_data_loader(
                dataset, self.feat_scaler, self.output_scaler, a
            )
            preds_a = torch.cat(self.trainer.predict(self.model, loader))
            preds_a = preds_a.cpu().numpy()
            preds_a = self.output_scaler.inverse_transform(preds_a)
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        self.model.eval()
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        pred = torch.cat(self.trainer.predict(self.model, loader))
        pred = pred.cpu().numpy()
        pred = self.output_scaler.inverse_transform(pred)[:, 0]

        return np.mean((dataset.outcome - pred) ** 2)
    

class DrnetGCN(SpaceAlgo):
    """
    Wrapper for Graph Convolutional Network (GCN) trainer.
    """
    supports_continuous = True
    supports_binary = True

    def __init__(
        self,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        act: str = "relu",
        epochs: int = 2000,
        auto_lr: bool = True,
        verbose: bool = True,
    ):
        """Initialize Graph Convolutional Network class

        Attributes
        ----------
        hidden_dim : int
            The number of hidden channels in the GCN.
        hidden_layers : int
            The number of hidden layers in the GCN.
        dropout : float
            The dropout rate for the network, which is a regularization technique to prevent overfitting.
            By default, the dropout rate is 0.0, which means no dropout is applied.
        lr : float
            The learning rate for the optimizer in the network.
            By default, the learning rate is 0.001.
        weight_decay : float
            The L2 regularization coefficient.
            By default, weight decay is 1e-3.
        act : str
            The activation function to use.
            By default, the activation function is relu. Must be a valid activation function
            from torch.nn.functional.
        epochs : int
            The number of epochs to train the network for.
            By default, the number of epochs is 1000.
        auto_lr : bool
            Use auto_lr to find the optimal learning rate.
            By default, auto_lr is True.
        verbose : bool
            Print model summary and training progress.
        """
        super().__init__()
        self.impl_kwargs = {
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "act": act,
        }
        self.epochs = epochs
        self.auto_lr = auto_lr
        self.verbose = verbose

    def fit(self, dataset: SpaceDataset):
        LOGGER.debug("Building GCN model...")
        input_dim = dataset.covariates.shape[1]
        self.model = _GCN_impl(input_dim, **self.impl_kwargs)

        LOGGER.debug("Preparing torch geometric data loader...")
        loader, self.feat_scaler, self.output_scaler = graph_data_loader(dataset)

        LOGGER.debug("Preparing trainer...")
        callbacks = (
            [LearningRateFinder(min_lr=1e-5, max_lr=1.0)] if self.auto_lr else []
        )
        self.trainer = pl.Trainer(
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=10.0,
            enable_progress_bar=self.verbose,
            callbacks=callbacks,
            max_epochs=self.epochs,
            enable_model_summary=self.verbose,
        )

        LOGGER.debug("Training GCN model...")
        self.trainer.fit(self.model, loader)
        LOGGER.debug("Finished training GCN model.")
        self.model.eval()

    def eval(self, dataset: SpaceDataset):
        LOGGER.debug("Preparing torch geometric data loader with existing scaler...")
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        preds = torch.cat(self.trainer.predict(self.model, loader))
        preds = preds.cpu().numpy()

        # scale back the preds
        preds = self.output_scaler.inverse_transform(preds)
        resid = dataset.outcome - preds[:, 0]

        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            loader, *_ = graph_data_loader(
                dataset, self.feat_scaler, self.output_scaler, a
            )
            preds_a = torch.cat(self.trainer.predict(self.model, loader))
            preds_a = preds_a.cpu().numpy()
            preds_a = self.output_scaler.inverse_transform(preds_a)
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        self.model.eval()
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        pred = torch.cat(self.trainer.predict(self.model, loader))
        pred = pred.cpu().numpy()
        pred = self.output_scaler.inverse_transform(pred)[:, 0]

        return np.mean((dataset.outcome - pred) ** 2)
    

class DragonGCN(SpaceAlgo):
    """
    Wrapper for Graph Convolutional Network (GCN) trainer.
    """
    supports_continuous = True
    supports_binary = True

    def __init__(
        self,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        act: str = "relu",
        epochs: int = 2000,
        auto_lr: bool = True,
        verbose: bool = True,
    ):
        """Initialize Graph Convolutional Network class

        Attributes
        ----------
        hidden_dim : int
            The number of hidden channels in the GCN.
        hidden_layers : int
            The number of hidden layers in the GCN.
        dropout : float
            The dropout rate for the network, which is a regularization technique to prevent overfitting.
            By default, the dropout rate is 0.0, which means no dropout is applied.
        lr : float
            The learning rate for the optimizer in the network.
            By default, the learning rate is 0.001.
        weight_decay : float
            The L2 regularization coefficient.
            By default, weight decay is 1e-3.
        act : str
            The activation function to use.
            By default, the activation function is relu. Must be a valid activation function
            from torch.nn.functional.
        epochs : int
            The number of epochs to train the network for.
            By default, the number of epochs is 1000.
        auto_lr : bool
            Use auto_lr to find the optimal learning rate.
            By default, auto_lr is True.
        verbose : bool
            Print model summary and training progress.
        """
        super().__init__()
        self.impl_kwargs = {
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "act": act,
        }
        self.epochs = epochs
        self.auto_lr = auto_lr
        self.verbose = verbose

    def fit(self, dataset: SpaceDataset):
        LOGGER.debug("Building GCN model...")
        input_dim = dataset.covariates.shape[1]
        self.model = _GCN_impl(input_dim, **self.impl_kwargs)

        LOGGER.debug("Preparing torch geometric data loader...")
        loader, self.feat_scaler, self.output_scaler = graph_data_loader(dataset)

        LOGGER.debug("Preparing trainer...")
        callbacks = (
            [LearningRateFinder(min_lr=1e-5, max_lr=1.0)] if self.auto_lr else []
        )
        self.trainer = pl.Trainer(
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            gradient_clip_val=10.0,
            enable_progress_bar=self.verbose,
            callbacks=callbacks,
            max_epochs=self.epochs,
            enable_model_summary=self.verbose,
        )

        LOGGER.debug("Training GCN model...")
        self.trainer.fit(self.model, loader)
        LOGGER.debug("Finished training GCN model.")
        self.model.eval()

    def eval(self, dataset: SpaceDataset):
        LOGGER.debug("Preparing torch geometric data loader with existing scaler...")
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        preds = torch.cat(self.trainer.predict(self.model, loader))
        preds = preds.cpu().numpy()

        # scale back the preds
        preds = self.output_scaler.inverse_transform(preds)
        resid = dataset.outcome - preds[:, 0]

        LOGGER.debug("Computing counterfactuals...")
        ite = []
        for a in dataset.treatment_values:
            loader, *_ = graph_data_loader(
                dataset, self.feat_scaler, self.output_scaler, a
            )
            preds_a = torch.cat(self.trainer.predict(self.model, loader))
            preds_a = preds_a.cpu().numpy()
            preds_a = self.output_scaler.inverse_transform(preds_a)
            ite.append(preds_a)
        ite = np.concatenate(ite, axis=1)
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def tune_metric(self, dataset: SpaceDataset) -> float:
        self.model.eval()
        loader, *_ = graph_data_loader(dataset, self.feat_scaler, self.output_scaler)
        pred = torch.cat(self.trainer.predict(self.model, loader))
        pred = pred.cpu().numpy()
        pred = self.output_scaler.inverse_transform(pred)[:, 0]

        return np.mean((dataset.outcome - pred) ** 2)


class _GCN_impl(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
        act="relu",
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim + 1, hidden_dim)
        self.convh = nn.ModuleList(
            [GCNConv(hidden_dim + 1, hidden_dim) for _ in range(hidden_layers - 1)]
        )
        self.convf = GCNConv(hidden_dim + 1, output_dim)
        self.act = getattr(F, act)
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, batch: torch_geometric.data.Data):
        x = batch.x
        edge_index = batch.edge_index
        treatment = x[:, 0]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convh:
            x = torch.cat([treatment[:, None], x], dim=1)
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat([treatment[:, None], x], dim=1)
        x = self.convf(x, edge_index)

        return x

    def training_step(self, batch):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import sys

    import spacebench

    env_name = spacebench.DataMaster().list_envs()[0]
    env = spacebench.SpaceEnv(env_name)
    dataset = env.make()

    algo = GCN()
    algo.fit(dataset)
    effects = algo.eval(dataset)

    sys.exit(0)
