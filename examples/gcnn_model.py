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
    def __init__(
        self,
        dim,
        hidden_channels,
        output_channels,
        dropout=0.0,
        lr=0.01,
        weight_decay=1e-3,
    ):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim, hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.dropout2 = nn.Dropout(dropout)
        # self.conv3 = GCNConv(hidden_channels, output_channels)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x) # silu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        # x = F.silu(x)
        # x = self.dropout2(x)
        # x = self.conv3(x, edge_index)

        return x

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch.x, batch.edge_index)
        # pdb.set_trace()
        loss = F.mse_loss(y_hat, batch.y)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True
        )  # Logs loss to TensorBoard
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        # pdb.set_trace()
        loss = F.mse_loss(y_hat, batch.y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        return optimizer


def run_gcn(
    dataset, binary_treatment, dataset_num, dropout=0.0, lr=0.01, weight_decay=1e-3, auto_lr=False
):
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
    model = GCN(
        features.shape[1],
        hidden_channels=16,
        output_channels=1,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )
    callbacks = []
    if auto_lr:
        lr_finder = LearningRateFinder(min_lr=1e-4, max_lr=1.0)
        callbacks.append(lr_finder)
    trainer = pl.Trainer(
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=1,
        enable_progress_bar=False,
        callbacks=callbacks,
    )  # gpus=1 if torch.cuda.is_available() else 0)

    x = torch.FloatTensor(features)
    y = torch.FloatTensor(output)
    edge_index = torch.LongTensor(dataset.edges).T
    train_loader = DataLoader(
        [Data(x=x, y=y, edge_index=edge_index)],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainer.fit(model, train_loader)

    # get residuals
    model.eval()
    with torch.no_grad():
        preds = model(x, edge_index).cpu().numpy()
        residuals = y - preds

    # predict counterfactuals
    tvals = dataset.treatment_values
    counterfactuals = []

    for tval in tvals:
        trainmat = np.hstack([covariates, np.full_like(treatment, tval)])
        trainmat = feats_scaler.transform(trainmat)
        xcf = torch.FloatTensor(trainmat)
        with torch.no_grad():
            cfspred = model(xcf, edge_index) + residuals
            cfspred = model(xcf, edge_index) + residuals
            cfspred = cfspred.cpu().numpy()
            cfspred = output_scaler.inverse_transform(cfspred)[:, 0]
        counterfactuals.append(cfspred)  # cfspred_array[0])

    counterfactuals = np.stack(counterfactuals, axis=1)

    evaluator = DatasetEvaluator(dataset)

    if binary_treatment:
        ate = (counterfactuals[:, 1] - counterfactuals[:, 0]).mean()
        counterfactuals = np.squeeze(counterfactuals)
        err_eval = evaluator.eval(ate=ate, counterfactuals=counterfactuals)
    else:
        erf = counterfactuals.mean(0)
        counterfactuals = np.squeeze(counterfactuals)
        err_eval = evaluator.eval(erf=erf, counterfactuals=counterfactuals)

    # this is because json cannot serialize numpy arrays
    for key, value in err_eval.items():
        if isinstance(value, np.ndarray):
            err_eval[key] = value.tolist()

    res = {}
    res.update(**err_eval)
    res["smoothness"] = dataset.smoothness_of_missing
    res["confounding"] = dataset.confounding_of_missing
    res["dataset_num"] = dataset_num

    return res


import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--auto_lr", default=False, action="store_true")
    parser.add_argument("--logfile", type=str, default="results_GCN_relu_16h.jsonl")
    args = parser.parse_args()

    start = time.perf_counter()

    datamaster = DataMaster()
    datasets = datamaster.master

    filename = f"results/{args.logfile}"

    envs = datamaster.list_datasets()

    # Clean the file
    if args.overwrite:
        if os.path.exists(filename):
            os.remove(filename)

    for envname in envs:
        env = SpaceEnv(envname, dir="downloads")
        env_list = list(env.make_all())

        binary = True if "disc" in envname else False

        # remove from the list the datasets that have been already computed
        if os.path.exists(filename):
            with jsonlines.open(filename) as reader:
                results = list(reader)
        else:
            results = []

        # this overwriting code is buggy still!!!!!!! will fix later.
        # to_remove = []
        # for i, e in enumerate(env_list):
        #     for result in results:
        #         for i, d in enumerate(env_list):
        #         if result["envname"] == envname and result["dataset_num"] == i:
        #             to_remove.append(id(e))
        # if len(to_remove) > 0:
        #     env_list = [dataset for dataset in env_list if id(dataset) not in set(to_remove)]

        with concurrent.futures.ProcessPoolExecutor(args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_gcn,
                    dataset,
                    binary,
                    dataset_num=i,
                    dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    auto_lr=args.auto_lr,
                )
                for i, dataset in enumerate(env_list)
            }
            # As each future completes, write its result
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with jsonlines.open(filename, mode="a") as writer:
                    result["envname"] = envname
                    writer.write(result)

    finish = time.perf_counter()

    print(f"Finished in {round(finish-start, 2)} second(s)")