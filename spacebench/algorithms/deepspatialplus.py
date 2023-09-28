import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

from spacebench.algorithms import SpaceAlgo
from spacebench.env import SpaceDataset
from spacebench.log import LOGGER
from spacebench.algorithms.datautils import spatial_train_test_split
from spacebench.algorithms.spatialplus import compute_phi


def tps_deep_pred(
    nn: nn.Module,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    Phi = compute_phi(coords, cp_coords)
    A = torch.cat([coords, Phi], dim=1)
    return nn(covars)[:, 0] + A @ params


def tps_deep_loss(
    net: nn.Module,
    y: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
    lam: float,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    k = cp_coords.shape[0]

    pred = tps_deep_pred(net, coords, cp_coords, covars, params)
    if not binary_loss:
        pred_loss = F.mse_loss(pred, y, reduction="none")
    else:
        pred_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")

    pred_loss = (pred_loss * mask).mean() if mask is not None else pred_loss.mean()

    c = compute_phi(coords, cp_coords) @ params[-k:, None]  # vector of size k x 1
    c = params[-k:, None]
    reg_loss = c.pow(2).sum()

    return pred_loss + lam * reg_loss


def tps_deep_opt(
    net: nn.Module,
    y: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    lam: float,
    lr: float = 0.003,
    max_iter: int = 20_000,
    atol: float = 1e-4,
    plateau_patience: int = 10,
    verbose: bool = True,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
    weight_decay: float = 1e-3,
    scheduler: bool = True,
) -> torch.Tensor:
    """
    Optimize the parameters for Thin Plate Spline regression.
    It is recommended that the outcome and covariates are
    all centered and scaled to have unit variance.

    Arguments
    ----------
        y (torch.Tensor): Target values of length n.
        coords (torch.Tensor): Coordinate matrix of size (n, d).
        cp_idx (list[int]): Indexes of control points of length k.
        covariates (torch.Tensor): Covariate matrix of size (n, p).
        lam (float): Regularization parameter.
        lr (float, optional): Learning rate. Defaults to 0.003.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10_000.
        atol (float, optional): Tolerance for max abs gradient. Defaults to 1e-6.
        plateau_patience (int, optional): Patience for learning rate scheduler.
            Defaults to 10.
        verbose (bool, optional): Whether to scheduler messages. Defaults to True.
        binary_loss (bool, optional): Whether to use binary cross entropy loss.
            Defaults to False.
        mask (torch.Tensor, optional): Mask of length n to use for loss. Defaults to None.
        weight_decay (float, optional): Weight decay for Adam optimizer. Defaults to 1e-3.

    Returns:
        torch.Tensor: Optimized parameters of length 1 + d + p + k
        the first element is the intercept, the next d elements are the
        coefficients of the coordinates, the next p elements are the
        coefficients of the covariates, and the last k elements are the
        coefficients of the radial basis.
    """
    k = cp_coords.shape[0]
    d = coords.shape[1]

    # Initialization of params
    params = torch.zeros(d + k, requires_grad=True)
    opt = torch.optim.Adam(
        list(net.parameters()) + [params], lr=lr, weight_decay=weight_decay
    )
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.9,
            patience=plateau_patience,
            verbose=verbose,
            min_lr=1e-4,
        )

    # Run the optimizer
    it = 0

    while it < max_iter:
        opt.zero_grad()
        loss = tps_deep_loss(
            net, y, coords, cp_coords, covars, params, lam, binary_loss, mask=mask
        )
        loss.backward()
        opt.step()
        if scheduler:
            sched.step(loss)

    #     # Check for convergence
    #     max_grad = params.grad.abs().max()
    #     if max_grad < atol:
    #         LOGGER.info(f"TPS converged after {it} iterations.")
    #         break

        it += 1

    # if it == max_iter - 1:
    #     LOGGER.warning(f"TPS did not converge after {max_iter} iterations.")

    return params.detach()


def tps_deep_pred_dragon(
    nn: nn.Module,
    t: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    Phi = compute_phi(coords, cp_coords)
    A = torch.cat([coords, Phi], dim=1)
    out = nn(covars)
    t_eff = torch.gather(out[:, :2], 1, t.long()).squeeze(1)
    return t_eff + A @ params, out


def tps_deep_loss_dragon(
    net: nn.Module,
    y: torch.Tensor,
    t: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
    lam: float,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    k = cp_coords.shape[0]

    pred, nn_out = tps_deep_pred_dragon(
        net, t.unsqueeze(1), coords, cp_coords, covars, params
    )
    if not binary_loss:
        pred_loss = F.mse_loss(pred, y, reduction="none")
    else:
        pred_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")

    logits = nn_out[:, -1]
    treatment_loss = F.binary_cross_entropy_with_logits(
        logits.clamp(-10, 10), t.float().clamp(0.01, 0.99), reduction="none"
    )

    pred_loss = (pred_loss * mask).mean() if mask is not None else pred_loss.mean()
    treatment_loss = (
        (treatment_loss * mask).mean() if mask is not None else treatment_loss.mean()
    )

    c = compute_phi(coords, cp_coords) @ params[-k:, None]  # vector of size k x 1
    c = params[-k:, None]
    reg_loss = c.pow(2).sum()

    return pred_loss + lam * reg_loss + 0.1 * treatment_loss


def tps_deep_opt_dragon(
    net: nn.Module,
    y: torch.Tensor,
    t: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    lam: float,
    lr: float = 0.003,
    max_iter: int = 20_000,
    atol: float = 1e-4,
    plateau_patience: int = 10,
    verbose: bool = True,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
    weight_decay: float = 1e-3,
    scheduler: bool = True,
) -> torch.Tensor:
    """
    Optimize the parameters for Thin Plate Spline regression.
    It is recommended that the outcome and covariates are
    all centered and scaled to have unit variance.

    Arguments
    ----------
        y (torch.Tensor): Target values of length n.
        coords (torch.Tensor): Coordinate matrix of size (n, d).
        cp_idx (list[int]): Indexes of control points of length k.
        covariates (torch.Tensor): Covariate matrix of size (n, p).
        lam (float): Regularization parameter.
        lr (float, optional): Learning rate. Defaults to 0.003.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10_000.
        atol (float, optional): Tolerance for max abs gradient. Defaults to 1e-6.
        plateau_patience (int, optional): Patience for learning rate scheduler.
            Defaults to 10.
        verbose (bool, optional): Whether to scheduler messages. Defaults to True.
        binary_loss (bool, optional): Whether to use binary cross entropy loss.
            Defaults to False.
        mask (torch.Tensor, optional): Mask of length n to use for loss. Defaults to None.
        weight_decay (float, optional): Weight decay for Adam optimizer. Defaults to 1e-3.

    Returns:
        torch.Tensor: Optimized parameters of length 1 + d + p + k
        the first element is the intercept, the next d elements are the
        coefficients of the coordinates, the next p elements are the
        coefficients of the covariates, and the last k elements are the
        coefficients of the radial basis.
    """
    k = cp_coords.shape[0]
    d = coords.shape[1]

    # Initialization of params
    params = torch.zeros(d + k, requires_grad=True)
    opt = torch.optim.Adam(
        list(net.parameters()) + [params], lr=lr, weight_decay=weight_decay
    )
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.9,
            patience=plateau_patience,
            verbose=verbose,
            min_lr=1e-4,
        )

    # Run the optimizer
    it = 0

    while it < max_iter:
        opt.zero_grad()
        loss = tps_deep_loss_dragon(
            net, y, t, coords, cp_coords, covars, params, lam, binary_loss, mask=mask
        )
        loss.backward()
        opt.step()
        if scheduler:
            sched.step(loss)

        # # Check for convergence
        # max_grad = params.grad.abs().max()
        # if max_grad < atol:
        #     LOGGER.info(f"TPS converged after {it} iterations.")
        #     break

        it += 1

    # if it == max_iter - 1:
    #     LOGGER.warning(f"TPS did not converge after {max_iter} iterations.")

    return params.detach()


def tps_deep_pred_drnet(
    nn: nn.Module,
    t: torch.Tensor,
    cutpoints: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    Phi = compute_phi(coords, cp_coords)
    A = torch.cat([coords, Phi], dim=1)
    wts = nn(covars)
    basis = F.relu(t - cutpoints)  # n x cutpoints
    t_eff = wts[:, 0] + wts[:, 1] * t[:, 0] + (basis * wts[:, 2:]).sum(-1)
    return t_eff + A @ params


def tps_deep_loss_drnet(
    net: nn.Module,
    y: torch.Tensor,
    t: torch.Tensor,
    cutpoints: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
    lam: float,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    k = cp_coords.shape[0]

    pred = tps_deep_pred_drnet(
        net, t.unsqueeze(1), cutpoints, coords, cp_coords, covars, params
    )
    if not binary_loss:
        pred_loss = F.mse_loss(pred, y, reduction="none")
    else:
        pred_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")

    pred_loss = (pred_loss * mask).mean() if mask is not None else pred_loss.mean()

    # c = compute_phi(coords, cp_coords) @ params[-k:, None]  # vector of size k x 1
    c = params[-k:, None]
    reg_loss = c.pow(2).sum()

    return pred_loss + lam * reg_loss


def tps_deep_opt_drnet(
    net: nn.Module,
    y: torch.Tensor,
    t: torch.Tensor,
    cutpoints: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    lam: float,
    lr: float = 0.003,
    max_iter: int = 20_000,
    atol: float = 1e-4,
    plateau_patience: int = 10,
    verbose: bool = True,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
    weight_decay: float = 1e-3,
    scheduler: bool = True,
) -> torch.Tensor:
    k = cp_coords.shape[0]
    d = coords.shape[1]

    # Initialization of params
    params = torch.zeros(d + k, requires_grad=True)
    opt = torch.optim.Adam(
        list(net.parameters()) + [params], lr=lr, weight_decay=weight_decay
    )
    if scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.9,
            patience=plateau_patience,
            verbose=verbose,
            min_lr=1e-4,
        )

    # Run the optimizer
    it = 0

    while it < max_iter:
        opt.zero_grad()
        loss = tps_deep_loss_drnet(
            net,
            y,
            t,
            cutpoints,
            coords,
            cp_coords,
            covars,
            params,
            lam,
            binary_loss,
            mask=mask,
        )
        loss.backward()
        opt.step()
        if scheduler:
            sched.step(loss)

        # Check for convergence
        # max_grad = params.grad.abs().max()
        # if max_grad < atol:
        #     LOGGER.info(f"TPS converged after {it} iterations.")
        #     break

        it += 1

    # if it == max_iter - 1:
    #     LOGGER.warning(f"TPS did not converge after {max_iter} iterations.")

    return params.detach()


class MLPSpatial(SpaceAlgo):
    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
    ):
        """Implementation of the Spatial Method using Thin Plate Spline regression.

        Arguments
        ----------
            k (int): max number of control points. Defaults to 100.
            max_iter (int): max number of iterations for optimization. Defaults to 20_000.
            lam (float): regularization parameter. Defaults to 0.001.
            spatial_split_kwargs (dict, optional): args for spatial_train_test_split.
                Defaults to None. When not none, it creates a training and test mask
                to use for minimizing the loss and the tune metric, respectively. Must specify
                init_frac and levels.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.lam = lam
        self.spatial_split_kwargs = spatial_split_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def fit(self, dataset: SpaceDataset):
        din = 1 + dataset.covariates.shape[1]
        if self.hidden_layers == 0:
            layers = [nn.Linear(din, 1)]
        else:
            layers = [nn.Linear(din, self.hidden_dim)]
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, 1))
        self.nn = nn.Sequential(*layers)

        if self.spatial_split_kwargs is not None:
            train_ix, _, _ = spatial_train_test_split(
                nx.from_edgelist(dataset.edges), **self.spatial_split_kwargs
            )
            self.mask = torch.zeros(dataset.size())
            self.mask[train_ix] = torch.tensor(1.0)
        else:
            self.mask = None

        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = torch.cat([t[:, None], covars], dim=1)

        # sample control points
        k = min(self.k, dataset.size())
        self.cp_idx = torch.LongTensor(
            np.random.choice(dataset.size(), k, replace=False)
        )
        self.cp_coords = coords[self.cp_idx]

        # standardize
        self.coords_mu, self.coords_std = coords.mean(0), coords.std(0)
        self.inputs_mu, self.inputs_std = inputs.mean(0), inputs.std(0)
        self.y_mu, self.y_std = y.mean(), y.std()
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std

        # fit
        self.params = tps_deep_opt(
            self.nn,
            y,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # 0 coef is intercept, 1,2 are for coords, 3 is treatment
        self.t_coef = (self.params[3] * self.y_std / self.inputs_std[0]).item()
        self.nn.eval()

    def eval(self, dataset: SpaceDataset):
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = torch.cat([t[:, None], covars], dim=1)
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std

        ite = []
        for a in dataset.treatment_values:
            inputs_a = torch.cat([torch.full((dataset.size(), 1), a), covars], dim=1)
            inputs_a = (inputs_a - self.inputs_mu) / self.inputs_std
            with torch.no_grad():
                ite.append(
                    tps_deep_pred(
                        self.nn, coords, self.cp_coords, inputs_a, self.params
                    )
                )
        ite = torch.stack(ite, dim=1)
        ite = self.y_std * ite + self.y_mu
        ite = ite.numpy()

        # add residuals
        with torch.no_grad():
            pred = tps_deep_pred(self.nn, coords, self.cp_coords, inputs, self.params)
        pred = self.y_std * pred + self.y_mu
        resid = dataset.outcome - pred.numpy()
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def tune_metric(self, dataset: SpaceDataset) -> float:
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = torch.cat([t[:, None], covars], dim=1)
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        pred = tps_deep_pred(self.nn, coords, self.cp_coords, inputs, self.params)
        loss = F.mse_loss(pred, y, reduction="none")

        if self.mask is not None:
            loss = (loss * (1.0 - self.mask)).mean()
        else:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            loss = loss.mean()

        return loss.item()


class MLPSpatialPlus(SpaceAlgo):
    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
    ):
        """Implementation of the SpatialPlus Method using Thin Plate Spline regression.

        Arguments
        ----------
            k (int): max number of control points. Defaults to 100.
            max_iter (int): max number of iterations for optimization. Defaults to 20_000.
            lam_t (float): regularization parameter for treatment model.
            lam_y (float): regularization parameter for outcome model.
            spatial_split_kwargs (dict, optional): args for spatial_train_test_split.
                Defaults to None. When not none, it creates a training and test mask
                to use for minimizing the loss and the tune metric, respectively. Must specify
                init_frac and levels.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.lam_t = lam_t
        self.lam_y = lam_y
        self.spatial_split_kwargs = spatial_split_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def fit(self, dataset: SpaceDataset):
        if self.hidden_layers == 0:
            layers = [nn.Linear(dataset.covariates.shape[1], 1)]
        else:
            layers = [nn.Linear(dataset.covariates.shape[1], self.hidden_dim)]
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, 1))
        self.nn_t = nn.Sequential(*layers)

        if self.hidden_layers == 0:
            layers = [nn.Linear(1 + dataset.covariates.shape[1], 1)]
        else:
            layers = [nn.Linear(1 + dataset.covariates.shape[1], self.hidden_dim)]
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, 1))
        self.nn_y = nn.Sequential(*layers)

        if self.spatial_split_kwargs is not None:
            train_ix, _, _ = spatial_train_test_split(
                nx.from_edgelist(dataset.edges), **self.spatial_split_kwargs
            )
            self.mask = torch.zeros(dataset.size())
            self.mask[train_ix] = torch.tensor(1.0)
        else:
            self.mask = None

        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        t = torch.FloatTensor(dataset.treatment)
        self.t_mu, self.t_std = t.mean(), t.std()

        # standardize t if not binary:
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        # sample control points
        k = min(self.k, dataset.size())
        self.cp_idx = torch.LongTensor(
            sorted(np.random.choice(dataset.size(), k, replace=False))
        )
        self.cp_coords = coords[self.cp_idx]

        # standardize
        self.coords_mu, self.coords_std = coords.mean(0), coords.std(0)
        self.covars_mu, self.covars_std = covars.mean(0), covars.std(0)
        coords = (coords - self.coords_mu) / self.coords_std
        covars = (covars - self.covars_mu) / self.covars_std

        # fit a model for the treatment
        self.t_params = tps_deep_opt(
            self.nn_t,
            t,
            coords,
            self.cp_coords,
            covars,
            lam=self.lam_t,
            max_iter=self.max_iter,
            verbose=False,
            binary_loss=dataset.has_binary_treatment(),
            mask=self.mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # predict
        y = torch.FloatTensor(dataset.outcome)
        t_pred = tps_deep_pred(self.nn_t, coords, self.cp_coords, covars, self.t_params)
        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)
        with torch.no_grad():
            t_resid = t - t_pred

        # fit a model for the outcome
        self.y_mu, self.y_std = y.mean(), y.std()
        y = (y - self.y_mu) / self.y_std
        inputs = torch.cat(
            [t_resid[:, None], torch.FloatTensor(dataset.covariates)], dim=1
        )
        self.inputs_mu, self.inputs_std = inputs.mean(0), inputs.std(0)
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        self.y_params = tps_deep_opt(
            self.nn_y,
            y,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam_y,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # 0 coef is intercept, 1,2 are for coords, 3 is treatment
        self.t_coef = (self.y_params[3] * self.y_std / self.t_mu).item()
        self.nn_t.eval()
        self.nn_y.eval()

    def eval(self, dataset: SpaceDataset):
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = torch.cat([t[:, None], covars], dim=1)
        covars = (covars - self.covars_mu) / self.covars_std
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std

        # standardize t if not binary
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        # obtain t_resid
        with torch.no_grad():
            t_pred = tps_deep_pred(
                self.nn_t, coords, self.cp_coords, covars, self.t_params
            )

        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)

        ite = []
        for a in dataset.treatment_values:
            t_resid = torch.full((dataset.size(), 1), a) - t_pred[:, None]
            inputs_a = torch.cat([t_resid, covars], dim=1)
            inputs_a = (inputs_a - self.inputs_mu) / self.inputs_std
            with torch.no_grad():
                ite.append(
                    tps_deep_pred(
                        self.nn_y, coords, self.cp_coords, inputs_a, self.y_params
                    )
                )
        ite = torch.stack(ite, dim=1)
        ite = self.y_std * ite + self.y_mu
        ite = ite.numpy()

        # add residuals
        with torch.no_grad():
            pred = tps_deep_pred(
                self.nn_y, coords, self.cp_coords, inputs, self.y_params
            )
            pred = self.y_std * pred + self.y_mu
        resid = dataset.outcome - pred.numpy()
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def tune_metric(self, dataset: SpaceDataset) -> float:
        if self.mask is None:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            tune_mask = None
        else:
            tune_mask = 1.0 - self.mask

        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        covars = (covars - self.covars_mu) / self.covars_std
        coords = (coords - self.coords_mu) / self.coords_std
        t = (t - self.t_mu) / self.t_std
        t_loss = tps_deep_loss(
            self.nn_t,
            t,
            coords,
            self.cp_coords,
            covars,
            self.t_params,
            lam=0.0,
            binary_loss=dataset.has_binary_treatment(),
            mask=tune_mask,
        )
        t_pred = tps_deep_pred(self.nn_t, coords, self.cp_coords, covars, self.t_params)
        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)
        t_resid = t - t_pred

        inputs = torch.cat(
            [t_resid[:, None], torch.FloatTensor(dataset.covariates)], dim=1
        )
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        y_loss = tps_deep_loss(
            self.nn_y,
            y,
            coords,
            self.cp_coords,
            inputs,
            self.y_params,
            lam=0.0,
            mask=tune_mask,
        )

        return t_loss.item() + y_loss.item()


class DragonSpatial(SpaceAlgo):
    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
    ):
        """Implementation of the Spatial Method using Thin Plate Spline regression.

        Arguments
        ----------
            k (int): max number of control points. Defaults to 100.
            max_iter (int): max number of iterations for optimization. Defaults to 20_000.
            lam (float): regularization parameter. Defaults to 0.001.
            spatial_split_kwargs (dict, optional): args for spatial_train_test_split.
                Defaults to None. When not none, it creates a training and test mask
                to use for minimizing the loss and the tune metric, respectively. Must specify
                init_frac and levels.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.lam = lam
        self.spatial_split_kwargs = spatial_split_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def fit(self, dataset: SpaceDataset):
        din = dataset.covariates.shape[1]
        if self.hidden_layers == 0:
            layers = [nn.Linear(din, 1)]
        else:
            layers = [nn.Linear(din, self.hidden_dim)]
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, 3))
        self.nn = nn.Sequential(*layers)

        if self.spatial_split_kwargs is not None:
            train_ix, _, _ = spatial_train_test_split(
                nx.from_edgelist(dataset.edges), **self.spatial_split_kwargs
            )
            self.mask = torch.zeros(dataset.size())
            self.mask[train_ix] = torch.tensor(1.0)
        else:
            self.mask = None

        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        # inputs = torch.cat([t[:, None], covars], dim=1)
        inputs = covars

        # sample control points
        k = min(self.k, dataset.size())
        self.cp_idx = torch.LongTensor(
            np.random.choice(dataset.size(), k, replace=False)
        )
        self.cp_coords = coords[self.cp_idx]

        # standardize
        self.coords_mu, self.coords_std = coords.mean(0), coords.std(0)
        self.inputs_mu, self.inputs_std = inputs.mean(0), inputs.std(0)
        self.y_mu, self.y_std = y.mean(), y.std()
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std

        # fit
        self.params = tps_deep_opt_dragon(
            self.nn,
            y,
            t,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # don't use dropout at eval time
        self.nn.eval()


    def eval(self, dataset: SpaceDataset):
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        inputs = covars
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std

        ite = []
        for a in dataset.treatment_values:
            t_a = torch.full((dataset.size(), 1), a)
            with torch.no_grad():
                ite.append(
                    tps_deep_pred_dragon(
                        self.nn, t_a, coords, self.cp_coords, inputs, self.params
                    )[0]
                )
        ite = torch.stack(ite, dim=1)
        ite = self.y_std * ite + self.y_mu
        ite = ite.numpy()

        # add residuals
        with torch.no_grad():
            t = torch.FloatTensor(dataset.treatment).unsqueeze(1)
            pred = tps_deep_pred_dragon(
                self.nn, t, coords, self.cp_coords, inputs, self.params
            )[0]
        pred = self.y_std * pred + self.y_mu
        resid = dataset.outcome - pred.numpy()
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def tune_metric(self, dataset: SpaceDataset) -> float:
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = covars
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        pred = tps_deep_pred_dragon(
            self.nn, t.unsqueeze(1), coords, self.cp_coords, inputs, self.params
        )[0]
        loss = F.mse_loss(pred, y, reduction="none")

        if self.mask is not None:
            loss = (loss * (1.0 - self.mask)).mean()
        else:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            loss = loss.mean()

        return loss.item()


class DrnetSpatial(SpaceAlgo):
    supports_binary = False
    supports_continuous = True

    def __init__(
        self,
        n_cutpoints: int = 5,
        k: int = 100,
        max_iter: int = 20_000,
        lam: float = 0.001,
        spatial_split_kwargs: dict | None = None,
        hidden_dim: int = 16,
        hidden_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 1e-3,
    ):
        """Implementation of the Spatial Method using Thin Plate Spline regression.

        Arguments
        ----------
            k (int): max number of control points. Defaults to 100.
            max_iter (int): max number of iterations for optimization. Defaults to 20_000.
            lam (float): regularization parameter. Defaults to 0.001.
            spatial_split_kwargs (dict, optional): args for spatial_train_test_split.
                Defaults to None. When not none, it creates a training and test mask
                to use for minimizing the loss and the tune metric, respectively. Must specify
                init_frac and levels.
        """
        super().__init__()
        self.k = k
        self.max_iter = max_iter
        self.lam = lam
        self.spatial_split_kwargs = spatial_split_kwargs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.n_cutpoints = n_cutpoints

    def fit(self, dataset: SpaceDataset):
        din = dataset.covariates.shape[1]
        if self.hidden_layers == 0:
            layers = [nn.Linear(din, self.n_cutpoints + 2)]
        else:
            layers = [nn.Linear(din, self.hidden_dim)]
            for _ in range(self.hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, self.n_cutpoints + 2))
        self.nn = nn.Sequential(*layers)

        if self.spatial_split_kwargs is not None:
            train_ix, _, _ = spatial_train_test_split(
                nx.from_edgelist(dataset.edges), **self.spatial_split_kwargs
            )
            self.mask = torch.zeros(dataset.size())
            self.mask[train_ix] = torch.tensor(1.0)
        else:
            self.mask = None

        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)

        # compute cutpoints from t
        self.cutpoints = torch.linspace(-1.5, 1.5, self.n_cutpoints)

        # inputs = torch.cat([t[:, None], covars], dim=1)
        inputs = covars

        # sample control points
        k = min(self.k, dataset.size())
        self.cp_idx = torch.LongTensor(
            np.random.choice(dataset.size(), k, replace=False)
        )
        self.cp_coords = coords[self.cp_idx]

        # standardize
        self.coords_mu, self.coords_std = coords.mean(0), coords.std(0)
        self.inputs_mu, self.inputs_std = inputs.mean(0), inputs.std(0)
        self.y_mu, self.y_std = y.mean(), y.std()
        self.t_mu, self.t_std = t.mean(), t.std()
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        t = (t - self.t_mu) / self.t_std

        # fit
        self.params = tps_deep_opt_drnet(
            self.nn,
            y,
            t,
            self.cutpoints,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # 0 coef is intercept, 1,2 are for coords, 3 is treatment
        self.nn.eval()

    def eval(self, dataset: SpaceDataset):
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        t = torch.FloatTensor(dataset.treatment).unsqueeze(1)
        inputs = covars
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        t = (t - self.t_mu) / self.t_std

        ite = []
        for a in dataset.treatment_values:
            t_a = torch.full((dataset.size(), 1), a)
            t_a = (t_a - self.t_mu) / self.t_std
            with torch.no_grad():
                ite.append(
                    tps_deep_pred_drnet(
                        self.nn,
                        t_a,
                        self.cutpoints,
                        coords,
                        self.cp_coords,
                        inputs,
                        self.params,
                    )
                )
        ite = torch.stack(ite, dim=1)
        ite = self.y_std * ite + self.y_mu
        ite = ite.numpy()

        # add residuals
        with torch.no_grad():
            pred = tps_deep_pred_drnet(
                self.nn, t, self.cutpoints, coords, self.cp_coords, inputs, self.params
            )
        pred = self.y_std * pred + self.y_mu
        resid = dataset.outcome - pred.numpy()
        ite += resid[:, None]

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def tune_metric(self, dataset: SpaceDataset) -> float:
        coords = torch.FloatTensor(dataset.coordinates)
        covars = torch.FloatTensor(dataset.covariates)
        y = torch.FloatTensor(dataset.outcome)
        t = torch.FloatTensor(dataset.treatment)
        inputs = covars
        coords = (coords - self.coords_mu) / self.coords_std
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        self.t = (t - self.t_mu) / self.t_std
        pred = tps_deep_pred_drnet(
            self.nn, t.unsqueeze(1), self.n_cutpoints, coords, self.cp_coords, inputs, self.params
        )
        loss = F.mse_loss(pred, y, reduction="none")

        if self.mask is not None:
            loss = (loss * (1.0 - self.mask)).mean()
        else:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            loss = loss.mean()

        return loss.item()


if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt
    import numpy as np

    import spacebench
    from spacebench.algorithms.datautils import spatial_train_test_split

    # # test 1d tps solve
    # n = 100
    # coords = torch.FloatTensor(sorted(np.random.rand(n) * 2 * np.pi)).unsqueeze(1)
    # # coords = torch.linspace(0, 2 * torch.pi, n).unsqueeze(1)
    # covars = torch.randn(n, 1)
    # # cp_idx = torch.LongTensor(np.random.choice(n, 100, replace=False))
    # cp_idx = torch.arange(n)
    # y = torch.sin(2 * coords.squeeze()) + 0.1 * torch.randn(n)

    # # standardize
    # coords = (coords - coords.mean()) / coords.std()
    # covars = (covars - covars.mean()) / covars.std()
    # y = (y - y.mean()) / y.std()

    # params = tps_opt(y, coords, cp_idx, covars, lam=0.03)
    # pred = tps_pred(coords, cp_idx, covars, params)

    # fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # ax.scatter(coords[:, 0], y, label="Y", c="blue", alpha=0.2, s=10)
    # ax.plot(coords[:, 0], pred, label="pred", c="red")
    # ax.legend()
    # plt.show()
    # plt.close()

    # plt.bar(np.arange(len(params)), params)
    # plt.show()
    # plt.close()
    spatial_split_kwargs = {"init_frac": 0.02, "levels": 1, "seed": 1}

    env = spacebench.SpaceEnv("healthd_dmgrcs_mortality_disc")
    dataset = env.make()
    evaluator = spacebench.DatasetEvaluator(dataset)

    # Run Spatial
    algo = MLPSpatial(max_iter=1000, spatial_split_kwargs=spatial_split_kwargs)
    algo.fit(dataset)
    effects1 = algo.eval(dataset)
    tune_metric1 = algo.tune_metric(dataset)

    # # Run SpatialPlus
    algo = MLPSpatialPlus(
        max_iter=500,
        spatial_split_kwargs=spatial_split_kwargs,
        lam_t=1.0,
        lam_y=0.0,
        k=200,
    )
    algo.fit(dataset)
    effects2 = algo.eval(dataset)
    tune_metric2 = algo.tune_metric(dataset)
    errors2 = evaluator.eval(**effects2)

    algo = DragonSpatial(
        max_iter=2500,
        spatial_split_kwargs=spatial_split_kwargs,
        lam=0.01,
        k=200,
    )
    algo.fit(dataset)
    effects3 = algo.eval(dataset)
    tune_metric3 = algo.tune_metric(dataset)
    errors3 = evaluator.eval(**effects3)

    # algo = DrnetSpatial(
    #     max_iter=2500,
    #     spatial_split_kwargs=spatial_split_kwargs,
    #     lam=0.01,
    #     k=100,
    #     weight_decay=1e-2,
    # )
    # algo.fit(dataset)
    # effects3 = algo.eval(dataset)
    # tune_metric3 = algo.tune_metric(dataset)
    # errors3 = evaluator.eval(**effects3)

    sys.exit(0)
