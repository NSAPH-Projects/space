import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from spacebench.algorithms import SpaceAlgo
from spacebench.env import SpaceDataset
from spacebench.log import LOGGER
from spacebench.algorithms.datautils import spatial_train_test_split


def compute_phi(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    Compute the radial basis matrix using PyTorch.

    Arguments
    ----------
        c1 (torch.Tensor): Coords tensor of size n x d for the rows of the radial basis.
        c2 (torch.Tensor): Coords tensor of size k x d for the columns of the radial basis.

    Returns
    ------------
        torch.Tensor: Radial basis matrix of size n x k.
    """
    d = c1.shape[1]
    assert 1 <= d <= 3, "Only 1D, 2D, and 3D coordinates are supported."
    diffs = c1.unsqueeze(1) - c2.unsqueeze(0)
    dist = diffs.pow(2).sum(dim=-1).sqrt()

    if d in (1, 3):
        return dist.pow(4 - d)
    else:
        return dist.pow(2) * torch.log(dist + 1e-6)


def tps_pred(
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Thin Plate Spline predictions given coordinates, covariates, and parameters.

    Arguments
    ----------
        coords (torch.Tensor): Coordinate matrix of size (n, d).
        cp_coords (list[int]): Coordinate matrix of contorl points of size (k, d).
        covars (torch.Tensor): Covariate matrix of size (n, p).
        params (torch.Tensor): Model parameters of length 1 + d + p + k

    Returns
    ----------
        torch.Tensor: Predicted values of length n.
    """
    Phi = compute_phi(coords, cp_coords)
    intercept = torch.ones((coords.shape[0], 1))
    A = torch.cat([intercept, coords, covars, Phi], dim=1)
    return A @ params


def tps_loss(
    y: torch.Tensor,
    coords: torch.Tensor,
    cp_coords: torch.Tensor,
    covars: torch.Tensor,
    params: torch.Tensor,
    lam: float,
    binary_loss: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the loss for Thin Plate Spline regression.

    Arguments
    ----------
        y (torch.Tensor): Target values of length n.
        coords (torch.Tensor): Coordinate matrix of size (n, d).
        cp_coords (list[int]): Coordinate matrix of contorl points of size (k, d).
        covars (torch.Tensor): Covariate matrix of size (n, p).
        params (torch.Tensor): Model parameters of length 1 + d + p + k
        lam (float): Regularization parameter.
        binary_loss (bool, optional): Whether to use binary cross entropy loss.
            Defaults to False.
        mask (torch.Tensor, optional): Mask of length n to use for loss. Defaults to None.

    Returns
    ----------
        torch.Tensor: Loss value.
    """
    k = cp_coords.shape[0]

    pred = tps_pred(coords, cp_coords, covars, params)
    if not binary_loss:
        pred_loss = F.mse_loss(pred, y, reduction="none")
    else:
        pred_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")

    pred_loss = (pred_loss * mask).mean() if mask is not None else pred_loss.mean()

    c = compute_phi(coords, cp_coords) @ params[-k:, None]  # vector of size k x 1
    c = params[-k:, None]
    reg_loss = c.pow(2).sum()

    return pred_loss + lam * reg_loss


def tps_opt(
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
) -> torch.Tensor:
    """
    Optimize the parameters for Thin Plate Spline regression.
    It is recommended that the outcome and covariates are
    all centered and scaled to have unit variance.

    Arguments
    ----------
        y (torch.Tensor): Target values of length n.
        coords (torch.Tensor): Coordinate matrix of size (n, d).
        cp_coords (torch.Tensor): Coordinate matrix of control points of size (k, d).
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

    Returns:
        torch.Tensor: Optimized parameters of length 1 + d + p + k
        the first element is the intercept, the next d elements are the
        coefficients of the coordinates, the next p elements are the
        coefficients of the covariates, and the last k elements are the
        coefficients of the radial basis.
    """
    d = coords.shape[1]
    p = covars.shape[1]
    k = cp_coords.shape[0]

    # Initialization of params
    params = torch.zeros(1 + d + p + k, requires_grad=True)
    opt = torch.optim.Adam([params], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.9,
        patience=plateau_patience,
        verbose=verbose,
        min_lr=1e-8,
    )

    # Run the optimizer
    it = 0

    while it < max_iter:
        opt.zero_grad()
        loss = tps_loss(y, coords, cp_coords, covars, params, lam, binary_loss, mask=mask)
        loss.backward()
        opt.step()
        sched.step(loss)

        # # Check for convergence
        # max_grad = params.grad.abs().max()
        # if max_grad < atol:
        #     LOGGER.info(f"TPS converged after {it} iterations.")
        #     break

        it += 1

    if it == max_iter - 1:
        LOGGER.warning(f"TPS did not converge after {max_iter} iterations.")

    return params.detach()


class Spatial(SpaceAlgo):
    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam: float = 0.001,
        spatial_split_kwargs: dict | None = None,
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

    def fit(self, dataset: SpaceDataset):
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
        self.params = tps_opt(
            y,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
        )
        # 0 coef is intercept, 1,2 are for coords, 3 is treatment
        self.t_coef = (self.params[3] * self.y_std / self.inputs_std[0]).item()

    def eval(self, dataset: SpaceDataset):
        ite = [
            dataset.outcome + self.t_coef * (a - dataset.treatment)
            for a in dataset.treatment_values
        ]
        ite = np.stack(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef

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
        with torch.no_grad():
            pred = tps_pred(coords, self.cp_coords, inputs, self.params)
            loss = F.mse_loss(pred, y, reduction="none")

        if self.mask is not None:
            loss = (loss * (1.0 - self.mask)).mean()
        else:
            LOGGER.warning("No mask specified for the tune metric. Using full dataset.")
            loss = loss.mean()

        return loss.item()


class SpatialPlus(SpaceAlgo):
    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        k: int = 100,
        max_iter: int = 20_000,
        lam_t: float = 0.001,
        lam_y: float = 0.001,
        spatial_split_kwargs: dict | None = None,
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

    def fit(self, dataset: SpaceDataset):
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

        # standardize t if not binary treatment
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
        self.t_params = tps_opt(
            t,
            coords,
            self.cp_coords,
            covars,
            lam=self.lam_t,
            max_iter=self.max_iter,
            verbose=False,
            binary_loss=dataset.has_binary_treatment(),
            mask=self.mask,
        )

        # predict
        y = torch.FloatTensor(dataset.outcome)
        t_pred = tps_pred(coords, self.cp_coords, covars, self.t_params)
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
        self.y_params = tps_opt(
            y,
            coords,
            self.cp_coords,
            inputs,
            lam=self.lam_y,
            max_iter=self.max_iter,
            verbose=False,
            mask=self.mask,
        )

        # 0 coef is intercept, 1,2 are for coords, 3 is treatment
        self.t_coef = (self.y_params[3] * self.y_std / self.t_mu).item()

    def eval(self, dataset: SpaceDataset):
        ite = [
            dataset.outcome + self.t_coef * (a - dataset.treatment)
            for a in dataset.treatment_values
        ]
        ite = np.stack(ite, axis=1)

        effects = {"erf": ite.mean(0), "ite": ite}

        if dataset.has_binary_treatment():
            effects["ate"] = self.t_coef

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

        # standardize t if not binary treatment
        if not dataset.has_binary_treatment():
            t = (t - self.t_mu) / self.t_std

        t_loss = tps_loss(
            t,
            coords,
            self.cp_coords,
            covars,
            self.t_params,
            lam=0.0,
            binary_loss=dataset.has_binary_treatment(),
            mask=tune_mask,
        )
        t_pred = tps_pred(coords, self.cp_coords, covars, self.t_params)
        if dataset.has_binary_treatment():
            t_pred = torch.sigmoid(t_pred)
        t_resid = t - t_pred

        inputs = torch.cat(
            [t_resid[:, None], torch.FloatTensor(dataset.covariates)], dim=1
        )
        inputs = (inputs - self.inputs_mu) / self.inputs_std
        y = (y - self.y_mu) / self.y_std
        y_loss = tps_loss(
            y,
            coords,
            self.cp_coords,
            inputs,
            self.y_params,
            lam=0.0,
            mask=tune_mask,
        )

        return t_loss.item() + y_loss.item()


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

    env = spacebench.SpaceEnv("healthd_hhinco_mortality_cont")
    dataset = env.make()
    evaluator = spacebench.DatasetEvaluator(dataset)

    # Run Spatial
    algo = Spatial(max_iter=1000, spatial_split_kwargs=spatial_split_kwargs)
    algo.fit(dataset)
    effects1 = algo.eval(dataset)
    tune_metric1 = algo.tune_metric(dataset)
    errors1 = evaluator.eval(**effects1)

    # Run SpatialPlus
    algo = SpatialPlus(
        max_iter=1000,
        spatial_split_kwargs=spatial_split_kwargs,
        lam_t=1.0,
        lam_y=0.0,
        k=200,
    )
    algo.fit(dataset)
    effects2 = algo.eval(dataset)
    tune_metric2 = algo.tune_metric(dataset)
    errors2 = evaluator.eval(**effects2)

    sys.exit(0)
    