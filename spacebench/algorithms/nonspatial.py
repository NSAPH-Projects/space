import numpy as np
from spacebench import SpaceDataset
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from spacebench.algorithms import SpaceAlgo
from typing import Optional

from spacebench.env import SpaceDataset


class OLS(SpaceAlgo):
    """
    Simple wrapper for OLS regression for comparison purposes.
    """

    supports_binary = True
    supports_continuous = True

    def fit(self, dataset: SpaceDataset):
        self.model = Ridge(alpha=1e-6, fit_intercept=True)
        inputs = np.concatenate(
            [dataset.treatment[:, None], dataset.covariates], axis=1
        )
        self.model.fit(inputs, dataset.outcome)

    def eval(self, dataset: SpaceDataset):
        treatment_values = dataset.treatment_values
        inputs = np.concatenate(
            [dataset.treatment[:, None], dataset.covariates], axis=1
        )
        preds = self.model.predict(inputs)
        residuals = dataset.outcome - preds

        mu_cf = []
        for a in treatment_values:
            treatment_a = np.full((dataset.size(), 1), a)
            inputs_a = np.concatenate([treatment_a, dataset.covariates], axis=1)
            pred_a = self.model.predict(inputs_a)
            mu_cf.append(pred_a)
        mu_cf = np.stack(mu_cf, axis=1)

        effects = {
            "erf": mu_cf.mean(0),
            "ite": mu_cf + residuals[:, None],
        }

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]


class XGBoost(SpaceAlgo):
    """
    Wrapper for XGBoost regression for comparison purposes.
    """

    supports_binary = True
    supports_continuous = True

    def __init__(
        self,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        n_estimators: int = 100,
        coords: bool = False,
    ) -> None:
        self.model_kwargs = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
        }
        self.coords = coords

    def fit(self, dataset: SpaceDataset):
        self.model = XGBRegressor(**self.model_kwargs)
        inputs = np.concatenate(
            [dataset.treatment[:, None], dataset.covariates], axis=1
        )
        if self.coords:
            inputs = np.concatenate([inputs, dataset.coordinates], axis=1)
        self.model.fit(inputs, dataset.outcome)

    def eval(self, dataset: SpaceDataset):
        treatment_values = dataset.treatment_values
        inputs = np.concatenate(
            [dataset.treatment[:, None], dataset.covariates], axis=1
        )
        if self.coords:
            inputs = np.concatenate([inputs, dataset.coordinates], axis=1)
        preds = self.model.predict(inputs)
        residuals = dataset.outcome - preds

        mu_cf = []
        for a in treatment_values:
            treatment_a = np.full((dataset.size(), 1), a)
            inputs_a = np.concatenate([treatment_a, dataset.covariates], axis=1)
            if self.coords:
                inputs_a = np.concatenate([inputs_a, dataset.coordinates], axis=1)
            pred_a = self.model.predict(inputs_a)
            mu_cf.append(pred_a)
        mu_cf = np.stack(mu_cf, axis=1)

        effects = {
            "erf": mu_cf.mean(0),
            "ite": mu_cf + residuals[:, None],
        }

        if dataset.has_binary_treatment():
            effects["ate"] = effects["erf"][1] - effects["erf"][0]

        return effects

    def available_estimands(self):
        return ["ate", "erf", "ite"]

    def tune_metric(self, dataset: SpaceDataset) -> float:
        inputs = np.concatenate(
            [dataset.treatment[:, None], dataset.covariates], axis=1
        )
        if self.coords:
            inputs = np.concatenate([inputs, dataset.coordinates], axis=1)
        preds = self.model.predict(inputs)
        return np.mean((dataset.outcome - preds) ** 2)


if __name__ == "__main__":
    import sys
    import spacebench

    env_name = spacebench.DataMaster().list_envs()[0]
    env = spacebench.SpaceEnv(env_name)
    dataset = env.make()

    algo = OLS()
    algo.fit(dataset)
    effects1 = algo.eval(dataset)

    algo = XGBoost()
    algo.fit(dataset)
    effects2 = algo.eval(dataset)

    sys.exit(0)
