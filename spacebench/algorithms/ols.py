import numpy as np
from spacebench import SpaceDataset
from sklearn.linear_model import Ridge


class OLS:
    """
    Simple wrapper for OLS regression for comparison purposes.
    """

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


if __name__ == "__main__":
    import sys
    import spacebench

    env_name = spacebench.DataMaster().list_envs()[0]
    env = spacebench.SpaceEnv(env_name)
    dataset = env.make()

    algo = OLS()
    algo.fit(dataset)
    effects = algo.eval(dataset)
    sys.exit(0)
