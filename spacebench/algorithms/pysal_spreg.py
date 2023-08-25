import libpysal as lp
import numpy as np
from pysal.model import spreg

from spacebench import SpaceDataset
from spacebench.log import LOGGER


class GMError:
    """
    Wrapper of PySAL GM_Error model.
    """

    def fit(self, dataset: SpaceDataset):
        x = np.concatenate([dataset.treatment[:, None], dataset.covariates], axis=1)
        y = dataset.outcome

        LOGGER.debug("Computing spatial weights")
        w = lp.weights.util.full2W(dataset.adjacency_matrix())

        self.model = spreg.GM_Error(x=x, y=y, w=w)
        self.t_coef = self.model.betas[1, 0]

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


class GMLag:
    """
    Wrapper of PySAL GM_Lag model.
    """

    def __init__(self, w_lags: int = 1):
        """
        Arguments
        ----------

        w_lags : int
            Number of spatial lags to include in the model. See the GM_Lag
            documentation for more details.
        """
        self.w_lags = w_lags

    def fit(self, dataset: SpaceDataset):
        x = np.concatenate([dataset.treatment[:, None], dataset.covariates], axis=1)
        y = dataset.outcome

        LOGGER.debug("Computing spatial weights")
        w = lp.weights.util.full2W(dataset.adjacency_matrix())

        self.model = spreg.GM_Lag(x=x, y=y, w=w, w_lags=self.w_lags)
        self.t_coef = self.model.betas[1, 0]

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


if __name__ == "__main__":
    import sys
    import spacebench

    env_name = spacebench.DataMaster().list_envs()[0]
    env = spacebench.SpaceEnv(env_name)
    dataset = env.make()

    algo = GMError()
    algo.fit(dataset)
    effects1 = algo.eval(dataset)

    algo = GMLag()
    algo.fit(dataset)
    effects2 = algo.eval(dataset)

    sys.exit(0)
