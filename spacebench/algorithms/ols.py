import numpy as np
from spacebench import SpaceDataset
from typing import Literal
import libpysal as lp
from pysal.model.spreg import GM_Lag, GM_Error
from sklearn.linear_model import Ridge


class OSLModel:
    """
    This class is for implementing linear models, in particular the Ordinary Least
    Squares (OLS) model with Ridge Regression and the PySal models.


    Attributes:
        dataset (SpaceDataset): The input dataset for the model.
        binary_treatment (bool): Indicator for whether the treatment is binary.
        treatment (np.ndarray): The treatment values from the dataset.
        covariates (np.ndarray): The covariates from the dataset.
        outcome (np.ndarray): The outcome values from the dataset.
        trainmat (np.ndarray): The matrix used for training the model.
        tvals (np.ndarray): The treatment values from the dataset.
    """

    def __init__(self, dataset: SpaceDataset, binary_treatment: bool):
        """
        Constructor for the OSLModel class.

        Args:
            dataset (SpaceDataset): The input dataset for the model.
            binary_treatment (bool): Indicator for whether the treatment is binary.
        """

        if not isinstance(dataset, SpaceDataset):
            raise ValueError("causal_dataset must be an instance" "of SpaceDataset")

        self.dataset = dataset
        self.binary_treatment = binary_treatment
        self.treatment = dataset.treatment[:, None]
        self.covariates = dataset.covariates
        self.outcome = dataset.outcome
        self.trainmat = np.hstack([self.covariates, self.treatment])
        self.tvals = dataset.treatment_values

    def ridge(self, nugget: float):
        """
        Fits a Ridge regression model with the given nugget and evaluates counterfactuals.

        Args:
            nugget (float): The regularization strength.

        Returns:
            tuple: Beta values and counterfactuals.
        """
        model = Ridge(alpha=nugget, fit_intercept=True)
        model.fit(self.trainmat, self.outcome)

        beta = model.coef_[-1]
        counterfactuals = self._create_counterfactuals(beta)

        if self.binary_treatment:
            return beta, counterfactuals
        else:
            erf = counterfactuals.mean(0)
            return erf, counterfactuals

    def _process_method(self, method):
        options = {
            "GM_Lag": GM_Lag,
            "GM_Error": GM_Error,
        }
        return options.get(method)

    def sreg(self, method_name: Literal["GM_Lag", "GM_Error"]):
        """
        Runs PySAL regression based on the provided method name.

        Args:
            method_name (Literal): The name of the method for PySAL regression.

        Returns:
            tuple: Beta values and counterfactuals.
        """
        W = lp.weights.util.full2W(self.dataset.adjacency_matrix())

        method = self._process_method(method_name)
        model = method(self.outcome, self.trainmat, w=W)

        beta = model.betas[-2]
        counterfactuals = self._create_counterfactuals(beta)

        return beta, counterfactuals

    def _create_counterfactuals(self, beta):
        """
        Helper function to create counterfactuals.

        Args:
            model (Ridge or PySAL model): The trained model.

        Returns:
            np.ndarray: The counterfactuals.
        """
        counterfactuals = []
        for tval in self.tvals:
            # simplified formula for linear models
            diff = np.squeeze(tval - self.treatment, axis=-1)
            counterfactuals.append(self.outcome + beta * (diff))

        counterfactuals = np.stack(counterfactuals, axis=1)
        return counterfactuals
