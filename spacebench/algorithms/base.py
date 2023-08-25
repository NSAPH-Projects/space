from abc import abstractmethod

from spacebench.env import SpaceDataset


class SpaceAlgo:
    supports_binary: bool | None = None
    supports_continuous: bool | None = None

    def __init__(self):
        self.__validate_class()

    @classmethod
    def __validate_class(cls):
        """Initialize the spatial method."""
        assert (
            cls.supports_binary is not None
        ), "supports_binary must be set as class attribute"
        assert (
            cls.supports_continuous is not None
        ), "supports_continuous must be set as class attribute"

    @abstractmethod
    def fit(self, dataset: SpaceDataset, **kwargs) -> None:
        """Estimates the causal effect of a treatment on an outcome.
        The available estimands are defined by the estimands() method.
        The method must either return a single dictionary with the
        estimand as key and the estimated effect as value or a tuple
        of two dictionaries, where the first dictionary contains the
        estimands and the second dictionary contains additional information
        about the estimation process.

        Arguments
        ---------
        dataset : SpaceDataset
            The dataset used to learn the causal effect.
        """
        pass

    @property
    @abstractmethod
    def available_estimands(self) -> list[str]:
        """Aavailable estimands.

        Returns
        -------
        list[str]
            A list of available estimands (erf, ate, att, ite, etc.)
        """
        pass

    @abstractmethod
    def eval(self, dataset: SpaceDataset) -> dict[str, float | list[float]]:
        """Return a dictionary with the estimated effects for all
        available estimands.

        Arguments
        ---------
        dataset : SpaceDataset
            The dataset to be evaluated.

        Returns
        -------
        dict[str, float | list[float]]
            A dictionary with the estimands as keys and the estimated
            effects as values.
        """
        pass

    def tune_metric(self, datset: SpaceDataset) -> float:
        """Return a metric to be used for hyperparameter tuning.

        Arguments
        ---------
        dataset : SpaceDataset
            The dataset to be evaluated.

        Returns
        -------
        float
            A metric to be used for hyperparameter tuning.
        """
        raise NotImplementedError
