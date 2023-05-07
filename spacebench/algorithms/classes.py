from abc import ABC, abstractmethod


class SpatialMethod:
    @abstractmethod
    def estimate(self, estimand: str, **kwargs):
        """Estimates the causal effect of a treatment on an outcome.
        The available estimands are defined by the estimands() method."""
        raise NotImplementedError

    @classmethod
    def estimands(cls):
        """Returns a list of causal estimands that this method can estimate"""
        raise NotImplementedError