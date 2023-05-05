from abc import ABC, abstractmethod


class SpatialMethod:
    @abstractmethod
    def estimate(self, metric: str, **kwargs):
        pass

    def available_metrics(self):
        raise NotImplementedError