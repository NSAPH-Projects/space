from functools import reduce
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
from sklearn.gaussian_process.kernels import WhiteKernel


class ErrorSampler(ABC):
    """Base class for error samplers"""

    def __init__(self, sampler_type: str) -> None:
        self.sampler_type = sampler_type

    @abstractmethod
    def sample(self) -> pd.DataFrame:
        """Samples error for a dataset"""
        pass


def _make_kernel(error_params: dict) -> kernels.Kernel:
    """Creates a kernel object from a dictionary of parameters"""
    # obtain actual kernel objects
    for k in error_params:
        error_params[k] = getattr(kernels, k)(**error_params[k])

    # replace the previous line with a product
    return reduce(lambda x, y: x * y, error_params.values())


class GPSampler(ErrorSampler):
    def __init__(self, error_params: str | dict):
        super().__init__("gp")
        if isinstance(error_params, str):
            error_params = json.loads(error_params)
        self.kernel = _make_kernel(error_params)

    def sample(
        self, inputs: pd.DataFrame | np.ndarray | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Samples error for a dataset"""
        gp = GaussianProcessRegressor(kernel=self.kernel + WhiteKernel(0.001))
        return gp.sample_y(inputs)[:, 0]


if __name__ == "__main__":
    error_params = '{"ConstantKernel": {"constant_value": 1.764076575399331, "constant_value_bounds": [0.0001, 10]}, "RBF": {"length_scale": [0.03261315862423933, 0.0002321701500986085], "length_scale_bounds": [0.0001, 0.1]}}'
    s = GPSampler(error_params)
    print("ok")
