from collections import defaultdict

import numpy as np

from spacebench.env import SpaceEnv, SpaceDataset


class DatasetEvaluator:
    """
    Class for evaluating the performance of a causal inference method
    in a specific SpaceDataset.
    """

    def __init__(
        self,
        dataset: SpaceDataset,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> None:
        self.dataset = dataset
        self.buffer = defaultdict(list)
        self.tmin = tmin
        self.tmax = tmax
        self.mask = np.ones(len(dataset.treatment_values), dtype=bool)
        if tmin is not None and tmax is not None:
            assert tmin < tmax
            self.mask = (dataset.treatment_values >= tmin) & (dataset.treatment_values <= tmax)

    def eval(
        self,
        ate: np.ndarray | None = None,
        att: np.ndarray | None = None,
        atc: np.ndarray | None = None,
        ite: np.ndarray | None = None,
        erf: np.ndarray | None = None,
    ) -> dict[str, float]:
        errors = {}
        cf_true = self.dataset.counterfactuals[:, self.mask]
        if ite is not None:
            ite = ite[:, self.mask]
        t = self.dataset.treatment
        scale = np.std(self.dataset.outcome)

        if ate is not None:
            assert self.dataset.has_binary_treatment(), "ATE only valid in binary"
            ate_true = (cf_true[:, 1] - cf_true[:, 0]).mean()
            errors["ate_error"] = (ate - ate_true) / scale
            errors["ate"] = np.abs(errors["ate_error"])

        if atc is not None:
            assert self.dataset.has_binary_treatment(), "ATC only valid in binary"
            atc_true = (cf_true[t == 0, 1] - cf_true[t == 0, 0]).mean()
            errors["atc_error"] = (atc - atc_true) / scale
            errors["atc"] = np.abs(errors["atc_error"])

        if att is not None:
            assert self.dataset.has_binary_treatment(), "ATT only valid in binary"
            assert np.min(t) == 0.0 and np.max(t) == 1.0
            att_true = (cf_true[t == 1, 1] - cf_true[t == 1, 0]).mean()
            errors["att_error"] = (att - att_true) / scale
            errors["att"] = np.abs(errors["att_error"])

        if ite is not None:
            # compute the precision at estimating heterogeneous effects (PEHE)
            cferr = (ite - cf_true) / scale
            errors["ite_curve"] = np.sqrt((cferr**2).mean(0))
            errors["ite"] = errors["ite_curve"].mean()

        if erf is not None:
            erf_true = self.dataset.erf()
            errors["erf_error"] = (erf - erf_true) / scale
            errors["erf"] = np.abs(errors["erf_error"]).mean()

        return errors


class EnvEvaluator:
    """
    Class for evaluating the performance of a causal inference method
    in a specific SpaceEnv.
    """

    def __init__(self, env: SpaceEnv) -> None:
        self.env = env
        self.buffer = defaultdict(list)

    def add(
        self,
        dataset: SpaceDataset,
        ate: np.ndarray | None = None,
        att: np.ndarray | None = None,
        counterfactuals: np.ndarray | None = None,
        erf: np.ndarray | None = None,
    ) -> None:
        """
        Add a dataset to the buffer.
        """
        evaluator = DatasetEvaluator(dataset)
        metrics = evaluator.eval(
            ate=ate,
            att=att,
            ite=counterfactuals,
            erf=erf,
        )
        for k, v in metrics.items():
            self.buffer[k].append(v)

    def summarize(self) -> dict[str, float]:
        """
        Evaluate the error in causal prediction.
        """
        if len(self.buffer) == 0:
            raise ValueError("Use add first")

        res = dict()

        # ate bias and variance
        if "ate_error" in self.buffer:
            res["ate_bias"] = np.array(self.buffer["ate_error"]).mean()
            res["ate_variance"] = np.array(self.buffer["ate_error"]).var()

        # att bias and variance
        if "att_error" in self.buffer:
            res["att_bias"] = np.array(self.buffer["att_error"]).mean()
            res["att_variance"] = np.array(self.buffer["att_error"]).var()

        # pehe bias and variance
        if "ite_curve" in self.buffer:
            res["ite_curve"] = np.array(self.buffer["ite_curve"]).mean(0)
            res["ite"] = np.array(self.buffer["ite"]).mean(0)

        # response curve bias and variance
        if "erf_error" in self.buffer:
            rc = np.array(self.buffer["erf_error"])
            res["erf_bias"] = rc.mean(0)
            res["erf_variance"] = rc.var(0)

        return res
