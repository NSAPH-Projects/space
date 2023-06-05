from collections import defaultdict

import numpy as np

from spacebench.env import SpaceEnv, SpaceDataset


class DatasetEvaluator:
    """Class for evaluating the performance of a causal inference method
    in a specific SpaceDataset."""

    def __init__(self, dataset: SpaceDataset) -> None:
        self.dataset = dataset
        self.buffer = defaultdict(list)

    def eval(
        self,
        ate: np.ndarray | None = None,
        att: np.ndarray | None = None,
        counterfactuals: np.ndarray | None = None,
        erf: np.ndarray | None = None,
    ) -> dict[str, float]:
        errors = {}
        cf_true = self.dataset.counterfactuals
        t = self.dataset.treatment

        if ate is not None:
            assert self.dataset.has_binary_treatment(), "ATE only valid in binary"
            ate_true = (cf_true[:, 1] - cf_true[:, 0]).mean()
            errors["ate_error"] = ate - ate_true
            errors["ate_se"] = np.square(errors["ate_error"])

        if att is not None:
            assert self.dataset.has_binary_treatment(), "ATT only valid in binary"
            assert np.min(t) == 0.0 and np.max(t) == 1.0
            att_true = (cf_true[t == 1, 1] - cf_true[t == 1, 0]).mean()
            errors["att_error"] = att - att_true
            errors["att_se"] = np.square(errors["att_error"])

        if counterfactuals is not None:
            errors["pehe_curve"] = ((counterfactuals - cf_true) ** 2).mean(0)
            errors["pehe_av"] = errors["pehe_curve"].mean()

        if erf is not None:
            erf_true = self.dataset.erf()
            errors["erf_error"] = erf - erf_true
            errors["erf_av"] = np.square(errors["erf_error"]).mean()

        return errors


class EnvEvaluator:
    """Class for evaluating the performance of a causal inference method
    in a specific SpaceEnv."""

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
        """Add a dataset to the buffer"""
        evaluator = DatasetEvaluator(dataset)
        metrics = evaluator.eval(
            ate=ate,
            att=att,
            counterfactuals=counterfactuals,
            erf=erf,
        )
        for k, v in metrics.items():
            self.buffer[k].append(v)

    def summarize(self) -> dict[str, float]:
        """Evaluate the error in causal prediction"""
        if len(self.buffer) == 0:
            raise ValueError("Use add first")

        res = dict()

        # ate bias and variance
        if "ate_error" in self.buffer:
            res["ate_bias"] = np.array(self.buffer["ate_error"]).mean()
            res["ate_variance"] = np.array(self.buffer["ate_error"]).var()

        # att bias and variance
        if "att_error" in self.buffer:
            res["att_bias"] = np.array(self.buffer["att"]).mean()
            res["att_variance"] = np.array(self.buffer["att"]).var()

        # pehe bias and variance
        if "pehe" in self.buffer:
            res["pehe"] = np.array(self.buffer["pehe"]).mean(0)

        # response curve bias and variance
        if "erf_error" in self.buffer:
            rc = np.array(self.buffer["erf_error"])
            res["erf_bias"] = rc.mean(0)
            res["erf_variance"] = rc.var(0)

        return res
