import sys
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from spacebench.algorithms import SpaceAlgo
from spacebench.env import SpaceDataset
from spacebench.log import LOGGER


class NoMatchError(Exception):
    """Class for no matches found in DAPS matching."""

    def __init__(self):
        super().__init__("No matches found.")


@dataclass
class DapsmMatches:
    treated: np.ndarray = (np.empty(0, dtype=int),)
    controls: np.ndarray = (np.empty(0, dtype=int),)
    scores: np.ndarray = (np.empty(0, dtype=float),)
    spatial_dists: np.ndarray = (np.empty(0, dtype=float),)
    ps_dists: np.ndarray = (np.empty(0, dtype=float),)

    def pairs(self):
        return list(zip(self.treated, self.controls))

    def is_empty(self):
        return len(self.treated) == 0


def dapsm_matching(
    spatial_dists: np.ndarray,
    ps_dists: np.ndarray,
    weight: float = 0.8,
    caliper: float = np.inf,
    caliper_type: Literal["daps", "ps"] = "daps",
    matching_algorithm: Literal["optimal", "greedy"] = "optimal",
    target_group: Literal["treated", "controls"] = "treated",
    warn: bool = True,
):
    """Spatially weighted matching.

    Arguments
    ---------
    distmat: np.ndarray
        A matrix of distances between treated (rows) and controls (columns).
    ps_dist: np.ndarray
        A matrix of absolute differences in propensity scores between treated
        (rows) and controls (cols).
    caliper: float
        A caliper for matching (minimum distance acceptable for a match
        to occur), weighted by the standard deviation of the distances
        specified by caliper_type.
    caliper_type: str
        Whether the caliper is set on the DAPS or on the PS.
    weight: float
        A weight for the DAPS.
    matching_algorithm: str
        Whether to use optimal or greedy matching.
    target_group: str
        If target_group == "treated", then the algorithm will find a match
        for each treated unit. If target_group == "controls", then the
        algorithm will find a match for each control unit.

    Returns
    -------
        pairs: type?
            A list of matched pairs.
        match_diff: type?
            A list of differences in DAPS between matched pairs.
    """

    # check inputs
    assert (
        spatial_dists.shape == ps_dists.shape
    ), "distmat and ps_dist must have same shape"
    assert 0 <= weight <= 1, "weight must be between 0 and 1"

    # normalize distances in 0-1
    m, M = spatial_dists.min(), spatial_dists.max()
    wdist = (spatial_dists - m) / (M - m)

    # DAPS matrix
    score = (1 - weight) * wdist + weight * ps_dists

    # potential matches outside caliper are set to inf
    # need to use float_info.max because np.inf raises an error in
    # linear_sum_assignment
    if caliper_type == "daps":
        invalid = score > caliper * np.std(score)
    elif caliper_type == "ps":
        invalid = ps_dists > caliper * np.std(ps_dists)
    else:
        raise NotImplementedError(caliper_type)

    score[invalid] = sys.float_info.max

    # find matches, keep only matches with finite score
    if matching_algorithm == "optimal":
        if target_group == "controls":
            ctrl_indices, trt_indices = linear_sum_assignment(score.T)
        elif target_group == "treated":
            trt_indices, ctrl_indices = linear_sum_assignment(score)
        else:
            raise ValueError(target_group)
    else:
        trt_indices = np.arange(score.shape[0])
        ctrl_indices = np.argmin(score, axis=1)

    if len(trt_indices) == 0:
        if warn:
            warnings.warn("No matches found.")
        return DapsmMatches()

    match_scores = score[trt_indices, ctrl_indices]

    # drop matches that have dapscore inf.
    valid_matched_ix = match_scores < sys.float_info.max

    # return matches
    matches = DapsmMatches(
        treated=trt_indices[valid_matched_ix],
        controls=ctrl_indices[valid_matched_ix],
        scores=match_scores[valid_matched_ix],
        spatial_dists=spatial_dists[
            trt_indices[valid_matched_ix], ctrl_indices[valid_matched_ix]
        ],
        ps_dists=ps_dists[
            trt_indices[valid_matched_ix], ctrl_indices[valid_matched_ix]
        ],
    )

    return matches


def find_spatial_weight(
    covars_treated: np.ndarray,
    covars_controls: np.ndarray,
    ps_dists: np.ndarray,
    spatial_dists: np.ndarray,
    search_values: np.ndarray = np.linspace(0.01, 0.99, 10),
    balance_cutoff: float = 0.5,
    **kwargs,
):
    """Find spatial weight for spatially weighted matching using binary search.
    Pick the smallest weight that satisfied the cutoff.

    Arguments
    ---------

    covars_treated: np.ndarray
        A matrix of covariates of treated.
    covars_controls: np.ndarray
        A matrix of covariates of controls.
    ps_dist: np.ndarray
        A matrix of absolute differences in propensity scores
    spatial_dists: np.ndarray
        A matrix of distances between treated (rows) and controls (columns).
    cutoff: type?
        A maximum standardized difference in covariates between matched pairs.
    search_values: type?
        values to search for the weight.
    max_attempts: type?
        maximum number of attempts (recursion depth) to find a weight
        that satisfies the cutoff.
    **kwargs:
        optiones to be passed to dapsm_matching.

    Returns
    -------
        weight: type?
            A spatial weight used in matching.
        pairs: list
            A list of matched pairs.
    """
    # start recursion with middle of search interval
    found_weight = None
    found_matches = DapsmMatches()

    # go from lower to higher spatial weights until cutoff is satisfied
    balances = []
    for weight in search_values:
        matches = dapsm_matching(
            spatial_dists=spatial_dists,
            ps_dists=ps_dists,
            weight=weight,
            warn=False,
            **kwargs,
        )

        if not matches.is_empty():  # if matches exist in daps_out
            mean_trt = covars_treated[matches.treated].mean(axis=0)
            mean_ctrl = covars_controls[matches.controls].mean(axis=0)
            sd_trt = covars_treated[matches.treated].std(axis=0)
            std_diff = np.abs(mean_trt - mean_ctrl) / sd_trt
            balances.append(std_diff.max())
            if np.all(std_diff < balance_cutoff):
                found_weight = weight
                found_matches = matches
                break
        else:
            balances.append(np.nan)

    if found_weight is None:
        raise NoMatchError()

    return found_weight, found_matches, balances


def dapsm(
    outcome_treated: np.ndarray,
    outcome_controls: np.ndarray,
    covars_treated: np.ndarray,
    covars_controls: np.ndarray,
    ps_dists: np.ndarray,
    spatial_dists: np.ndarray,
    search_values: np.ndarray = np.linspace(0.01, 0.99, 10),
    balance_cutoff: float = 0.5,
    **kwargs,
):
    """Implementation of the DAPS matching algorithm and estimation of the
    average treatment effect on the treated (ATT).

    Arguments
    ---------
    outcome_treated:
        outcome variable of treated.
    outcome_controls:
        outcome variable of controls.
    covars_treated:
        matrix of covariates of treated.
    covars_controls:
        matrix of covariates of controls.
    ps_dist:
        matrix of absolute differences in propensity scores
    spatial_dists:
        matrix of distances between treated (rows) and controls (columns).
    cutoff:
        maximum standardized difference in covariates between matched pairs.
    search_values:
        values to search for the weight.
    max_attempts:
        maximum number of attempts (recursion depth) to find a weight
        that satisfies the cutoff.
    **kwargs:
        optiones to be passed to dapsm_matching.
    Returns
    -------
        att: average treatment effect on the treated.
        weight: spatial weight used in matching.
        pairs: list of matched pairs.
    """

    weight, matches, balances = find_spatial_weight(
        covars_treated=covars_treated,
        covars_controls=covars_controls,
        ps_dists=ps_dists,
        spatial_dists=spatial_dists,
        search_values=search_values,
        balance_cutoff=balance_cutoff,
        **kwargs,
    )

    # ATT estimation
    att = (outcome_treated[matches.treated] - outcome_controls[matches.controls]).mean()

    return att, weight, matches, balances


class DAPSm(SpaceAlgo):
    """
    Wrapper for DAPS matching algorithm for use with causal datasets
    """
    supports_continuous = False
    supports_binary = True

    def __init__(
        self,
        spatial_weight: float = 0.5,
        propensity_score_penalty_value: float = 0.1,
        propensity_score_penalty_type: Literal["l1", "l2", "elasticnet"] = "l2",
        ps_clip: float = 1e-2,
        caliper_type: Literal["daps", "ps"] = "daps",
        matching_algorithm: Literal["optimal", "greedy"] = "optimal",
    ):
        """Initialize DAPSm method.

        Arguments
        ---------
        spatial_weight: np.ndarray
            weight assigned to the spatial distance vs propensity score in matching.
        propensity_score_penalty_value: float
            penalty value for propensity score model.
        propensity_score_penalty_type: str
            penalty type for propensity score model.
        ps_clip: float
            clip propensity scores to be in [ps_clip, 1 - ps_clip].
        caliper_type: str
            whether the caliper is set on the DAPS or on the PS.
        matching_algorithm: str
            whether to use optimal or greedy matching.

        balance_cutoff: float
            maximum standardized difference in covariates between matched pairs.
        """
        super().__init__()
        self.spatial_weight = spatial_weight
        self.propensity_score_penalty_value = propensity_score_penalty_value
        self.propensity_score_penalty_type = propensity_score_penalty_type
        self.ps_clip = ps_clip
        self.dapsm_kwargs = {
            "caliper_type": caliper_type,
            "matching_algorithm": matching_algorithm,
            "search_values": [spatial_weight],
            "balance_cutoff": np.inf,
        }

    def fit(self, dataset: SpaceDataset):
        assert dataset.has_binary_treatment(), "treatment of dataset must be binary"
        treatment = dataset.treatment.astype(bool)

        LOGGER.debug("Computing distance matrix from treated to untreated")
        # fit coords to unit square
        m = dataset.coordinates.min(axis=0)
        M = dataset.coordinates.max(axis=0)
        coords = (dataset.coordinates - m) / (M - m)
        coords_1 = coords[treatment]
        coords_0 = coords[~treatment]
        spatial_dists = np.sqrt(
            np.square(coords_1[:, None] - coords_0[None, :]).sum(axis=-1)
        )

        # standardize covariates
        mu = dataset.covariates.mean(axis=0)
        sd = dataset.covariates.std(axis=0) + 1e-3
        covars = (dataset.covariates - mu) / sd

        # split covars, outcome in treated and controls
        covars_1 = covars[treatment]
        covars_0 = covars[~treatment]
        outcome_1 = dataset.outcome[treatment]
        outcome_0 = dataset.outcome[~treatment]

        # fit propensity score model
        LOGGER.debug("Fitting propensity score model")
        model = LogisticRegression(
            penalty=self.propensity_score_penalty_type,
            C=self.propensity_score_penalty_value,
            solver="liblinear",
        )
        model.fit(covars, treatment)
        ps = model.predict_proba(covars)[:, 1]
        ps = np.clip(ps, self.ps_clip, 1 - self.ps_clip)
        ps_dists = np.abs(ps[treatment, None] - ps[None, ~treatment])

        # call DAPSm, catch for no matches
        self.att, *_, (self.att_balance, ) = dapsm(
            outcome_treated=outcome_1,
            outcome_controls=outcome_0,
            covars_treated=covars_1,
            covars_controls=covars_0,
            ps_dists=ps_dists,
            spatial_dists=spatial_dists,
            target_group="controls",
            **self.dapsm_kwargs,
        )

        # compute the atc by inverting the treatment, but add a minus sign
        # atc(t) = - att(1 - t)
        self.atc, *_, (self.atc_balance, ) = dapsm(
            outcome_treated=outcome_1,
            outcome_controls=outcome_0,
            covars_treated=covars_1,
            covars_controls=covars_0,
            ps_dists=ps_dists,
            spatial_dists=spatial_dists,
            target_group="controls",
            **self.dapsm_kwargs,
        )

    def eval(self, dataset: SpaceDataset):
        t = dataset.treatment.astype(bool)
        y = dataset.outcome
        w = np.nanmean(t)
        ate = w * self.att + (1 - w) * self.atc
        erf = [(1 - w) * y[~t].mean() + w * (y[t] - self.att).mean()]
        erf.append(erf[0] + ate)

        ite = np.zeros((len(y), 2))
        ite[t, 1] = y[t]
        ite[~t, 0] = y[~t]
        ite[~t, 1] = y[~t] + self.atc
        ite[t, 0] = y[t] - self.att

        return {"att": self.att, "ate": ate, "atc": self.atc, "erf": erf, "ite": ite}

    def tune_metric(self, dataset: SpaceDataset):
        # Returns the negative weighted covariate balance with spatial weight bonus
        w = np.nanmean(dataset.treatment.astype(bool))
        balance = w * self.att_balance + (1 - w) * self.atc_balance
        return - balance - 0.1 * self.spatial_weight

    @property
    def available_estimands(self):
        return ["att", "ate", "atc"]
