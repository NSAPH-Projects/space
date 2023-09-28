import sys
import warnings
from typing import Literal
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from spacebench.env import SpaceDataset
from spacebench.algorithms.classes import SpatialMethod


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
    wdist = (spatial_dists - np.min(spatial_dists)) / (
        spatial_dists.max() - spatial_dists.min()
    )

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
        raise ValueError(caliper_type)

    score[invalid] = sys.float_info.max

    # find matches, keep only matches with finite score
    if matching_algorithm == "optimal":
        trt_indices, ctrl_indices = linear_sum_assignment(score)
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
        raise ValueError("No weight found that satisfies the cutoff.")

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

    weight, matches, _ = find_spatial_weight(
        covars_treated=covars_treated,
        covars_controls=covars_controls,
        ps_dists=ps_dists,
        spatial_dists=spatial_dists,
        search_values=search_values,
        balance_cutoff=balance_cutoff,
    )

    # ATT estimation
    att = (outcome_treated[matches.treated] - outcome_controls[matches.controls]).mean()

    return att, weight, matches


class DAPSm(SpatialMethod):
    """Class for implementing the DAPS matching algorithm for use with causal datasets"""

    def __init__(
            self,
            causal_dataset: SpaceDataset,
            ps_score: np.ndarray,
            spatial_dists: np.ndarray | None = None,
            spatial_dists_full: np.ndarray | None = None,
            search_values: np.ndarray = np.linspace(0.01, 0.99, 10),
            balance_cutoff: float = 0.5,
             **kwargs
        ):
        """Initialize DAPSm class

        Arguments
        ---------
        causal_dataset: SpaceDataset 
            An instance of SpaceDataset class
        ps_score: np.ndarray 
            An array of propensity score of each observation
        spatial_dists: np.ndarray 
            A matrix of distances between treated (rows) and controls (columns).
            either spatial_dists or spatial_dists_full must be provided.
        spatial_dists_full: np.ndarray 
            A matrix of distances between all observations. Either spatial_dists 
            or spatial_dists_full must be provided.
        **kwargs: 
            options to be passed to dapsm function
        """
        # validate args
        if not isinstance(causal_dataset, SpaceDataset):
            raise ValueError("causal_dataset must be an instance" 
                             "of SpaceDataset")
        else:
            assert causal_dataset.has_binary_treatment(), "treatment must be binary"
        assert spatial_dists is not None or spatial_dists_full is not None, (
            "either spatial_dists or spatial_dists_full must be provided"
        )
        treatment_values = causal_dataset.treatment_values
        tix = causal_dataset.treatment == treatment_values[1]
        self.tix = tix

        # standardize covariates
        covars = causal_dataset.covariates.copy()
        covars -= covars.mean(axis=0)
        covars /= covars.std(axis=0)

        self.outcome_treated = causal_dataset.outcome[tix]
        self.outcome_controls = causal_dataset.outcome[~tix]
        self.covars_treated = covars[tix]
        self.covars_controls = covars[~tix]

        if spatial_dists is None:
            self.spatial_dists = spatial_dists_full[tix][:, ~tix]
        else:
            self.spatial_dists = spatial_dists
        self.ps_dists = np.abs(ps_score[tix, None] - ps_score[None, ~tix])
        self.dapsm_kwargs = kwargs
        self.search_values = search_values
        self.balance_cutoff = balance_cutoff

    @classmethod
    def estimands(cls):
        return ["att", "ate", "atc"]
    
    def estimate(self, estimand: str):
        assert estimand in self.estimands(), \
            f"estimand {estimand} not available; see the estimands method"
        if estimand == "att":
            return dapsm(
                outcome_treated=self.outcome_treated,
                outcome_controls=self.outcome_controls,
                covars_treated=self.covars_treated,
                covars_controls=self.covars_controls,
                ps_dists=self.ps_dists,
                spatial_dists=self.spatial_dists,
                search_values=self.search_values,
                balance_cutoff=self.balance_cutoff,
                **self.dapsm_kwargs,
            )
        elif estimand == "atc":
            return dapsm(
                outcome_treated=self.outcome_controls,
                outcome_controls=self.outcome_treated,
                covars_treated=self.covars_controls,
                covars_controls=self.covars_treated,
                ps_dists=self.ps_dists.T,
                spatial_dists=self.spatial_dists.T,
                search_values=self.search_values,
                balance_cutoff=self.balance_cutoff,
                **self.dapsm_kwargs,
            )
        elif estimand == "ate":
            att, _, _ = self.estimate("att")
            atc, _, _ = self.estimate("atc")
            w = np.mean(self.tix)
            return w * att + (1 - w) * atc
