import numpy as np
from typing import Literal
from scipy.optimize import linear_sum_assignment


def match(
    score: list[float] | np.ndarray,
    distmat: np.ndarray,
    treatment: list[int|bool],
    weight: float = 0.5,
    target: Literal["treated", "untreated"] = "treated",
    strategy: Literal["greedy", "optimal"] = "greedy",
) -> np.ndarray:
    """
    Compute the optimal match between treated and untreated individuals.
    """
    num_treated = sum(treatment)
    num_untreated = len(treatment) - num_treated

    # make sure score is in the right format
    if isinstance(score, list) or len(score.shape) == 1:
        score = np.array(score)[:, None]

    if not isinstance(treatment, np.ndarray) or treatment.dtype != bool:
        treatment = np.array(treatment, dtype=bool)

    # check at least one treated and untreated
    if num_treated == 0 or num_untreated == 0:
        raise ValueError("At least one treated and untreated individual is required.")

    if target == "treated":
        matches = []
        needs_match = treatment & np.ones(len(treatment), dtype=bool)
        while any(needs_match):
            # get indices of treaeted and untreated from treatment
            treated_idx = np.where(treatment & needs_match)[0]  # = N
            untreated_idx = np.where(~treatment)[0]  # = M

            # make distance matrix of size N x M
            D = distmat[treated_idx, :][:, untreated_idx]  # = N x M

            # make propensity score N X M distance matrix
            score_dist2 = 0.0
            for j in range(score.shape[1]):
                score_dist2 += (score[treated_idx, j][:, None] - score[untreated_idx, j].T[None]) ** 2
            score_dist = np.sqrt(score_dist2)

            # make weighted distance matrix
            W = weight * D + (1 - weight) * score_dist  # = N x M

            # get optimal matching
            if strategy == "optimal":
                # TODO: check linear_sum_assignment is correct
                # giving weird results
                row_ind, col_ind = linear_sum_assignment(W)
            elif strategy == "greedy":
                row_ind = np.arange(len(treated_idx))
                col_ind = W.argmin(axis=1)
            else:
                raise NotImplementedError(f"Matching strategy {strategy} not implemented.")

            # get indices of matched treated and untreated
            matched_treated_idx = treated_idx[row_ind]
            matched_untreated_idx = untreated_idx[col_ind]

            # update eneeds match
            needs_match[matched_treated_idx] = False

            # update matches as list of tuples
            matches.extend(zip(matched_untreated_idx, matched_treated_idx))
        return matches
    elif target == "untreated":
        # call needs match with treated indicator flipped
        matches = match(score, distmat, ~treatment, weight, "treated", strategy)

        # reverse the order of the matches
        return [(m[1], m[0]) for m in matches]
    else:
        raise ValueError(f"Target {target} must be treated or untreated.")

def dapsm(
    score: list[float] | np.ndarray,
    distmat: np.ndarray,
    treatment: list[int|bool],
    outcome: list[float],
    weight: float = 0.5,
    target: Literal["population", "treated", "untreated"] = "population",
    strategy: Literal["greedy", "optimal"] = "greedy",
) -> float:
    """
    Compute the causal effect according to dapsm
    """
    if target == "population":
        # compute the effect for treated and untreated
        effect_treated, matches_treated = dapsm(score, distmat, treatment, outcome, weight, "treated", strategy)
        effect_untreated, matches_untreated = dapsm(score, distmat, treatment, outcome, weight, "untreated", strategy)

        # compute the difference in the effects
        wt = np.mean(treatment)
        effect = wt * effect_treated + (1 - wt) * effect_untreated

        return effect, matches_treated + matches_untreated
    else:
        matches = match(score, distmat, treatment, weight, target, strategy)

        # compute the difference in outcomes from treated and untreated
        # each differences in m[1] - m[0] for the matched pairs
        diffs = [outcome[m[1]] - outcome[m[0]] for m in matches]

        # compute the effect
        effect = np.mean(diffs)

        return effect, matches


if __name__ == "__main__":
    # generate some data
    np.random.seed(12345)
    N = 1000
    U = np.random.rand(N)
    distmat = np.abs(U[:, None] - U[None, :])
    score = np.random.uniform(size=N)
    s = 0.5  # spatial confounding intensity
    treatment = np.random.rand(N) < (1 - s ) * score + s * U
    wt = np.mean(treatment)
    err = np.random.rand(N) * 0.1
    outcome = (
        err
        + 0.25 * treatment
        - 0.3 * score
        + s * U # spatial confounding
    )   

    # distance weight
    for dist_wt in np.linspace(0.1, 0.9, 10):
        treated_effect, matches = dapsm(score, distmat, treatment, outcome, weight=dist_wt, target="treated")
        untreated_effect, matches = dapsm(score, distmat, treatment, outcome, weight=dist_wt, target="untreated")
        ate = wt * treated_effect + (1 - wt) * untreated_effect
        print(f"distance weight: {dist_wt}, ate: {ate:.3f}")

