import pandas as pd
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment

def StandDist(x):
    standx = (x - np.min(x))/(np.max(x) - np.min(x))
    return(standx)    


# Modify dist_ps so it just takes in a distance matrix
def dist_ps(dist_mat, ps_diff, caliper, caliper_type, distance = StandDist,
            weight=0.8,
            matching_algorithm='optimal'):
    # This line is not needed if distances are already normalized between 0 and 1
    stand_dist_mat = distance(dist_mat)

    # DAPS matrix
    dapscore = (1-weight)*stand_dist_mat + weight*ps_diff

    # Creating the matrix we will use for matching, depending on
    # whether the caliper is set on DAPS or on the PS.
    if caliper_type == 'DAPS':
        dapscore[dapscore > caliper * np.std(dapscore)] = sys.float_info.max
    elif caliper_type == 'PS':
        dapscore[ps_diff > caliper * np.std(ps_diff)] = sys.float_info.max

    if matching_algorithm == 'greedy':
        # TO DO
        pass
    else:  # optimal matching
        trt_indices, con_indices = linear_sum_assignment(dapscore)

        # TO DO: if no matches return
        if len(trt_indices) == 0:
            print('No matches found.')
            return (None)
        else:
            pairs = list(zip(trt_indices, con_indices))
            match_diff = [dapscore[trt][con] for trt, con in pairs]

            # Drop matches that have dapscore inf.
            pairs = [pair for pair, score in zip(
                pairs, match_diff) if score != sys.float_info.max]
            match_diff = [dapscore[trt][con] for trt, con in pairs]

            distances = [dist_mat[trt][con] for trt, con in pairs]
            prop_diff = [ps_diff[trt][con] for trt, con in pairs]
            stand_distance = [stand_dist_mat[trt][con] for trt, con in pairs]

            return pairs, match_diff, distances, prop_diff, stand_distance


def WeightChoice(treated, control, dist_mat, ps_diff, caliper, caliper_type, cov_cols,
                 cutoff, interval, matching_algorithm, distance=StandDist):
    weight = np.mean(interval)
    daps_out = dist_ps(dist_mat, ps_diff, caliper, caliper_type, distance,
                       weight=weight,
                       matching_algorithm=matching_algorithm)
    pairs = daps_out[0]

    if pairs:  # if matches exist in daps_out
        #trt, cnt = pairs  # check this
        trt = [pair[0] for pair in pairs]
        cnt = [pair[1] for pair in pairs]
        mean_trt = np.mean(treated[trt][:,cov_cols], axis=0)
        mean_cnt = np.mean(control[cnt][:,cov_cols], axis=0)
        sd_trt = np.std(treated[trt][:,cov_cols], axis=0)
        stand_diff = (mean_trt - mean_cnt)/sd_trt

    if not np.any(abs(stand_diff) > cutoff):  # if none is above cutoff
        success = True
        new_interval = [interval[0], weight]
    else:
        success = False
        new_interval = [weight, interval[1]]
        print('No matched pairs for weight = ' +
              str(weight) + '. Trying larger weight.')
    return weight, new_interval, stand_diff, pairs, success


def DAPSopt(treated, control, dist_mat, ps_diff, caliper, caliper_type, matching_algorithm, cov_cols, cutoff=0.1,
            w_tol=0.01, distance=StandDist,
            quiet=False):
    interval = [0, 1]
    while (interval[1] - interval[0]) > w_tol/2:
        if not quiet:
            print(interval)
        x = WeightChoice(treated=treated, control=control, dist_mat=dist_mat, ps_diff=ps_diff,
                         caliper=caliper, caliper_type=caliper_type, cov_cols=cov_cols,
                         cutoff=cutoff, interval=interval, matching_algorithm=matching_algorithm, distance=distance)
        interval = x[1]

        success = x[4]
        if success:
            weight = x[0]
            pairs = x[3]
            stand_diff = x[2]

    if not success:
        print('Standardized balance not achieved. Weight set to 1.')
        weight = 1
        daps_out = dist_ps(dist_mat=dist_mat, ps_diff=ps_diff, caliper=caliper, caliper_type=caliper_type, distance=distance,
                           weight=weight,
                           matching_algorithm=matching_algorithm)
        pairs = daps_out[0]

        trt = [pair[0] for pair in pairs]
        cnt = [pair[1] for pair in pairs]
        mean_trt = np.mean(treated[trt][:,cov_cols], axis=0)
        mean_cnt = np.mean(control[cnt][:,cov_cols], axis=0)
        sd_trt = np.std(treated[trt][:,cov_cols], axis=0)
        stand_diff = (mean_trt - mean_cnt)/sd_trt

    return weight, stand_diff, pairs
