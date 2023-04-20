import pandas as pd
import numpy as np
import sys
from geopy.distance import geodesic
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def CalculateBalance(dtaBef, cols, trt, dtaAfter=None, diff_means = False):
    stand_means = np.empty((2, len(cols)))
    stand_means[:] = np.nan
    row_names = ["Before matching", "After matching"]
    stand_means_df = pd.DataFrame(stand_means, index=row_names, columns=cols)
    
    for col in cols:
        print(col)
        mean_1 = dtaBef.loc[dtaBef[trt] == 1,col].mean(skipna = True)
        mean_0 = dtaBef.loc[dtaBef[trt] == 0,col].mean(skipna = True)
        stand_means_df[col]['Before matching'] = mean_1 - mean_0
        if not diff_means:
            sd_1 = dtaBef.loc[dtaBef[trt] == 1,col].std(skipna = True)
            stand_means_df[col]['Before matching'] = stand_means_df[col]['Before matching'] / sd_1 # is this att specific since only treated sd...
    
    if dtaAfter is not None:
        for col in cols:
            mean_1 = dtaAfter.loc[dtaAfter[trt] == 1,col].mean(skipna = True)
            mean_0 = dtaAfter.loc[dtaAfter[trt] == 0,col].mean(skipna = True)
            stand_means_df[col]['After matching'] = mean_1 - mean_0
            if not diff_means: # need to fix
                sd_1 = dtaAfter.loc[dtaAfter[trt] == 1,col].std(skipna = True)
                stand_means_df[col]['After matching'] = stand_means_df[col]['After matching'] / sd_1
    
    return(stand_means_df)


# In[5]:


def FormDataset(dataset, ignore_cols = None, out_col = None, trt_col = None): # assume dataset input is matrix
    dataset.columns = dataset.columns.astype(str)
    if out_col is not None:
        dataset.columns.values[out_col] = "Y"
    if trt_col is not None:
        dataset.columns.values[int(trt_col)] = 'X'
    if ignore_cols is not None:
        dataset.drop(dataset.columns[ignore_cols], axis=1, inplace=True)

    return(dataset)


# In[6]:


def StandDist(x):
    standx = (x - np.min(x))/(np.max(x) - np.min(x))
    return(standx)    


# In[8]:


# assume treated, control are dataframes
def dist_ps(treated, control, caliper = 0.1, weight = 0.8, coords_columns = None, distance = StandDist, 
            caliper_type = 'DAPS', # 'PS'
            coord_dist = False, matching_algorithm = 'optimal', # 'greedy'
           remove_unmatchables = False):
    if coords_columns is not None:
        treated.columns.values[coords_columns[0]] = 'Longitude'
        treated.columns.values[coords_columns[1]] = 'Latitude'
        control.columns.values[coords_columns[0]] = 'Longitude'
        control.columns.values[coords_columns[1]] = 'Latitude'
    if coord_dist:
        dist_mat = np.zeros((len(treated_coords), len(control_coords)))
        for i, treated_coord in enumerate(treated_coords):
            for j, control_coord in enumerate(control_coords):
                dist_mat[i, j] = geodesic(treated_coord, control_coord).miles
    else: # order is latitude, longitude in python
        dist_mat = cdist(np.column_stack((treated.iloc[:,coords_columns[1]], treated.iloc[:,coords_columns[0]])), 
                         np.column_stack((control.iloc[:,coords_columns[1]], control.iloc[:,coords_columns[0]])) )
    stand_dist_mat = distance(dist_mat)
    
    ps_diff = np.abs(np.subtract.outer(list(treated['prop_scores']), list(control['prop_scores'])))
    
    # DAPS matrix
    dapscore = (1-weight)*stand_dist_mat + weight*ps_diff

    # Creating the matrix we will use for matching, depending on
    # whether the caliper is set on DAPS or on the PS.
    if caliper_type == 'DAPS':
        # M = dapscore + sys.float_info.max*(dapscore > caliper * np.std(dapscore))
        dapscore[dapscore > caliper * np.std(dapscore)] = sys.float_info.max
    elif caliper_type == 'PS':
        # M = dapscore + sys.float_info.max*(ps.diff > caliper * np.std(ps.diff))
        dapscore[ps_diff > caliper * np.std(ps_diff)] = sys.float_info.max
        
    if matching_algorithm == 'greedy': 
        # TO DO
        pass
    else: # optimal matching
        trt_indices, con_indices = linear_sum_assignment(dapscore) # how to work in unmatchables here
        # TO DO: if no matches return empty matrix
        pairs = np.empty((len(trt_indices), 2)) 
        pairs[:,0] = trt_indices
        pairs[:,1] = con_indices
        
    # Save the results
    mat = pd.DataFrame({'match': np.repeat(np.nan, treated.shape[0]),
                    'distance': np.repeat(np.nan, treated.shape[0]),
                    'prop_diff': np.repeat(np.nan, treated.shape[0]),
                    'match_diff': np.repeat(np.nan, treated.shape[0]),
                    'stand_distance': np.repeat(np.nan, treated.shape[0])})
    mat.index = treated.index
    matched_trt = pairs[:, 0]
    matched_con = pairs[:, 1]
    matched_trt = [int(x) for x in matched_trt]
    matched_con = [int(x) for x in matched_con]

    for ii in range(len(matched_trt)):
        wh_trt = matched_trt[ii]
        wh_con = matched_con[ii]

        mat.iloc[wh_trt]['stand_distance'] = stand_dist_mat[wh_trt, wh_con]
        mat.iloc[wh_trt]['prop_diff'] = treated.iloc[wh_trt]['prop_scores'] - control.iloc[wh_con]['prop_scores']
        mat.iloc[wh_trt]['match_diff'] = dapscore[wh_trt, wh_con]
        mat.iloc[wh_trt]['distance'] = dist_mat[wh_trt, wh_con]
        mat.iloc[wh_trt]['match'] = control.index.values[wh_con]
    # Drop any matches with 'infinite' DAPS difference (any matches that shouldn't have made given the caliper)
    mat.drop(mat[mat['match_diff'] == sys.float_info.max].index, inplace = True)

    return(mat)





def WeightChoice(dataset, caliper, caliper_type, coords_cols, cov_cols,
                         cutoff, interval, matching_algorithm, distance = StandDist, trt_col = None, 
                         coord_dist = False,
                         remove_unmatchables = False):
    dataset = FormDataset(dataset, trt_col = trt_col)
    stand_diff = [cutoff + 1]*len(cov_cols)
    weight = np.mean(interval)
    r = {}
    daps_out = dist_ps(treated = dataset.loc[dataset['X'] == 1],
                      control = dataset.loc[dataset['X'] == 0],
                      caliper = caliper, weight = weight, coords_columns = coords_cols,
                      distance = distance, caliper_type = caliper_type,
                      coord_dist = coord_dist, matching_algorithm = matching_algorithm,
                      remove_unmatchables = remove_unmatchables)
    if daps_out.shape[0] > 0:
        pairs_out = [int(x) for x in daps_out['match']] # check the casting
        names = [int(x) for x in daps_out.index] 
        pairs_daps = dataset.iloc[names + pairs_out]
        if len(cov_cols) == 1:
            diff_mat = dataset.iloc[:, cov_cols]
            stand_diff = np.mean(diff_mat.loc[trt]) - np.mean(diff_mat.loc[cnt])
            stand_diff = stand_diff / np.std(diff_mat.loc[trt])
        else:
            diff_mat = pairs_daps.iloc[:, cov_cols]
            trt_cov = pairs_daps[pairs_daps['X'] == 1]
            mean_trt = trt_cov.iloc[:,cov_cols].mean()
            cnt_cov = pairs_daps[pairs_daps['X'] == 0]
            mean_cnt = cnt_cov.iloc[:,cov_cols].mean()
            sd_trt = trt_cov.iloc[:,cov_cols].std()
            stand_diff = (mean_trt - mean_cnt)/sd_trt
 
        r['weight']= weight
        r['stand_diff'] = stand_diff
        r['ind_trt'] = names
        r['ind_cnt'] = pairs_out
        r['pairs'] = pairs_daps 
        r['success'] = False
        
        if not any(abs(stand_diff).gt(cutoff)): # if none is above cutoff
            r['success'] = True
            r['new_interval'] = [interval[0], weight]
        else: # no matched pairs
            r['success'] = False
            r['new_interval'] = [weight, interval[1]]
            print('No matched pairs for weight = ' + str(weight) + '. Trying larger weight. Consider setting remove.unmatchables = True.')
        return(r)
        


# In[10]:


def DAPSopt(dataset, caliper, caliper_type, matching_algorithm, coords_cols, cov_cols, cutoff = 0.1,
                    trt_col = None, w_tol = 0.01, distance = StandDist,
                    quiet = False,
                    coord_dist = False,
                    remove_unmatchables = False):
    dataset = FormDataset(dataset, trt_col = trt_col, out_col = None,
                         ignore_cols = None)
    # To return
    r = {}
    interval = [0,1]
    while (interval[1] - interval[0])>w_tol/2:
        if not quiet:
            print(interval)
        x = WeightChoice(dataset = dataset, caliper = caliper, coords_cols = coords_cols,
                          cov_cols = cov_cols, cutoff = cutoff, interval = interval,
                          distance = distance, caliper_type = caliper_type,
                          coord_dist = coord_dist, matching_algorithm = matching_algorithm,
                          remove_unmatchables = remove_unmatchables)
        interval = x['new_interval']
        if x['success']:
            r['weight'] = x['weight']
            r['pairs'] = x['pairs']
            r['stand_diff'] = x['stand_diff']
            r['ind_trt'] = x['ind_trt']
            r['ind_cnt'] = x['ind_cnt']
    if bool(r): # if r is empty
        print('Standardized balance not achieved. Weight set to 1.')
        daps_out = dist_ps(treated = dataset.loc[dataset['X'] == 1],
                    control = dataset.loc[dataset['X'] == 0],
                    caliper = caliper, weight = 1, coords_columns = coords_cols,
                    distance = distance, caliper_type = caliper_type,
                    coord_dist = coord_dist, matching_algorithm = matching_algorithm,
                    remove_unmatchables = remove_unmatchables)
        r['weight'] = 1
        pairs_out = [int(x) for x in daps_out['match']]
        names = [int(x) for x in daps_out.index]
        # TO DO: na.omit
        pairs_daps = dataset.iloc[names + pairs_out]
        diff_mat = pairs_daps.iloc[:, cov_cols]
        trt_cov = pairs_daps[pairs_daps['X'] == 1]
        mean_trt = trt_cov.iloc[:,cov_cols].mean()
        cnt_cov = pairs_daps[pairs_daps['X'] == 0]
        mean_cnt = cnt_cov.iloc[:,cov_cols].mean()
        sd_trt = trt_cov.iloc[:,cov_cols].std()
        stand_diff = (mean_trt - mean_cnt)/sd_trt
        r['stand_diff'] = stand_diff
        r['pairs'] = pairs_daps
        r['ind_trt'] = names
        r['ind_cnt'] = pairs_out
        
    return(r)
