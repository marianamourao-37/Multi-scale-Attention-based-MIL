# external imports 

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def stratified_train_val_split(dev_df, frac_split = 0.2, shuffle = True, args = None):
    
    # shuffle data
    if shuffle:
        dev_df = dev_df.sample(frac = 1).reset_index(drop=True)

    group_patient_id_list = np.array(dev_df['patient_id'].values)
    
    sgkf = StratifiedGroupKFold(n_splits = int(1/frac_split))

    train_idxs, val_idxs = next(sgkf.split(dev_df, dev_df[args.label].values, groups = group_patient_id_list))

    train_patients, val_patients = group_patient_id_list[train_idxs], group_patient_id_list[val_idxs]

    # making sure that patients from train and test datasets do not overlap
    assert len(set(train_patients) & set(val_patients)) == 0
    
    train_df = dev_df.iloc[train_idxs]
    val_df = dev_df.iloc[val_idxs]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def generator_cross_val_folds(dev_df, k_folds = 5, label_name = '', shuffle = True):

    # shuffle data
    if shuffle:
        dev_df = dev_df.sample(frac = 1).reset_index(drop=True)
        
    sgkf_cross_val = StratifiedGroupKFold(n_splits = k_folds)
    
    group_patient_id_list = np.array(dev_df['patient_id'].values)

    for fold_idx, (train_idxs, val_idxs) in enumerate(sgkf_cross_val.split(dev_df, dev_df[label_name].values, groups = group_patient_id_list)):

        train_patients, val_patients = group_patient_id_list[train_idxs], group_patient_id_list[val_idxs]

        # making sure that patients from dev and val datasets do not overlap
        assert len(set(train_patients) & set(val_patients)) == 0

        train_fold_df = dev_df.iloc[train_idxs]
        val_fold_df = dev_df.iloc[val_idxs]

        yield train_fold_df.reset_index(drop=True), val_fold_df.reset_index(drop=True)
