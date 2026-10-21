#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:18:37 2026
@author: renata

This module contains the core Sequential Feature Selection (SFS) logic.
It can be used with ANY scikit-learn regressor.
"""

from sklearn.utils import shuffle
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import LeaveOneOut, KFold
import pandas as pd


def ml_regressors(data, k_feat, regressor, n_samples):
    """Core SFS logic.
    
    Performs forward Sequential Feature Selection (SFS) on the dataset.
    Returns the fitted SFS object containing all subsets and metrics.
    """
    # Extract feature columns (everything except the last column = target)
    feature_cols = data.columns.drop(data.columns[-1])
    X = data[feature_cols].copy()
    y = data[data.columns[-1]].values

    # Reproducible shuffle (always same order for fair comparison)
    X, y = shuffle(X, y, random_state=42)    

    # Create SFS object
    sfs = SFS(
        regressor(),                     # any sklearn regressor passed as class
        k_features=k_feat,               # how many features to select
        forward=True,                    # forward selection (starts with 0 features)
        floating=False,                  # Set to True for SFFS (much better results!)
        scoring='r2',                    # score to maximize
        cv=0,                            # cv=0 → no cross-validation (fast on small data)
        n_jobs=1,                        # 1 = safe for parallel outer loop; change to -1 if needed
        verbose=0
    )
    sfs.fit(X, y)                        # Run the actual selection
    return sfs


def sfs_tab(data, k_feat, regressor, n_samples):
    """Builds clean results table.
    
    Takes the SFS object and creates a nice pandas DataFrame with:
    - Number of features
    - Selected feature indices/names
    - Incremental new feature
    - R² and standard error
    """
    sfs = ml_regressors(data, k_feat, regressor, n_samples)   # get fitted SFS
    results = sfs.subsets_                                    # dictionary of all subsets

    # Efficient way to track which feature was added at each step
    data_list = []
    prev_features = set()

    for i in range(1, k_feat + 1):
        current = results[i]
        # Identify the NEW feature added in this step
        new_feature = list(set(current['feature_names']) - prev_features)[0] if prev_features else current['feature_names'][0]
        prev_features = set(current['feature_names'])

        data_list.append({
            'Number of Features': i,
            'Selected Features': str(current['feature_idx']),
            'Selected Features Names': str(current['feature_names']),
            'Index Names': new_feature
        })

    df = pd.DataFrame(data_list)

    # Add R² and std_err from SFS metric dict
    metrics = pd.DataFrame.from_dict(sfs.get_metric_dict(), orient='index').reset_index(drop=True)
    df1 = pd.concat([df, metrics[['avg_score', 'std_err']]], axis=1)
    df1 = df1.rename(columns={'avg_score': 'R-squared', 'std_err': 'Std. Error'})

    return df1


def run_ml(data_x, data_y, n_samples, k_feat, regressor):
    """Main function – entry point for the whole module.
    
    Combines X and Y, runs SFS, adds delta R² column, and returns clean table.
    """
    # Combine inputs + output and slice to desired sample size
    data = pd.concat([data_x, data_y], axis=1).iloc[:n_samples].reset_index(drop=True)
    
    df_sfs = sfs_tab(data, k_feat, regressor, n_samples)

    if not df_sfs.empty:
        r_list = df_sfs["R-squared"].tolist()
        # Calculate incremental delta R² (how much each new feature improves the score)
        r_col = [round(r_list[0], 4)] + [
            round(r_list[i] - r_list[i-1], 4) for i in range(1, len(r_list))
        ]
        df_sfs["R2"] = r_col          # "R2" = delta R² contribution of each step

    return df_sfs