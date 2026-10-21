#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:11:12 2026

@author: renata
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:59:52 2026
@author: renata

Main script: Parallel SFS + Sobol analysis for ALL years.
Compares ML feature importance (incremental R²) with Sobol indices (S1 & ST).
Generates CSV + 3 evolution plots.
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.discovery import all_estimators
from SALib.analyze import sobol
import warnings
from joblib import Parallel, delayed

# Import our cleaned SFS functions
from ml_reg_sintef_parallel_module import sfs_tab, run_ml


# ===================================================================
# Sobol problem definition (must match your 7 input parameters)
# ===================================================================
problem = {
    'num_vars': 7,
    'names': ['tegrad', 'pesand', 'peshale', 'pefeeder', 'posand', 'totopsur', 'dummy'],
    'bounds': [[30, 40],      # temperature gradient
               [1100, 5000],  # permeability sand
               [0.00075, 0.0015],  # permeability shale
               [1100, 5000],  # permeability feeder
               [0.27, 0.4],   # porosity sand
               [-10, 10],     # topography
               [0, 1]]        # dummy variable (should stay ~0)
}

# ===================================================================
# Configuration – CHANGE THESE ONLY
# ===================================================================
estimators = all_estimators(type_filter="regressor")
data_x = pd.read_csv('/home/renata/Old_lenovo/across/python/input_params271123_sent.csv') 
del data_x['Unnamed: 0']

warnings.filterwarnings('ignore')

files = glob.glob('/home/renata/Old_lenovo/across/python/coarse3x3/trapped_co2/*.csv')

n_samples = 1024          # Number of samples to use (32/64/128/256/512/1024)
k_features = 6            # Max features to select with SFS max. 7
regressor = LinearRegression   # ← CHANGE HERE: LinearRegression / GradientBoostingRegressor / GaussianProcessRegressor
r_name = str(regressor)        # for filename

df_all_regressors = pd.DataFrame()
df_all_sorted = pd.DataFrame()


def process_one_file(name, data_x, n_samples, k_features, regressor):
    """Process ONE yearly CSV file (designed for parallel execution).
    
    Steps inside:
    1. Load trapped_co2 for that year
    2. Run SFS + delta R²
    3. Run Sobol analysis on the same data
    4. Merge everything and return one DataFrame
    """
    data_y = pd.read_csv(name)
    year = int(name.split("/")[-1].split(".")[0].split("_")[-1])
    del data_y['sample']
    
    # Run ML feature selection
    df_sfs = run_ml(data_x, data_y, n_samples, k_features, regressor)
    df_sfs["year"] = year

    # Sobol analysis on the same Y values
    data_y_sobol = data_y.loc[:n_samples-1]
    Si = sobol.analyze(problem, data_y_sobol['trapped_co2'].values, print_to_console=False)
    total_Si, S1, S2 = Si.to_df()
    S1 = S1.drop('S1_conf', axis=1)    
    total_Si = total_Si.drop('ST_conf', axis=1)  
    
    # Merge SFS table with Sobol indices using feature name as index
    df_sfs = df_sfs.set_index('Index Names')
    df_sfs = df_sfs.join(S1, how='left') 
    df_sfs = df_sfs.join(total_Si, how='left') 
    
    print(f"✓ Processed {name} (year {year})")
    return df_sfs


# ===================================================================
# PARALLEL EXECUTION
# ===================================================================
print(f"Starting parallel processing of {len(files)} files using {regressor.__name__}...")

# Run all years at once on all CPU cores
df_list = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_one_file)(name, data_x, n_samples, 6, regressor)
    for name in files
)

# Combine all years into one big table
df_all_regressors = pd.concat(df_list, ignore_index=False)

print(f"Parallel processing completed! Final shape: {df_all_regressors.shape}")

# ===================================================================
# Final sorting + output
# ===================================================================
regressor_name = str(regressor()).strip(')').strip('(')

df_all_sorted = df_all_regressors.sort_values(by=["year", 'R2'], ascending=[True, False])
df_all_sorted.to_csv(f'results_{r_name}_{n_samples}.csv')

# ===================================================================
# PLOTTING: R² evolution, S1, and ST over time
# ===================================================================
params = data_x.columns
r_max = [df_all_sorted.loc[df_all_sorted['year']==i, 'R-squared'].max() for i in df_all_sorted.year.unique()]

# Plot 1: Incremental R² per feature
fig = plt.Figure()
for p in params:
    x = df_all_sorted.loc[df_all_sorted.index == p, 'year']
    y = df_all_sorted.loc[df_all_sorted.index == p, 'R2']
    plt.plot(x, y, label=p)

plt.plot(df_all_sorted.year.unique(), r_max, linestyle='--', label='total')
plt.title(f"R-squared Evolution in time for Secondary Trapped CO2\n{n_samples} samples, {r_name}")
plt.ylabel("R-squared")
plt.xlabel("Years")
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), ncol=1)
plt.savefig(f'r2_evolution_{r_name}_{n_samples}.png',dpi=150,bbox_inches='tight')
plt.show()

# Plot 2: First-order Sobol (S1)
for p in params:
    x = df_all_sorted.loc[df_all_sorted.index == p, 'year']
    y = df_all_sorted.loc[df_all_sorted.index == p, 'S1']
    plt.plot(x, y, label=p)

plt.plot(df_all_sorted.year.unique(), r_max, linestyle='--', label='total')
plt.title(f"First-order Sobol (S1) Evolution in time\n{n_samples} samples, {r_name}")
plt.ylabel("S1 Sobol")
plt.xlabel("Years")
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), ncol=1)
plt.savefig(f'S1_evolution_{r_name}_{n_samples}.png',dpi=150,bbox_inches='tight')

plt.show()

# Plot 3: Total-order Sobol (ST)
for p in params:
    x = df_all_sorted.loc[df_all_sorted.index == p, 'year']
    y = df_all_sorted.loc[df_all_sorted.index == p, 'ST']
    plt.plot(x, y, label=p)

plt.plot(df_all_sorted.year.unique(), r_max, linestyle='--', label='total')
plt.title(f"Total-order Sobol (ST) Evolution in time\n{n_samples} samples, {r_name}")
plt.ylabel("ST Sobol")
plt.xlabel("Years")
plt.grid(True)
plt.legend(bbox_to_anchor=(1, 1), ncol=1)
plt.savefig(f'ST_evolution_{r_name}_{n_samples}.png',dpi=150,bbox_inches='tight')
plt.show()