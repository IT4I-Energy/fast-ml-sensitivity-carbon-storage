#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduction driver for the OPM Flow ML / Sobol sensitivity paper.

Runs the ORIGINAL logic (from ml_featuresVSsalib_allyears_comments.py +
ml_reg_sintef_parallel_module.py) across several QMC sample sizes and
regressors, writing a results CSV and the R2 / S1 / ST evolution plots
into ./outputs.

For sample sizes that are not a multiple of (2*num_vars+2) the SALib Sobol
estimator is undefined; in that case only the ML (delta-R2) plot is produced
and the Sobol columns are left as NaN.  This mirrors the fact that the
repository ships ML figures for n=10 and n=100 but Sobol figures only for
n in {32,64,128,256,512,1024}.
"""
import os, sys, glob, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from SALib.analyze import sobol
from joblib import Parallel, delayed

from ml_reg_sintef_parallel_module import run_ml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
DATA_X = "/home/renata/Old_lenovo/across/python/input_params271123_sent.csv"
Y_GLOB = "/home/renata/Old_lenovo/across/python/coarse3x3/trapped_co2/*.csv"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- Sobol problem
problem = {
    "num_vars": 7,
    "names": ["tegrad", "pesand", "peshale", "pefeeder", "posand", "totopsur", "dummy"],
    "bounds": [[30, 40], [1100, 5000], [0.00075, 0.0015], [1100, 5000],
               [0.27, 0.4], [-10, 10], [0, 1]],
}
SOBOL_MULT = 2 * problem["num_vars"] + 2   # = 16 evaluations per base sample

# ---------------------------------------------------------------- experiment grid
CONFIGS = [
    (LinearRegression,          [10, 32, 64, 100, 128, 256, 512, 1024]),
    (GradientBoostingRegressor, [10, 100, 1024]),
]
K_FEATURES = 6

data_x = pd.read_csv(DATA_X)
if "Unnamed: 0" in data_x.columns:
    del data_x["Unnamed: 0"]
files = sorted(glob.glob(Y_GLOB))
print(f"Loaded {data_x.shape[0]} input rows, {len(files)} yearly output files.")


def process_one_file(name, n_samples, regressor):
    data_y = pd.read_csv(name)
    year = int(name.split("/")[-1].split(".")[0].split("_")[-1])
    if "sample" in data_y.columns:
        del data_y["sample"]

    # ML: sequential forward selection + incremental R2
    df_sfs = run_ml(data_x, data_y, n_samples, K_FEATURES, regressor)
    df_sfs["year"] = year
    df_sfs = df_sfs.set_index("Index Names")

    # Sobol on the same Y values (only defined when len divisible by 2D+2)
    if n_samples % SOBOL_MULT == 0:
        try:
            data_y_sobol = data_y.loc[: n_samples - 1]
            Si = sobol.analyze(problem, data_y_sobol["trapped_co2"].values,
                               print_to_console=False)
            total_Si, S1, S2 = Si.to_df()
            S1 = S1.drop("S1_conf", axis=1)
            total_Si = total_Si.drop("ST_conf", axis=1)
            df_sfs = df_sfs.join(S1, how="left").join(total_Si, how="left")
        except Exception as e:                     # pragma: no cover
            print(f"  Sobol failed year {year}: {e}")
            df_sfs["S1"] = float("nan")
            df_sfs["ST"] = float("nan")
    else:
        df_sfs["S1"] = float("nan")
        df_sfs["ST"] = float("nan")
    return df_sfs


def make_plot(df, col, ylabel, title, fname, params, r_max_x, r_max_y):
    if col not in df.columns or df[col].notna().sum() == 0:
        return False
    plt.figure()
    for p in params:
        sub = df.loc[df.index == p]
        plt.plot(sub["year"], sub[col], label=p)
    plt.plot(r_max_x, r_max_y, linestyle="--", label="total")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Years")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), ncol=1)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def run_config(regressor, n_samples):
    rname = regressor.__name__
    tag = f"{rname}_{n_samples}"
    print(f"\n=== {tag} ===")
    df_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_one_file)(name, n_samples, regressor) for name in files
    )
    df = pd.concat(df_list, ignore_index=False)
    df = df.sort_values(by=["year", "R2"], ascending=[True, False])
    df.to_csv(os.path.join(OUTDIR, f"results_{tag}.csv"))

    params = list(data_x.columns)
    years = sorted(df["year"].unique())
    r_max_y = [df.loc[df["year"] == y, "R-squared"].max() for y in years]

    made = []
    if make_plot(df, "R2", "R-squared (incremental)",
                 f"R-squared Evolution in time for Secondary Trapped CO2\n{n_samples} samples, {rname}",
                 os.path.join(OUTDIR, f"r2_evolution_{tag}.png"),
                 params, years, r_max_y):
        made.append("r2")
    if make_plot(df, "S1", "S1 Sobol",
                 f"First-order Sobol (S1) Evolution in time\n{n_samples} samples, {rname}",
                 os.path.join(OUTDIR, f"S1_evolution_{tag}.png"),
                 params, years, r_max_y):
        made.append("S1")
    if make_plot(df, "ST", "ST Sobol",
                 f"Total-order Sobol (ST) Evolution in time\n{n_samples} samples, {rname}",
                 os.path.join(OUTDIR, f"ST_evolution_{tag}.png"),
                 params, years, r_max_y):
        made.append("ST")
    print(f"  wrote results_{tag}.csv and plots: {made}")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for regressor, sizes in CONFIGS:
        for n in sizes:
            if only and only not in (regressor.__name__, str(n)):
                continue
            run_config(regressor, n)
    print("\nAll configurations completed.")
