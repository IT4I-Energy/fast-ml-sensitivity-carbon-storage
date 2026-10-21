# Reproduction of the OPM Flow ML / Sobol sensitivity figures

## Environment
A fresh conda environment `opmflow_ml` (Python 3.11) was created and the
packages from the repository README were installed via pip:

```bash
conda create -y -n opmflow_ml python=3.11
conda activate opmflow_ml
pip install pandas scikit-learn mlxtend SALib joblib matplotlib numpy
```

Resolved versions: numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0,
mlxtend 0.25.0, SALib 1.5.2, scipy 1.17.1, matplotlib 3.11.1, joblib 1.5.3.

## Note on the original scripts
`ml_featuresVSsalib_allyears_comments.py` imports
`from ml_reg_sintef_parallel_module import sfs_tab, run_ml`, but the module
in the repository is named `ml_reg_parallel_module.py`. The module was copied
here under the expected name `ml_reg_sintef_parallel_module.py` so the code
runs unchanged.

## Driver
`run_all.py` runs the original logic across the QMC sample sizes and
regressors that appear in the paper, writing one results CSV and the
R² / S1 / ST evolution plots per configuration into `outputs/`:

- LinearRegression: n = 10, 32, 64, 100, 128, 256, 512, 1024
- GradientBoostingRegressor: n = 10, 100, 1024

For sample sizes that are not a multiple of 2·d+2 = 16 (i.e. n = 10, 100),
the SALib Sobol estimator is undefined, so only the ML (ΔR²) plot is produced;
this matches the figure set shipped in the repository.

```bash
conda activate opmflow_ml
python run_all.py            # all configurations
python run_all.py 1024       # only the n=1024 configurations
```

## Verification
The reproduced figures match the reference figures in `../Figures_Sobol` and
`../Figures_ML`. In particular the Sobol estimator at n = 32 reproduces the
documented instability (first-order S1 of *peshale* ≈ 2.2 and of *pefeeder*
≈ −0.65), and the ML estimator recovers the *posand → peshale* cross-over
down to n = 10. At year 50 / n = 1024 the ML increment for *posand* (0.7516)
matches its Sobol first-order index (0.7507), confirming the theoretical
equivalence for the dominant additive term.
