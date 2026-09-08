# Results

This directory contains version-controlled result artifacts from the complete executed analysis.

## Figures

`results/figures/` contains the original Matplotlib figures extracted directly from `notebooks/01_Complete_Executed_Analysis.ipynb`:

- `outcome_and_missingness.png`
- `calibration_curve.png`
- `decision_curve.png`
- `permutation_importance.png`

These are the same executed analytical figures preserved inside the notebook rather than redrawn substitutes.

## Tables

`results/tables/` contains compact machine-readable summaries for rapid inspection of model comparison, calibration, bootstrap intervals, class-imbalance ablations, sensitivity analyses, threshold metrics, Elastic Net selection frequency, coefficients, repeated-CV stability and permutation importance.

The complete computational record—including all modelling code, intermediate outputs, tuning steps, assertions and model-packaging steps—remains `notebooks/01_Complete_Executed_Analysis.ipynb`.

Runtime artifacts from a fresh local/Kaggle execution are written to `outputs/`, which is ignored by Git to avoid mixing generated runs with the version-controlled reference results.
