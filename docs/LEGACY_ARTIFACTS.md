# Legacy Artifacts and Current Analysis

The repository was originally created from an earlier January-2026 implementation. Those files are retained for project history, but they should not be treated as the current scientific result.

## Legacy root artifacts

- `AMI_Mortality_Prediction.ipynb`
- `Technical_Validation_Report.pdf`
- `eda.png`
- `predictors.png`
- `results.png`
- `roc_curve.png`

The earlier implementation used a weighted logistic-regression workflow and reported approximately:

- ROC-AUC: 0.758 / 0.76
- Brier score: 0.169
- strongest predictors: ST elevation, Killip class and age

That workflow is superseded by the current repeated nested-validation analysis.

## Current canonical analysis

Use the following for the current project:

- `src/isds_option_a_pipeline.py`
- `notebooks/ISDS_Option_A_Full_Workflow.ipynb`
- `docs/METHODOLOGY.md`
- `docs/RESULTS.md`
- root `README.md`

Current selected specification:

- ridge-penalised logistic regression;
- `C = 0.1`;
- no class rebalancing;
- repeated stratified 5-fold outer CV × 5 repeats;
- stratified 4-fold inner CV;
- tuning by log loss;
- ROC-AUC 0.7881;
- PR-AUC 0.2783;
- Brier 0.0556;
- log loss 0.2070;
- calibration intercept 0.1277;
- calibration slope 1.0580.

The older files are intentionally not deleted because they document the evolution of the project. Their metrics should not be mixed with the current results.
