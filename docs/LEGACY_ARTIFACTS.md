# Legacy analysis history

This repository originally contained an earlier January-2026 implementation based on weighted logistic regression. That analysis reported approximately ROC-AUC 0.758/0.76 and Brier score 0.169 and used older root-level notebook/report/figure files.

Those superseded artifacts have been removed from the **current repository tree** to prevent old and new results from being mixed in a portfolio review. They remain recoverable from Git history.

## Current canonical analysis

Use these files for the present project:

- `notebooks/01_Full_Analysis_Executed.ipynb` — GitHub portfolio edition of the executed final analysis and its key preserved outputs;
- `notebooks/02_Reproducible_Workflow.ipynb` — clean script-backed workflow;
- `src/isds_option_a_pipeline.py` — reusable full pipeline;
- `src/elastic_net_screen.py` — reproducible initial elastic-net development screen;
- `results/tables/` — curated machine-readable results from the final executed analysis;
- `docs/METHODOLOGY.md` and `docs/RESULTS.md` — technical documentation;
- `docs/MODEL_CARD.md` — intended use and limitations;
- root `README.md` — portfolio overview.

Current selected specification:

- ridge-penalised logistic regression;
- `C = 0.1`;
- no class rebalancing;
- stratified 5-fold outer CV × 5 repeats;
- stratified 4-fold inner CV;
- tuning by log loss;
- ROC-AUC 0.7881;
- PR-AUC 0.2783;
- Brier score 0.0556;
- log loss 0.2070;
- calibration intercept 0.1277;
- calibration slope 1.0580.

Do not use the January-2026 metrics as the current project result.
