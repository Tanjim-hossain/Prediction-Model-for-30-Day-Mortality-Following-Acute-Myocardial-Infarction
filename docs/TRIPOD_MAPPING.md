# Prediction-Model Reporting Traceability

This document maps major prediction-model reporting elements to the repository. It is intended as a transparency aid rather than as a statement of external validation or clinical readiness.

| Reporting element | Repository evidence |
|---|---|
| Data preprocessing / cleaning | Complete executed notebook; `METHODOLOGY.md`; `DATA_DICTIONARY.md` |
| Class imbalance | Ridge no-rebalancing, class-weighting and random-oversampling experiments |
| Missing data | Fold-specific imputation plus missing-indicator sensitivity |
| Statistical / ML techniques | Elastic Net, ridge, standard logistic, Random Forest, Gradient Boosting, intercept-only reference |
| Complexity control | Elastic Net tuning and ridge shrinkage |
| Hyperparameter tuning | 4-fold inner stratified CV nested inside repeated outer CV |
| Performance estimation | ROC-AUC, PR-AUC, Brier, log loss, calibration |
| Uncertainty | 2,000 patient-level bootstrap resamples |
| Ablation studies | class weighting, random oversampling, Killip coding, missing indicators |
| Model interpretation | coefficients, odds ratios and outer-fold permutation importance |
| Internal validation | 5-fold stratified outer CV × 5 repeats |
| Clinical-use analysis | threshold summaries and decision-curve analysis |
| Computational record | full executed notebook plus reusable source modules |
| Reproducibility | fixed seed, environment record, result tables, tests and CI |

No external validation dataset is included, and the repository states that limitation explicitly.
