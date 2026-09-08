# Official Option A / TRIPOD traceability

The course assignment asks that reporting be guided by TRIPOD elements 8–20 and explicitly cover preprocessing, missing data, statistical tests, modelling techniques, feature selection, hyperparameter tuning, model performance, ablations, interpretation, decision justification, internal validation and source code.

| Requirement | Repository evidence |
|---|---|
| Preprocessing / cleaning | Executed notebook Sections 3–5; `docs/METHODOLOGY.md`; `docs/DATA_DICTIONARY.md` |
| Class imbalance | Executed notebook Sections 8.3–8.5; `results/tables/class_imbalance_ablation.csv` |
| Missing data | Fold-specific imputation in notebook and pipeline; missing-indicator sensitivity |
| Statistical tests | Explicitly documents that no hypothesis-test-based predictor screening was used |
| Statistical / ML techniques | Elastic-net, ridge, standard logistic, Random Forest, Gradient Boosting, intercept-only reference |
| Feature selection / complexity control | Elastic-net tuning; final ridge shrinkage rather than a hard sparse subset |
| Hyperparameter tuning | 4-fold inner stratified CV nested inside repeated outer CV, tuned by log loss |
| Performance estimation | ROC-AUC, PR-AUC, Brier, log loss, calibration, repeated-CV stability |
| Uncertainty | 2,000 patient-level bootstrap resamples of final OOF predictions |
| Ablation studies | Class weighting, random oversampling, Killip coding, missing indicators |
| Model interpretation | Penalised coefficients/ORs plus outer-fold raw-predictor permutation importance |
| Internal validation | 5-fold stratified outer CV × 5 repeats |
| Clinical usefulness | Threshold summaries and decision-curve analysis with explicit non-causal caveats |
| Source code | Full executed notebook + `src/` scripts |
| Reproducibility | Fixed seed, environment record, result tables, CI tests |

## TRIPOD-oriented coverage

The project documents the data and participants available in the assignment, outcome and predictors, sample-size/event context, missing-data handling, model-building procedure, internal validation design, performance measures, model specification, model performance, model-use considerations and limitations. No external validation was available, and the repository states that limitation explicitly.
