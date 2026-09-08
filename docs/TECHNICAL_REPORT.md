# Technical report

## Development and internal validation of a 30-day mortality prediction model after acute myocardial infarction

### Objective

Develop and internally validate a multivariable prediction model for 30-day mortality after acute myocardial infarction using the supplied 785-patient Option A dataset.

### Data and quality control

The dataset contains 785 observations, 52 deaths and 17 candidate predictors. Seven source cells were explicitly missing. The audit also identified three hypotension values encoded as `Unknown`, 14 invalid `Killip_class=-1` values and two unmistakable height-entry errors (`1.75`, `1690`). Cleaning was limited to auditable recoding/corrections and retained all patients. After recoding, 24 predictor cells were missing.

### Preprocessing

Continuous/ordinal predictors were median-imputed and standardised. Binary predictors were most-frequent imputed. Smoking was imputed and one-hot encoded. All transformations were estimated inside training folds. Killip class was treated as ordinal in the primary model and categorical in a sensitivity analysis.

### Model development

A repeated nested validation design was used: five stratified outer folds repeated five times, and four-fold stratified inner CV for hyperparameter tuning by log loss. The initial penalised logistic search tuned `C ∈ {0.01,0.1,1}` and `l1_ratio ∈ {0,0.25,0.5,0.75,1}`. Twenty of 25 outer folds selected `C=0.1` and `l1_ratio=0`, supporting the ridge endpoint.

Candidate models were ridge logistic regression, standard logistic regression, Random Forest, Gradient Boosting and an intercept-only prevalence reference.

### Candidate-model performance

| Model | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Ridge logistic | 0.7881 | 0.2783 | 0.0556 | 0.2070 |
| Random Forest | 0.7806 | 0.2462 | 0.0564 | 0.2094 |
| Standard logistic | 0.7726 | 0.2816 | 0.0558 | 0.2127 |
| Gradient Boosting | 0.7509 | 0.2379 | 0.0571 | 0.2160 |
| Intercept-only reference | 0.4638 | 0.0631 | 0.0619 | 0.2438 |

Ridge was selected for its overall balance of discrimination, probability accuracy, calibration and validation stability.

### Class imbalance

Matched ridge ablations showed that class weighting and random oversampling worsened probability quality. No rebalancing achieved Brier 0.0556 and log loss 0.2070, compared with 0.1711/0.5138 for class weighting and 0.1700/0.5108 for random oversampling.

### Calibration and uncertainty

The selected ridge model had calibration intercept 0.1277 and slope 1.0580. Mean predicted risk (6.63%) closely matched observed mortality (6.62%). Two thousand patient-level bootstrap resamples produced a ROC-AUC 95% CI of 0.7236–0.8478 and PR-AUC CI of 0.1741–0.4054.

### Sensitivity analyses

Categorical Killip coding slightly worsened performance (ROC-AUC 0.7786, PR-AUC 0.2233, log loss 0.2103). Adding missingness indicators produced essentially no change (ROC-AUC 0.7889, log loss 0.2069). The simpler primary preprocessing was retained.

### Thresholds and decision curve

At 5%, 10%, 15% and 20% risk thresholds, sensitivities were 78.8%, 53.8%, 38.5% and 23.1%, while specificities were 61.9%, 83.1%, 92.5% and 95.6%. The threshold grid is descriptive rather than a recommended clinical policy. Internally validated decision-curve net benefit was positive across the explored range, but no causal treatment-benefit claim is made.

### Interpretation

The largest absolute final-model coefficients were for age, heart-rate indicator, time to relief, previous MI, Killip class and anterior infarct location. Outer-fold permutation analysis ranked age as the dominant predictive contributor, followed by Killip class, heart-rate indicator, time to relief and weight.

### Conclusion

Ridge-penalised logistic regression without class rebalancing provided the strongest overall internally validated performance among the evaluated candidates. The model remains an academic internal-validation result. External validation is required before any clinical use.

### Computational record

See `notebooks/01_Full_Analysis_Executed.ipynb` for the complete original code and preserved outputs, `src/` for reusable source code and `results/tables/` for machine-readable numerical results.
